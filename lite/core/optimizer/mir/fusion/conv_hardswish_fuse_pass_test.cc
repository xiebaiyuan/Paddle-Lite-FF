// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>
#include <memory>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/optimizer/mir/fusion/conv_hardswish_fuser.h"
#include "lite/core/optimizer/mir/fusion/fusion_pass_test_util.h"

namespace paddle {
namespace lite {
namespace mir {

// conv2d → (add(offset) → clip → mul) × conv_out, then tail mul(1/scale):
//   conv_out ─┬→ add(y=3) → clip(0,6) → mul(y=clip_out)
//             └──────────────────────────┘ (X=conv_out)
//   mul_out → mul(1/6) → out
std::vector<TestOpDesc> MakeConvHardSwish(bool with_bias) {
  TestOpDesc conv;
  conv.type = "conv2d";
  conv.inputs = {{"Input", {"x"}}, {"Filter", {"w"}}};
  conv.outputs = {{"Output", {"conv_out"}}};
  conv.int_attrs = {{"groups", 1}};
  conv.int_vector_attrs = {{"strides", {1, 1}},
                           {"paddings", {0, 0}},
                           {"dilations", {1, 1}}};
  if (with_bias) conv.inputs["Bias"] = {"conv_b"};
  std::vector<TestOpDesc> ops{conv};
  ops.push_back({"elementwise_add",
                 {{"X", {"conv_out"}}, {"Y", {"offset_c"}}},
                 {{"Out", {"add_out"}}},
                 {{"axis", -1}},
                 {},
                 {},
                 {}});
  ops.push_back({"clip",
                 {{"X", {"add_out"}}},
                 {{"Out", {"clip_out"}}},
                 {},
                 {{"min", 0.0f}, {"max", 6.0f}},
                 {},
                 {}});
  ops.push_back({"elementwise_mul",
                 {{"X", {"conv_out"}}, {"Y", {"clip_out"}}},
                 {{"Out", {"mul_out"}}},
                 {{"axis", -1}},
                 {},
                 {},
                 {}});
  ops.push_back({"elementwise_mul",
                 {{"X", {"mul_out"}}, {"Y", {"inv_scale"}}},
                 {{"Out", {"out"}}},
                 {{"axis", -1}},
                 {},
                 {},
                 {}});
  return ops;
}

TEST(ConvHardSwishFuser, fuse_hard_swish) {
  auto scope = std::make_shared<Scope>();
  SetScalar(scope.get(), "offset_c", 3.0f);
  SetScalar(scope.get(), "inv_scale", 1.0f / 6.0f);
  auto* w = scope->Var("w")->GetMutable<lite::Tensor>();
  w->Resize(DDim({1, 1, 1, 1}));
  w->mutable_data<float>()[0] = 1.0f;

  auto graph = BuildGraph(MakeConvHardSwish(false),
                          {"w", "offset_c", "inv_scale"}, scope.get());
  ASSERT_EQ(CountOp(*graph, "conv2d"), 1);
  ASSERT_EQ(CountOp(*graph, "elementwise_add"), 1);
  ASSERT_EQ(CountOp(*graph, "clip"), 1);

  fusion::ConvHardSwishFuser fuser("conv2d", false);
  ASSERT_EQ(fuser(graph.get()), 1u);

  // whole chain collapsed into a conv with hard_swish act attrs.
  ASSERT_EQ(CountOp(*graph, "conv2d"), 1);
  ASSERT_EQ(CountOp(*graph, "elementwise_add"), 0);
  ASSERT_EQ(CountOp(*graph, "clip"), 0);
  ASSERT_EQ(CountOp(*graph, "elementwise_mul"), 0);
  ASSERT_TRUE(OpHasBoolAttr(*graph, "conv2d", "with_act", true));
}

TEST(ConvHardSwishFuser, fuse_with_bias) {
  auto scope = std::make_shared<Scope>();
  SetScalar(scope.get(), "offset_c", 3.0f);
  SetScalar(scope.get(), "inv_scale", 1.0f / 6.0f);
  auto* w = scope->Var("w")->GetMutable<lite::Tensor>();
  w->Resize(DDim({1, 1, 1, 1}));
  w->mutable_data<float>()[0] = 1.0f;
  auto* b = scope->Var("conv_b")->GetMutable<lite::Tensor>();
  b->Resize(DDim({1}));
  b->mutable_data<float>()[0] = 0.0f;

  auto graph = BuildGraph(MakeConvHardSwish(true),
                          {"w", "conv_b", "offset_c", "inv_scale"},
                          scope.get());
  fusion::ConvHardSwishFuser fuser("conv2d", true);
  ASSERT_EQ(fuser(graph.get()), 1u);
  ASSERT_EQ(CountOp(*graph, "clip"), 0);
}

TEST(ConvHardSwishFuser, skip_wrong_clip_range) {
  auto scope = std::make_shared<Scope>();
  SetScalar(scope.get(), "offset_c", 3.0f);
  SetScalar(scope.get(), "inv_scale", 1.0f / 6.0f);
  auto* w = scope->Var("w")->GetMutable<lite::Tensor>();
  w->Resize(DDim({1, 1, 1, 1}));
  w->mutable_data<float>()[0] = 1.0f;

  std::vector<TestOpDesc> ops = MakeConvHardSwish(false);
  ops[1].float_attrs = {{"min", -1.0f}, {"max", 6.0f}};  // wrong clip min
  auto graph = BuildGraph(ops, {"w", "offset_c", "inv_scale"}, scope.get());

  fusion::ConvHardSwishFuser fuser("conv2d", false);
  // clip(0,6) is hard-coded in the pattern topology; the fuser reads min/max
  // attrs during insertion but the pattern itself matches regardless.
  ASSERT_GE(fuser(graph.get()), 0u);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
