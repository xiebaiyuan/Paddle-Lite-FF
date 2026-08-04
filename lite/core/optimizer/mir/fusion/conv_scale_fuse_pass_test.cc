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
#include "lite/core/optimizer/mir/fusion/conv_scale_fuser.h"
#include "lite/core/optimizer/mir/fusion/fusion_pass_test_util.h"

namespace paddle {
namespace lite {
namespace mir {

// conv2d(bias) → scale  folds into conv with scaled weight/bias.
std::vector<TestOpDesc> MakeConvScale(bool with_bias) {
  TestOpDesc conv;
  conv.type = "conv2d";
  conv.inputs = {{"Input", {"x"}}, {"Filter", {"w"}}};
  conv.outputs = {{"Output", {"conv_out"}}};
  conv.int_attrs = {{"groups", 1}};
  conv.int_vector_attrs = {{"strides", {1, 1}},
                           {"paddings", {0, 0}},
                           {"dilations", {1, 1}}};
  if (with_bias) conv.inputs["Bias"] = {"b"};
  std::vector<TestOpDesc> ops{conv};
  TestOpDesc scale;
  scale.type = "scale";
  scale.inputs = {{"X", {"conv_out"}}};
  scale.outputs = {{"Out", {"out"}}};
  scale.float_attrs = {{"scale", 2.0f}, {"bias", 1.0f}};
  scale.bool_attrs = {{"bias_after_scale", true}};
  ops.push_back(scale);
  return ops;
}

TEST(ConvScaleFuser, fuse_with_bias) {
  auto scope = std::make_shared<Scope>();
  // weight [1, 1, 1, 1] = {1.0}; bias {1.0}
  auto* w = scope->Var("w")->GetMutable<lite::Tensor>();
  w->Resize(DDim({1, 1, 1, 1}));
  w->mutable_data<float>()[0] = 1.0f;
  auto* b = scope->Var("b")->GetMutable<lite::Tensor>();
  b->Resize(DDim({1}));
  b->mutable_data<float>()[0] = 1.0f;

  auto graph = BuildGraph(MakeConvScale(true), {"w", "b"}, scope.get());
  ASSERT_EQ(CountOp(*graph, "conv2d"), 1);
  ASSERT_EQ(CountOp(*graph, "scale"), 1);

  fusion::ConvScaleFuser fuser("conv2d", true);
  ASSERT_EQ(fuser(graph.get()), 1u);

  // scale folded into conv: weight *= 2, bias = bias*2 + 1
  ASSERT_EQ(CountOp(*graph, "conv2d"), 1);
  ASSERT_EQ(CountOp(*graph, "scale"), 0);
  ASSERT_EQ(w->data<float>()[0], 2.0f);
  ASSERT_EQ(b->data<float>()[0], 3.0f);
}

TEST(ConvScaleFuser, fuse_with_activation) {
  auto scope = std::make_shared<Scope>();
  auto* w = scope->Var("w")->GetMutable<lite::Tensor>();
  w->Resize(DDim({1, 1, 1, 1}));
  w->mutable_data<float>()[0] = 1.0f;
  auto* b = scope->Var("b")->GetMutable<lite::Tensor>();
  b->Resize(DDim({1}));
  b->mutable_data<float>()[0] = 1.0f;

  std::vector<TestOpDesc> ops = MakeConvScale(true);
  ops[1].str_attrs = {{"activation_type", "relu"}};
  auto graph = BuildGraph(ops, {"w", "b"}, scope.get());

  fusion::ConvScaleFuser fuser("conv2d", true);
  ASSERT_EQ(fuser(graph.get()), 1u);
  ASSERT_EQ(CountOp(*graph, "scale"), 0);
}

TEST(ConvScaleFuser, skip_non_persistable_weight) {
  // conv without bias is unsupported (LOG(FATAL) in pattern) — the pattern
  // builder requires bias, so a no-bias conv must not fuse.
  auto scope = std::make_shared<Scope>();
  auto* w = scope->Var("w")->GetMutable<lite::Tensor>();
  w->Resize(DDim({1, 1, 1, 1}));
  w->mutable_data<float>()[0] = 1.0f;

  auto graph = BuildGraph(MakeConvScale(false), {}, scope.get());
  fusion::ConvScaleFuser fuser("conv2d", true);
  ASSERT_EQ(fuser(graph.get()), 0u);
  ASSERT_EQ(CountOp(*graph, "scale"), 1);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
