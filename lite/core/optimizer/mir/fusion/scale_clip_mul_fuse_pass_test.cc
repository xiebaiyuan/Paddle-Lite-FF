// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include "lite/core/optimizer/mir/fusion/fusion_pass_test_util.h"
#include "lite/core/optimizer/mir/fusion/scale_clip_mul_fuse_pass.h"
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

// Pattern: gate → scale(slope, 0.5) → clip(0,1) → mul(x, gate') → out
std::vector<TestOpDesc> MakeScaleClipMulChain(float slope) {
  std::vector<TestOpDesc> ops;
  TestOpDesc scale;
  scale.type = "scale";
  scale.inputs = {{"X", {"gate"}}};
  scale.outputs = {{"Out", {"scale_out"}}};
  scale.float_attrs = {{"scale", slope}, {"bias", 0.5f}};
  scale.bool_attrs = {{"bias_after_scale", true}};
  ops.push_back(scale);

  TestOpDesc clip;
  clip.type = "clip";
  clip.inputs = {{"X", {"scale_out"}}};
  clip.outputs = {{"Out", {"clip_out"}}};
  clip.float_attrs = {{"min", 0.0f}, {"max", 1.0f}};
  ops.push_back(clip);

  ops.push_back({"elementwise_mul",
                 {{"X", {"x"}}, {"Y", {"clip_out"}}},
                 {{"Out", {"out"}}},
                 {{"axis", -1}},
                 {},
                 {},
                 {}});
  return ops;
}

TEST(ScaleClipMulFusePass, fold_scale_clip_to_hard_sigmoid) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeScaleClipMulChain(0.1667f), {}, scope.get());
  ASSERT_EQ(CountOp(*graph, "scale"), 1);
  ASSERT_EQ(CountOp(*graph, "clip"), 1);
  ASSERT_EQ(CountOp(*graph, "hard_sigmoid"), 0);

  ScaleClipMulFusePass pass;
  pass.Apply(graph);

  // scale+clip folded into hard_sigmoid; mul preserved as the gate.
  ASSERT_EQ(CountOp(*graph, "scale"), 0);
  ASSERT_EQ(CountOp(*graph, "clip"), 0);
  ASSERT_EQ(CountOp(*graph, "hard_sigmoid"), 1);
  ASSERT_EQ(CountOp(*graph, "elementwise_mul"), 1);
}

TEST(ScaleClipMulFusePass, skip_wrong_bias) {
  // bias != 0.5 is not hard_sigmoid; must not fuse.
  std::vector<TestOpDesc> ops = MakeScaleClipMulChain(0.1667f);
  ops[0].float_attrs = {{"scale", 0.1667f}, {"bias", 0.3f}};  // wrong bias
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(ops, {}, scope.get());

  ScaleClipMulFusePass pass;
  pass.Apply(graph);

  ASSERT_EQ(CountOp(*graph, "scale"), 1);
  ASSERT_EQ(CountOp(*graph, "hard_sigmoid"), 0);
}

TEST(ScaleClipMulFusePass, skip_non_unit_clip_range) {
  // clip range != (0,1) is not hard_sigmoid; must not fuse.
  std::vector<TestOpDesc> ops = MakeScaleClipMulChain(0.1667f);
  ops[1].float_attrs = {{"min", -1.0f}, {"max", 1.0f}};  // wrong range
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(ops, {}, scope.get());

  ScaleClipMulFusePass pass;
  pass.Apply(graph);

  ASSERT_EQ(CountOp(*graph, "scale"), 1);
  ASSERT_EQ(CountOp(*graph, "hard_sigmoid"), 0);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
