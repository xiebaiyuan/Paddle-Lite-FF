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
#include "lite/core/optimizer/mir/fusion/fusion_pass_test_util.h"
#include "lite/core/optimizer/mir/fusion/scales_fuser.h"

namespace paddle {
namespace lite {
namespace mir {

// scale → scale (both plain) folds into scale with combined scale/bias:
//   out = (x*s1+b1)*s2+b2 = x*(s1*s2) + (b1*s2+b2)
std::vector<TestOpDesc> MakeScales() {
  std::vector<TestOpDesc> ops;
  TestOpDesc s1;
  s1.type = "scale";
  s1.inputs = {{"X", {"x"}}};
  s1.outputs = {{"Out", {"s1_out"}}};
  s1.float_attrs = {{"scale", 2.0f}, {"bias", 1.0f}};
  s1.bool_attrs = {{"bias_after_scale", true}};
  ops.push_back(s1);

  TestOpDesc s2;
  s2.type = "scale";
  s2.inputs = {{"X", {"s1_out"}}};
  s2.outputs = {{"Out", {"out"}}};
  s2.float_attrs = {{"scale", 3.0f}, {"bias", 0.5f}};
  s2.bool_attrs = {{"bias_after_scale", true}};
  ops.push_back(s2);
  return ops;
}

TEST(ScalesFuser, fuse_two_scales) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeScales(), {}, scope.get());
  ASSERT_EQ(CountOp(*graph, "scale"), 2);

  fusion::ScalesFuser fuser;
  ASSERT_EQ(fuser(graph.get()), 1u);

  // combined: scale=2*3=6, bias=1*3+0.5=3.5
  ASSERT_EQ(CountOp(*graph, "scale"), 1);
  ASSERT_TRUE(OpHasFloatAttr(*graph, "scale", "scale", 6.0f));
  ASSERT_TRUE(OpHasFloatAttr(*graph, "scale", "bias", 3.5f));
}

TEST(ScalesFuser, skip_scale_with_act) {
  auto scope = std::make_shared<Scope>();
  std::vector<TestOpDesc> ops = MakeScales();
  ops[0].str_attrs = {{"activation_type", "relu"}};
  auto graph = BuildGraph(ops, {}, scope.get());
  fusion::ScalesFuser fuser;
  ASSERT_EQ(fuser(graph.get()), 0u);
  ASSERT_EQ(CountOp(*graph, "scale"), 2);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
