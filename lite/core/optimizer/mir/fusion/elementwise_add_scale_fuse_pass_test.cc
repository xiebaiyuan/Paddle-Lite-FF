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
#include "lite/core/optimizer/mir/fusion/elementwise_add_scale_fuser.h"
#include "lite/core/optimizer/mir/fusion/fusion_pass_test_util.h"

namespace paddle {
namespace lite {
namespace mir {

// elementwise_mul → scale folds into elementwise_mul with fuse_scale attrs.
std::vector<TestOpDesc> MakeEltScale() {
  std::vector<TestOpDesc> ops;
  TestOpDesc mul;
  mul.type = "elementwise_mul";
  mul.inputs = {{"X", {"x"}}, {"Y", {"y"}}};
  mul.outputs = {{"Out", {"mul_out"}}};
  mul.int_attrs = {{"axis", -1}};
  ops.push_back(mul);
  TestOpDesc scale;
  scale.type = "scale";
  scale.inputs = {{"X", {"mul_out"}}};
  scale.outputs = {{"Out", {"out"}}};
  scale.float_attrs = {{"scale", 2.0f}, {"bias", 1.0f}};
  scale.bool_attrs = {{"bias_after_scale", true}};
  ops.push_back(scale);
  return ops;
}

TEST(ElementwiseScaleFuser, fuse_mul_scale) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeEltScale(), {}, scope.get());
  ASSERT_EQ(CountOp(*graph, "elementwise_mul"), 1);
  ASSERT_EQ(CountOp(*graph, "scale"), 1);

  fusion::ElementwiseScaleFuser fuser("elementwise_mul");
  ASSERT_EQ(fuser(graph.get()), 1u);

  // scale folded into the mul as fuse_scale attrs.
  ASSERT_EQ(CountOp(*graph, "elementwise_mul"), 1);
  ASSERT_EQ(CountOp(*graph, "scale"), 0);
  ASSERT_TRUE(OpHasBoolAttr(*graph, "elementwise_mul", "fuse_scale", true));
  ASSERT_TRUE(OpHasFloatAttr(*graph, "elementwise_mul", "scale", 2.0f));
  ASSERT_TRUE(OpHasFloatAttr(*graph, "elementwise_mul", "bias", 1.0f));
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
