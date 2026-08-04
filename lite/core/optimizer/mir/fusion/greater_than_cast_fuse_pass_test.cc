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
#include "lite/core/optimizer/mir/fusion/greater_than_cast_fuser.h"

namespace paddle {
namespace lite {
namespace mir {

// greater_than → cast folds into greater_than with fuse_greater_than=true.
std::vector<TestOpDesc> MakeGreaterThanCast() {
  std::vector<TestOpDesc> ops;
  TestOpDesc gt;
  gt.type = "greater_than";
  gt.inputs = {{"X", {"x"}}, {"Y", {"y"}}};
  gt.outputs = {{"Out", {"gt_out"}}};
  gt.int_attrs = {{"axis", -1}};
  gt.bool_attrs = {{"force_cpu", false}};
  ops.push_back(gt);
  TestOpDesc cast;
  cast.type = "cast";
  cast.inputs = {{"X", {"gt_out"}}};
  cast.outputs = {{"Out", {"out"}}};
  cast.int_attrs = {{"in_dtype", 0}, {"out_dtype", 0}};
  ops.push_back(cast);
  return ops;
}

TEST(GreaterThanCastFuser, fuse_gt_cast) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeGreaterThanCast(), {}, scope.get());
  ASSERT_EQ(CountOp(*graph, "greater_than"), 1);
  ASSERT_EQ(CountOp(*graph, "cast"), 1);

  fusion::GreaterThanCastFuser fuser;
  ASSERT_EQ(fuser(graph.get()), 1u);

  ASSERT_EQ(CountOp(*graph, "greater_than"), 1);
  ASSERT_EQ(CountOp(*graph, "cast"), 0);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
