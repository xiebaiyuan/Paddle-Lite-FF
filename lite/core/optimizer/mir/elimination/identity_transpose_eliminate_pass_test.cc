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
#include <set>
#include <string>
#include <vector>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/core/optimizer/mir/fusion/fusion_pass_test_util.h"
#include "lite/core/optimizer/mir/pass_manager.h"
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

namespace {

// pre_op(x) -> transpose2(axis0) -> transpose2(axis1) -> out
std::vector<TestOpDesc> MakeTransposeChain(const std::vector<int>& axis0,
                                           const std::vector<int>& axis1) {
  std::vector<TestOpDesc> ops;
  ops.push_back({"relu", {{"X", {"x"}}}, {{"Out", {"relu_out"}}}});
  TestOpDesc t0;
  t0.type = "transpose2";
  t0.inputs = {{"X", {"relu_out"}}};
  t0.outputs = {{"Out", {"mid"}}, {"XShape", {"mid_shape"}}};
  t0.int_vector_attrs = {{"axis", axis0}};
  ops.push_back(t0);
  TestOpDesc t1;
  t1.type = "transpose2";
  t1.inputs = {{"X", {"mid"}}};
  t1.outputs = {{"Out", {"out"}}, {"XShape", {"out_shape"}}};
  t1.int_vector_attrs = {{"axis", axis1}};
  ops.push_back(t1);
  return ops;
}

}  // namespace

// Two transpose2 ops whose permutations compose to the identity must be
// collapsed: the first transpose's input is rewired to the final output and
// the second transpose disappears. transpose2([0,2,1]) twice is the identity.
TEST(IdentityTransposeEliminatePass, eliminate_identity_transpose_pair) {
  auto scope = std::make_shared<Scope>();
  auto graph =
      BuildGraph(MakeTransposeChain({0, 2, 1}, {0, 2, 1}), {}, scope.get());

  ASSERT_EQ(CountOp(*graph, "transpose2"), 2);

  auto* pass =
      PassManager::Global().LookUp("identity_transpose_eliminate_pass");
  ASSERT_NE(pass, nullptr);
  pass->Apply(graph);

  ASSERT_EQ(CountOp(*graph, "transpose2"), 1);
  ASSERT_EQ(CountOp(*graph, "relu"), 1);
}

// Two transpose2 ops whose permutations do NOT compose to the identity must
// be preserved: transpose2([0,2,1]) then transpose2([0,1,2]) reorders once.
TEST(IdentityTransposeEliminatePass, keep_non_identity_transpose_pair) {
  auto scope = std::make_shared<Scope>();
  auto graph =
      BuildGraph(MakeTransposeChain({0, 2, 1}, {0, 1, 2}), {}, scope.get());

  auto* pass =
      PassManager::Global().LookUp("identity_transpose_eliminate_pass");
  ASSERT_NE(pass, nullptr);
  pass->Apply(graph);

  ASSERT_EQ(CountOp(*graph, "transpose2"), 2);
}

// A single transpose2 (no pair) must be preserved.
TEST(IdentityTransposeEliminatePass, keep_single_transpose) {
  auto scope = std::make_shared<Scope>();
  std::vector<TestOpDesc> ops = MakeTransposeChain({0, 2, 1}, {0, 2, 1});
  ops.pop_back();  // drop the second transpose

  auto graph = BuildGraph(ops, {}, scope.get());
  auto* pass =
      PassManager::Global().LookUp("identity_transpose_eliminate_pass");
  ASSERT_NE(pass, nullptr);
  pass->Apply(graph);

  ASSERT_EQ(CountOp(*graph, "transpose2"), 1);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
