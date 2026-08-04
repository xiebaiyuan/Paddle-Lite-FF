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
#include "lite/core/optimizer/mir/elimination/reshape_calc_offline_pass.h"
#include "lite/core/optimizer/mir/fusion/fusion_pass_test_util.h"
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

namespace {

// reshape(persistable X, shape=[2,4]) -> Out, consumed by sigmoid.
std::vector<TestOpDesc> MakeReshapeChain() {
  std::vector<TestOpDesc> ops;
  TestOpDesc r;
  r.type = "reshape";
  r.inputs = {{"X", {"x"}}};
  r.outputs = {{"Out", {"reshaped"}}};
  r.int_vector_attrs = {{"shape", {2, 4}}};
  ops.push_back(r);
  ops.push_back({"sigmoid", {{"X", {"reshaped"}}}, {{"Out", {"y"}}}});
  return ops;
}

}  // namespace

// A reshape whose input is a persistable constant can be computed offline:
// the output tensor is reshaped to (2,4), marked persistable, and the
// reshape op is removed from the graph.
TEST(ReshapeCalcOfflinePass, calc_offline_for_persistable_input) {
  std::set<std::string> persistable{"x"};
  auto scope = std::make_shared<Scope>();
  // Input must be a sized, allocated persistable tensor: CopyDataFrom copies
  // memory_size_ bytes, and ValidateShape needs a concrete input_dims.
  auto* x = scope->Var("x")->GetMutable<lite::Tensor>();
  x->Resize(DDim({8}));
  x->mutable_data<float>();
  x->set_persistable(true);

  auto graph = BuildGraph(MakeReshapeChain(), persistable, scope.get());
  ASSERT_EQ(CountOp(*graph, "reshape"), 1);

  ReshapeCalcOfflinePass pass;
  pass.Apply(graph);

  ASSERT_EQ(CountOp(*graph, "reshape"), 0);
  ASSERT_EQ(CountOp(*graph, "sigmoid"), 1);

  // The reshape output tensor was computed offline: resized to (2,4) and
  // marked persistable. `reshaped` lives in the exec scope (a child of the
  // test's root scope), reachable through a surviving stmt node's op scope.
  Scope* exec_scope = nullptr;
  for (auto& node : graph->nodes()) {
    if (node.IsStmt() && node.stmt()->op_type() == "sigmoid") {
      exec_scope = node.stmt()->op()->scope();
      break;
    }
  }
  ASSERT_NE(exec_scope, nullptr);
  auto* out = exec_scope->FindVar("reshaped")->GetMutable<lite::Tensor>();
  ASSERT_EQ(out->dims().size(), 2u);
  ASSERT_EQ(out->dims()[0], 2);
  ASSERT_EQ(out->dims()[1], 4);
  ASSERT_TRUE(out->persistable());
}

// A reshape whose input is NOT persistable must be left untouched: runtime
// reshape is required because the shape of a non-persistable tensor is only
// known at runtime.
TEST(ReshapeCalcOfflinePass, skip_non_persistable_input) {
  auto scope = std::make_shared<Scope>();
  // Non-persistable input: BuildGraph creates it in the exec scope.
  auto graph = BuildGraph(MakeReshapeChain(), {}, scope.get());

  ASSERT_EQ(CountOp(*graph, "reshape"), 1);

  ReshapeCalcOfflinePass pass;
  pass.Apply(graph);

  ASSERT_EQ(CountOp(*graph, "reshape"), 1);
  ASSERT_EQ(CountOp(*graph, "sigmoid"), 1);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
