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
#include "lite/core/optimizer/mir/elimination/assign_value_calc_offline_pass.h"
#include "lite/core/optimizer/mir/fusion/fusion_pass_test_util.h"
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

namespace {

// assign_value(dtype=FP32, shape=[2], fp32_values=[1.5, 2.5]) -> Out,
// consumed by sigmoid.
std::vector<TestOpDesc> MakeAssignValueChain() {
  std::vector<TestOpDesc> ops;
  TestOpDesc a;
  a.type = "assign_value";
  a.outputs = {{"Out", {"assigned"}}};
  a.int_attrs = {{"dtype", 5}};  // FP32
  a.int_vector_attrs = {{"shape", {2}}};
  a.float_vector_attrs = {{"fp32_values", {1.5f, 2.5f}}};
  ops.push_back(a);
  ops.push_back({"sigmoid", {{"X", {"assigned"}}}, {{"Out", {"y"}}}});
  return ops;
}

}  // namespace

// assign_value whose output var is also produced by a control-flow/increment
// op (the "extra producer" candidates checked by HasExtraProducers) cannot be
// computed offline: the pass skips it, leaving the op untouched so the runtime
// producer keeps writing the shared variable.
TEST(AssignValueCalcOfflinePass, skip_when_multiple_producers) {
  std::vector<TestOpDesc> ops;
  TestOpDesc a;
  a.type = "assign_value";
  a.outputs = {{"Out", {"assigned"}}};
  a.int_attrs = {{"dtype", 5}};  // FP32
  a.int_vector_attrs = {{"shape", {2}}};
  a.float_vector_attrs = {{"fp32_values", {1.5f, 2.5f}}};
  ops.push_back(a);
  // increment also writes `assigned` -> the output var has a second producer
  // (increment is one of the candidate ops checked by HasExtraProducers).
  TestOpDesc inc;
  inc.type = "increment";
  inc.inputs = {{"X", {"inc_in"}}};
  inc.outputs = {{"Out", {"assigned"}}};
  inc.float_attrs = {{"step", 1.0f}};
  ops.push_back(inc);
  ops.push_back({"sigmoid", {{"X", {"assigned"}}}, {{"Out", {"y"}}}});
  auto scope = std::make_shared<Scope>();
  auto* inc_in = scope->Var("inc_in")->GetMutable<lite::Tensor>();
  inc_in->Resize(DDim({2}));
  inc_in->mutable_data<float>();
  auto graph = BuildGraph(ops, {}, scope.get());

  ASSERT_EQ(CountOp(*graph, "assign_value"), 1);

  AssignValueCalcOfflinePass pass;
  pass.Apply(graph);

  ASSERT_EQ(CountOp(*graph, "assign_value"), 1);
  ASSERT_EQ(CountOp(*graph, "increment"), 1);
  ASSERT_EQ(CountOp(*graph, "sigmoid"), 1);
}

// assign_value with compile-time values is computed offline: the output
// tensor is filled and marked persistable, and the op removed.
TEST(AssignValueCalcOfflinePass, calc_offline_for_assign_value) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeAssignValueChain(), {}, scope.get());
  ASSERT_EQ(CountOp(*graph, "assign_value"), 1);

  AssignValueCalcOfflinePass pass;
  pass.Apply(graph);

  ASSERT_EQ(CountOp(*graph, "assign_value"), 0);
  ASSERT_EQ(CountOp(*graph, "sigmoid"), 1);

  Scope* exec_scope = nullptr;
  for (auto& node : graph->nodes()) {
    if (node.IsStmt() && node.stmt()->op_type() == "sigmoid") {
      exec_scope = node.stmt()->op()->scope();
      break;
    }
  }
  ASSERT_NE(exec_scope, nullptr);
  auto* out = exec_scope->FindVar("assigned")->GetMutable<lite::Tensor>();
  ASSERT_TRUE(out->persistable());
  ASSERT_EQ(out->dims().size(), 1u);
  ASSERT_EQ(out->dims()[0], 2);
  const float* od = out->data<float>();
  ASSERT_NEAR(od[0], 1.5f, 1e-5f);
  ASSERT_NEAR(od[1], 2.5f, 1e-5f);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
