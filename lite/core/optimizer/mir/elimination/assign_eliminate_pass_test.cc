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
#include "lite/core/optimizer/mir/elimination/assign_eliminate_pass.h"
#include "lite/core/optimizer/mir/fusion/fusion_pass_test_util.h"
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

// Helper: assert that a given op type consumes a var produced directly by
// another op type (i.e. no assign in between). We look up the op's input
// var and check its producer's op type.
int CountOpWithDirectInput(const SSAGraph& graph,
                           const std::string& op_type,
                           const std::string& input_var,
                           const std::string& producer_op_type) {
  int count = 0;
  for (auto& node : graph.nodes()) {
    if (!node.IsStmt()) continue;
    auto* stmt = node.stmt();
    if (stmt->op_type() != op_type) continue;
    // check that `input_var` is one of this op's inputs and its producer
    // is `producer_op_type`.
    auto* op_info = stmt->op_info();
    bool has_input = false;
    for (const auto& in_names : op_info->inputs()) {
      for (const auto& n : in_names.second) {
        if (n == input_var) has_input = true;
      }
    }
    if (!has_input) continue;
    // find the arg node and check its producer
    for (auto& n2 : graph.nodes()) {
      if (n2.IsArg() && n2.arg()->name == input_var) {
        for (auto* inlink : n2.inlinks) {
          if (inlink->IsStmt() && inlink->stmt()->op_type() == producer_op_type) {
            ++count;
          }
        }
      }
    }
  }
  return count;
}

// opA(x) -> assign(out=opA_out) -> opB(in=opA_out)
std::vector<TestOpDesc> MakeSimpleAssignChain() {
  std::vector<TestOpDesc> ops;
  ops.push_back({"relu", {{"X", {"x"}}}, {{"Out", {"a_out"}}}});
  ops.push_back({"assign", {{"X", {"a_out"}}}, {{"Out", {"b_in"}}}});
  ops.push_back({"sigmoid", {{"X", {"b_in"}}}, {{"Out", {"y"}}}});
  return ops;
}

TEST(AssignEliminatePass, eliminate_simple_assign) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeSimpleAssignChain(), {}, scope.get());

  // before: assign exists, sigmoid reads b_in (produced by assign)
  ASSERT_EQ(CountOp(*graph, "assign"), 1);
  ASSERT_EQ(CountOpWithDirectInput(*graph, "sigmoid", "b_in", "assign"), 1);

  AssignEliminatePass pass;
  pass.Apply(graph);

  // assign gone; sigmoid now reads a_out produced directly by relu
  ASSERT_EQ(CountOp(*graph, "assign"), 0);
  ASSERT_EQ(CountOpWithDirectInput(*graph, "sigmoid", "a_out", "relu"), 1);
}

// opA(x) -> assign1 -> assign2 -> opB  (2-level chain)
std::vector<TestOpDesc> MakeAssignChain() {
  std::vector<TestOpDesc> ops;
  ops.push_back({"relu", {{"X", {"x"}}}, {{"Out", {"a_out"}}}});
  ops.push_back({"assign", {{"X", {"a_out"}}}, {{"Out", {"b_in"}}}});
  ops.push_back({"assign", {{"X", {"b_in"}}}, {{"Out", {"c_in"}}}});
  ops.push_back({"sigmoid", {{"X", {"c_in"}}}, {{"Out", {"y"}}}});
  return ops;
}

TEST(AssignEliminatePass, eliminate_assign_chain) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeAssignChain(), {}, scope.get());

  ASSERT_EQ(CountOp(*graph, "assign"), 2);

  AssignEliminatePass pass;
  pass.Apply(graph);

  // both assigns gone; sigmoid reads a_out directly from relu
  ASSERT_EQ(CountOp(*graph, "assign"), 0);
  ASSERT_EQ(CountOpWithDirectInput(*graph, "sigmoid", "a_out", "relu"), 1);
}

// assign whose input is a feed var (no producer) must NOT be eliminated:
// rewriting consumers would change what the feed provides.
TEST(AssignEliminatePass, skip_assign_from_feed) {
  std::vector<TestOpDesc> ops;
  ops.push_back({"assign", {{"X", {"feed_x"}}}, {{"Out", {"b_in"}}}});
  ops.push_back({"sigmoid", {{"X", {"b_in"}}}, {{"Out", {"y"}}}});
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(ops, {}, scope.get());

  AssignEliminatePass pass;
  pass.Apply(graph);

  // feed var has no producer -> assign kept
  ASSERT_EQ(CountOp(*graph, "assign"), 1);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
