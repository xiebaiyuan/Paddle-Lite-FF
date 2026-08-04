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
#include "lite/core/optimizer/mir/elimination/remove_tf_redundant_ops_pass.h"
#include "lite/core/optimizer/mir/fusion/fusion_pass_test_util.h"
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

namespace {

// Sets the dims of a tensor in the exec scope's root scope. The ops attached
// during SSAGraph::Build hold a pointer to this scope, so a tensor created
// here is the same tensor the pass observes via op->scope().
void SetTensorDims(Scope* scope,
                   const std::string& name,
                   const std::vector<int64_t>& dims) {
  auto* t = scope->Var(name)->GetMutable<lite::Tensor>();
  t->Resize(dims);
  t->set_precision(PRECISION(kFloat));
}

}  // namespace

// Pattern removed by RemoveSqueeze2Reshape2Pattern:
//   out_arg --squeeze2--> reshape2 --softmax-->
// Requires dims[1] == 1001 and dims[2] == dims[3] == 1 (mobilenet-v2 last
// layer), and softmax consumes the reshape2 output directly.
std::vector<TestOpDesc> MakeSqueeze2Reshape2SoftmaxChain() {
  std::vector<TestOpDesc> ops;
  ops.push_back({"squeeze2",
                 {{"X", {"squeeze_in"}}},
                 {{"Out", {"squeeze_out"}}},
                 {},
                 {},
                 {{"inplace", false}},
                 {},
                 {{"axes", std::vector<int>{2, 3}}}});
  ops.push_back({"reshape2",
                 {{"X", {"squeeze_out"}}},
                 {{"Out", {"reshape_out"}}, {"XShape", {"reshape_xshape"}}},
                 {},
                 {},
                 {{"inplace", false}},
                 {},
                 {{"shape", std::vector<int>{1, 1001, 1, 1}}}});
  ops.push_back({"softmax",
                 {{"X", {"reshape_out"}}},
                 {{"Out", {"softmax_out"}}},
                 {{"axis", -1}},
                 {},
                 {},
                 {},
                 {}});
  return ops;
}

// Positive: the squeeze2->reshape2->softmax pattern is removed and softmax
// reads the original input directly.
TEST(RemoveTFRedundantOpsPass, eliminate_squeeze2_reshape2_softmax) {
  auto scope = std::make_shared<Scope>();
  // out_arg dims must be [1, 1001, 1, 1]: dims[1]==1001, dims[2]==dims[3]==1.
  SetTensorDims(scope.get(), "squeeze_in", {1, 1001, 1, 1});
  SetTensorDims(scope.get(), "squeeze_out", {1, 1001, 1, 1});
  SetTensorDims(scope.get(), "reshape_out", {1, 1001, 1, 1});

  auto graph = BuildGraph(MakeSqueeze2Reshape2SoftmaxChain(), {}, scope.get());
  ASSERT_EQ(CountOp(*graph, "squeeze2"), 1);
  ASSERT_EQ(CountOp(*graph, "reshape2"), 1);
  ASSERT_EQ(CountOp(*graph, "softmax"), 1);

  RemoveTFRedundantOpsPass pass;
  pass.Apply(graph);

  // Both redundant ops gone; softmax kept.
  ASSERT_EQ(CountOp(*graph, "squeeze2"), 0);
  ASSERT_EQ(CountOp(*graph, "reshape2"), 0);
  ASSERT_EQ(CountOp(*graph, "softmax"), 1);
}

// Negative: the last-layer channel count is not 1001, so the squeeze2->
// reshape2 pattern must NOT be removed.
TEST(RemoveTFRedundantOpsPass, keep_pattern_when_channel_not_1001) {
  auto scope = std::make_shared<Scope>();
  SetTensorDims(scope.get(), "squeeze_in", {1, 512, 1, 1});
  SetTensorDims(scope.get(), "squeeze_out", {1, 512, 1, 1});
  SetTensorDims(scope.get(), "reshape_out", {1, 512, 1, 1});

  auto graph = BuildGraph(MakeSqueeze2Reshape2SoftmaxChain(), {}, scope.get());

  RemoveTFRedundantOpsPass pass;
  pass.Apply(graph);

  ASSERT_EQ(CountOp(*graph, "squeeze2"), 1);
  ASSERT_EQ(CountOp(*graph, "reshape2"), 1);
  ASSERT_EQ(CountOp(*graph, "softmax"), 1);
}

// Negative: the reshape2 output feeds a relu (not softmax), so the pattern
// is not matched even with dims[1]==1001.
TEST(RemoveTFRedundantOpsPass, keep_pattern_when_consumer_is_not_softmax) {
  std::vector<TestOpDesc> ops;
  ops.push_back({"squeeze2",
                 {{"X", {"squeeze_in"}}},
                 {{"Out", {"squeeze_out"}}},
                 {},
                 {},
                 {{"inplace", false}},
                 {},
                 {{"axes", std::vector<int>{2, 3}}}});
  ops.push_back({"reshape2",
                 {{"X", {"squeeze_out"}}},
                 {{"Out", {"reshape_out"}}, {"XShape", {"reshape_xshape"}}},
                 {},
                 {},
                 {{"inplace", false}},
                 {},
                 {{"shape", std::vector<int>{1, 1001, 1, 1}}}});
  ops.push_back({"relu", {{"X", {"reshape_out"}}}, {{"Out", {"relu_out"}}}});

  auto scope = std::make_shared<Scope>();
  SetTensorDims(scope.get(), "squeeze_in", {1, 1001, 1, 1});
  SetTensorDims(scope.get(), "squeeze_out", {1, 1001, 1, 1});
  SetTensorDims(scope.get(), "reshape_out", {1, 1001, 1, 1});
  auto graph = BuildGraph(ops, {}, scope.get());

  RemoveTFRedundantOpsPass pass;
  pass.Apply(graph);

  ASSERT_EQ(CountOp(*graph, "squeeze2"), 1);
  ASSERT_EQ(CountOp(*graph, "reshape2"), 1);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
