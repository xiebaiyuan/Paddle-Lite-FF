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

// relu(x) -> dropout(...) -> out. dropout also emits a Mask output, which the
// elimination pattern requires.
std::vector<TestOpDesc> MakeDropoutChain(int is_test, float dropout_prob) {
  std::vector<TestOpDesc> ops;
  ops.push_back({"relu", {{"X", {"x"}}}, {{"Out", {"relu_out"}}}});
  TestOpDesc d;
  d.type = "dropout";
  d.inputs = {{"X", {"relu_out"}}};
  d.outputs = {{"Out", {"out"}}, {"Mask", {"mask"}}};
  d.int_attrs = {{"is_test", is_test}};
  d.float_attrs = {{"dropout_prob", dropout_prob}};
  d.str_attrs = {{"dropout_implementation", "upscale_in_train"}};
  ops.push_back(d);
  return ops;
}

}  // namespace

// In inference (is_test=1), a zero-prob dropout is an identity copy and must
// be eliminated: the pre-op output is rewired directly to `out`.
TEST(IdentityDropoutEliminatePass, eliminate_identity_dropout) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeDropoutChain(1, 0.0f), {}, scope.get());

  ASSERT_EQ(CountOp(*graph, "dropout"), 1);

  auto* pass = PassManager::Global().LookUp("identity_dropout_eliminate_pass");
  ASSERT_NE(pass, nullptr);
  pass->Apply(graph);

  ASSERT_EQ(CountOp(*graph, "dropout"), 0);
  ASSERT_EQ(CountOp(*graph, "relu"), 1);
}

// Training-mode dropout (is_test=0) must be preserved.
TEST(IdentityDropoutEliminatePass, keep_training_dropout) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeDropoutChain(0, 0.0f), {}, scope.get());

  auto* pass = PassManager::Global().LookUp("identity_dropout_eliminate_pass");
  ASSERT_NE(pass, nullptr);
  pass->Apply(graph);

  ASSERT_EQ(CountOp(*graph, "dropout"), 1);
}

// NOTE: the pattern does NOT check dropout_prob — any is_test=1 dropout with
// implementation "upscale_in_train" is eliminated even when dropout_prob>0.
// This documents the pass's actual (conservative) behavior: it treats all
// inference-mode dropouts as identity.
TEST(IdentityDropoutEliminatePass, eliminate_dropout_with_prob_gt_0) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeDropoutChain(1, 0.5f), {}, scope.get());

  auto* pass = PassManager::Global().LookUp("identity_dropout_eliminate_pass");
  ASSERT_NE(pass, nullptr);
  pass->Apply(graph);

  ASSERT_EQ(CountOp(*graph, "dropout"), 0);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
