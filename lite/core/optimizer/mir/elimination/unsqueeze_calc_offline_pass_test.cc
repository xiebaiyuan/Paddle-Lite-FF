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
#include "lite/core/optimizer/mir/elimination/unsqueeze_calc_offline_pass.h"
#include "lite/core/optimizer/mir/fusion/fusion_pass_test_util.h"
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

namespace {

// unsqueeze(persistable X, axes=[1]) -> Out, consumed by sigmoid.
std::vector<TestOpDesc> MakeUnsqueezeChain() {
  std::vector<TestOpDesc> ops;
  TestOpDesc u;
  u.type = "unsqueeze";
  u.inputs = {{"X", {"x"}}};
  u.outputs = {{"Out", {"unsq"}}};
  u.int_vector_attrs = {{"axes", {1}}};
  ops.push_back(u);
  ops.push_back({"sigmoid", {{"X", {"unsq"}}}, {{"Out", {"y"}}}});
  return ops;
}

}  // namespace

// An unsqueeze whose input is persistable can be computed offline: the
// output tensor dims gain the inserted axis, and the op is removed.
TEST(UnsqueezeCalcOfflinePass, calc_offline_for_persistable_input) {
  std::set<std::string> persistable{"x"};
  auto scope = std::make_shared<Scope>();
  auto* x = scope->Var("x")->GetMutable<lite::Tensor>();
  x->Resize(DDim({2, 3}));
  x->mutable_data<float>();
  x->set_persistable(true);

  auto graph = BuildGraph(MakeUnsqueezeChain(), persistable, scope.get());
  ASSERT_EQ(CountOp(*graph, "unsqueeze"), 1);

  UnsqueezeCalcOfflinePass pass;
  pass.Apply(graph);

  ASSERT_EQ(CountOp(*graph, "unsqueeze"), 0);
  ASSERT_EQ(CountOp(*graph, "sigmoid"), 1);

  Scope* exec_scope = nullptr;
  for (auto& node : graph->nodes()) {
    if (node.IsStmt() && node.stmt()->op_type() == "sigmoid") {
      exec_scope = node.stmt()->op()->scope();
      break;
    }
  }
  ASSERT_NE(exec_scope, nullptr);
  auto* out = exec_scope->FindVar("unsq")->GetMutable<lite::Tensor>();
  ASSERT_TRUE(out->persistable());
  // axes=[1] on a 2-D input inserts a dim at position 1: (2,1,3).
  ASSERT_EQ(out->dims().size(), 3u);
  ASSERT_EQ(out->dims()[0], 2);
  ASSERT_EQ(out->dims()[1], 1);
  ASSERT_EQ(out->dims()[2], 3);
}

// A non-persistable input must be left untouched.
TEST(UnsqueezeCalcOfflinePass, skip_non_persistable_input) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeUnsqueezeChain(), {}, scope.get());

  UnsqueezeCalcOfflinePass pass;
  pass.Apply(graph);

  ASSERT_EQ(CountOp(*graph, "unsqueeze"), 1);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
