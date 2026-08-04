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
#include "lite/core/optimizer/mir/elimination/range_calc_offline_pass.h"
#include "lite/core/optimizer/mir/fusion/fusion_pass_test_util.h"
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

namespace {

// range(Start=start, End=end, Step=step) -> Out, consumed by sigmoid.
std::vector<TestOpDesc> MakeRangeChain() {
  std::vector<TestOpDesc> ops;
  TestOpDesc r;
  r.type = "range";
  r.inputs = {{"Start", {"start"}}, {"End", {"end"}}, {"Step", {"step"}}};
  r.outputs = {{"Out", {"range_out"}}};
  ops.push_back(r);
  ops.push_back({"sigmoid", {{"X", {"range_out"}}}, {{"Out", {"y"}}}});
  return ops;
}

}  // namespace

// range with persistable Start/End/Step is computed offline: output holds
// [0, 1, 2, 3, 4] for start=0, end=5, step=1, marked persistable, op removed.
TEST(RangeCalcOfflinePass, calc_offline_for_persistable_inputs) {
  std::set<std::string> persistable{"start", "end", "step"};
  auto scope = std::make_shared<Scope>();
  auto* start = scope->Var("start")->GetMutable<lite::Tensor>();
  start->Resize(DDim({1}));
  start->mutable_data<float>()[0] = 0.0f;
  start->set_persistable(true);
  auto* end = scope->Var("end")->GetMutable<lite::Tensor>();
  end->Resize(DDim({1}));
  end->mutable_data<float>()[0] = 5.0f;
  end->set_persistable(true);
  auto* step = scope->Var("step")->GetMutable<lite::Tensor>();
  step->Resize(DDim({1}));
  step->mutable_data<float>()[0] = 1.0f;
  step->set_persistable(true);

  auto graph = BuildGraph(MakeRangeChain(), persistable, scope.get());
  ASSERT_EQ(CountOp(*graph, "range"), 1);

  RangeCalcOfflinePass pass;
  pass.Apply(graph);

  ASSERT_EQ(CountOp(*graph, "range"), 0);
  ASSERT_EQ(CountOp(*graph, "sigmoid"), 1);

  Scope* exec_scope = nullptr;
  for (auto& node : graph->nodes()) {
    if (node.IsStmt() && node.stmt()->op_type() == "sigmoid") {
      exec_scope = node.stmt()->op()->scope();
      break;
    }
  }
  ASSERT_NE(exec_scope, nullptr);
  auto* out = exec_scope->FindVar("range_out")->GetMutable<lite::Tensor>();
  ASSERT_TRUE(out->persistable());
  ASSERT_EQ(out->data_size(), 5);
  const float* od = out->data<float>();
  for (int i = 0; i < 5; ++i) {
    ASSERT_NEAR(od[i], static_cast<float>(i), 1e-5f);
  }
}

// range with a non-persistable input must be left untouched.
TEST(RangeCalcOfflinePass, skip_non_persistable_input) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeRangeChain(), {}, scope.get());

  RangeCalcOfflinePass pass;
  pass.Apply(graph);

  ASSERT_EQ(CountOp(*graph, "range"), 1);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
