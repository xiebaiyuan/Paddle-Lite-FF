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
#include "lite/core/optimizer/mir/elimination/scale_calc_offline_pass.h"
#include "lite/core/optimizer/mir/fusion/fusion_pass_test_util.h"
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

namespace {

// scale(persistable X, scale=2, bias=1) -> Out, consumed by sigmoid.
std::vector<TestOpDesc> MakeScaleChain() {
  std::vector<TestOpDesc> ops;
  TestOpDesc s;
  s.type = "scale";
  s.inputs = {{"X", {"x"}}};
  s.outputs = {{"Out", {"scaled"}}};
  s.float_attrs = {{"scale", 2.0f}, {"bias", 1.0f}};
  s.bool_attrs = {{"bias_after_scale", true}};
  ops.push_back(s);
  ops.push_back({"sigmoid", {{"X", {"scaled"}}}, {{"Out", {"y"}}}});
  return ops;
}

}  // namespace

// A scale whose input is a persistable constant can be computed offline:
// out = scale*x + bias, the output tensor is marked persistable, and the
// scale op is removed.
TEST(ScaleCalcOfflinePass, calc_offline_for_persistable_input) {
  std::set<std::string> persistable{"x"};
  auto scope = std::make_shared<Scope>();
  auto* x = scope->Var("x")->GetMutable<lite::Tensor>();
  x->Resize(DDim({4}));
  auto* xd = x->mutable_data<float>();
  for (int i = 0; i < 4; ++i) xd[i] = static_cast<float>(i);
  x->set_persistable(true);

  auto graph = BuildGraph(MakeScaleChain(), persistable, scope.get());
  ASSERT_EQ(CountOp(*graph, "scale"), 1);

  ScaleCalcOfflinePass pass;
  pass.Apply(graph);

  ASSERT_EQ(CountOp(*graph, "scale"), 0);
  ASSERT_EQ(CountOp(*graph, "sigmoid"), 1);

  // out = 2*x + 1; the output tensor lives in the exec scope.
  Scope* exec_scope = nullptr;
  for (auto& node : graph->nodes()) {
    if (node.IsStmt() && node.stmt()->op_type() == "sigmoid") {
      exec_scope = node.stmt()->op()->scope();
      break;
    }
  }
  ASSERT_NE(exec_scope, nullptr);
  auto* out = exec_scope->FindVar("scaled")->GetMutable<lite::Tensor>();
  ASSERT_EQ(out->dims().size(), 1u);
  ASSERT_EQ(out->dims()[0], 4);
  ASSERT_TRUE(out->persistable());
  const float* od = out->data<float>();
  for (int i = 0; i < 4; ++i) {
    ASSERT_NEAR(od[i], 2.0f * i + 1.0f, 1e-5f);
  }
}

// A scale whose input is NOT persistable must be left untouched.
TEST(ScaleCalcOfflinePass, skip_non_persistable_input) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeScaleChain(), {}, scope.get());

  ASSERT_EQ(CountOp(*graph, "scale"), 1);

  ScaleCalcOfflinePass pass;
  pass.Apply(graph);

  ASSERT_EQ(CountOp(*graph, "scale"), 1);
  ASSERT_EQ(CountOp(*graph, "sigmoid"), 1);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
