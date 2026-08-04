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
#include "lite/core/optimizer/mir/elimination/fill_constant_calc_offline_pass.h"
#include "lite/core/optimizer/mir/fusion/fusion_pass_test_util.h"
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

namespace {

// fill_constant(value=3.0, shape=[2,3]) -> Out, consumed by sigmoid.
std::vector<TestOpDesc> MakeFillConstantChain() {
  std::vector<TestOpDesc> ops;
  TestOpDesc f;
  f.type = "fill_constant";
  f.outputs = {{"Out", {"filled"}}};
  f.int_attrs = {{"dtype", 5}};  // FP32
  f.float_attrs = {{"value", 3.0f}};
  f.bool_attrs = {{"force_cpu", false}};
  f.int_vector_attrs = {{"shape", {2, 3}}};
  ops.push_back(f);
  ops.push_back({"sigmoid", {{"X", {"filled"}}}, {{"Out", {"y"}}}});
  return ops;
}

}  // namespace

// A fill_constant with a compile-time value and shape is computed offline:
// the output tensor is filled and marked persistable, and the op removed.
TEST(FillConstantCalcOfflinePass, calc_offline_for_fill_constant) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeFillConstantChain(), {}, scope.get());
  ASSERT_EQ(CountOp(*graph, "fill_constant"), 1);

  FillConstantCalcOfflinePass pass;
  pass.Apply(graph);

  ASSERT_EQ(CountOp(*graph, "fill_constant"), 0);
  ASSERT_EQ(CountOp(*graph, "sigmoid"), 1);

  Scope* exec_scope = nullptr;
  for (auto& node : graph->nodes()) {
    if (node.IsStmt() && node.stmt()->op_type() == "sigmoid") {
      exec_scope = node.stmt()->op()->scope();
      break;
    }
  }
  ASSERT_NE(exec_scope, nullptr);
  auto* out = exec_scope->FindVar("filled")->GetMutable<lite::Tensor>();
  ASSERT_EQ(out->dims().size(), 2u);
  ASSERT_EQ(out->dims()[0], 2);
  ASSERT_EQ(out->dims()[1], 3);
  ASSERT_TRUE(out->persistable());
  const float* od = out->data<float>();
  for (int i = 0; i < 6; ++i) {
    ASSERT_NEAR(od[i], 3.0f, 1e-5f);
  }
}

// A fill_constant with a ValueTensor input (runtime value) must be left
// untouched: the value is not a compile-time constant.
TEST(FillConstantCalcOfflinePass, skip_with_value_tensor_input) {
  std::vector<TestOpDesc> ops;
  TestOpDesc f;
  f.type = "fill_constant";
  f.inputs = {{"ValueTensor", {"vt"}}};
  f.outputs = {{"Out", {"filled"}}};
  f.int_attrs = {{"dtype", 5}};
  f.float_attrs = {{"value", 3.0f}};
  f.bool_attrs = {{"force_cpu", false}};
  f.int_vector_attrs = {{"shape", {2, 3}}};
  ops.push_back(f);
  ops.push_back({"sigmoid", {{"X", {"filled"}}}, {{"Out", {"y"}}}});
  auto scope = std::make_shared<Scope>();
  auto* vt = scope->Var("vt")->GetMutable<lite::Tensor>();
  vt->Resize(DDim({1}));
  vt->mutable_data<float>()[0] = 3.0f;
  auto graph = BuildGraph(ops, {}, scope.get());

  FillConstantCalcOfflinePass pass;
  pass.Apply(graph);

  ASSERT_EQ(CountOp(*graph, "fill_constant"), 1);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
