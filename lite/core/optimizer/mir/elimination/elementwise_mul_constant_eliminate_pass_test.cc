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

// relu(x) -> elementwise_mul(relu_out, fill_constant(value)) -> sigmoid -> y
// The elementwise_mul's Y is a fill_constant persistable tensor of `value`.
std::vector<TestOpDesc> MakeMulChain(float value) {
  std::vector<TestOpDesc> ops;
  ops.push_back({"relu", {{"X", {"x"}}}, {{"Out", {"relu_out"}}}});
  TestOpDesc fc;
  fc.type = "fill_constant";
  fc.inputs = {};
  fc.outputs = {{"Out", {"mul_y"}}};
  fc.int_attrs = {{"dtype", 5}};  // Paddle dtype 5 == FP32
  fc.float_attrs = {{"value", value}};
  fc.bool_attrs = {{"force_cpu", false}};
  fc.int_vector_attrs = {{"shape", {1}}};
  ops.push_back(fc);
  TestOpDesc mul;
  mul.type = "elementwise_mul";
  mul.inputs = {{"X", {"relu_out"}}, {"Y", {"mul_y"}}};
  mul.outputs = {{"Out", {"mul_out"}}};
  mul.int_attrs = {{"axis", -1}};
  ops.push_back(mul);
  ops.push_back({"sigmoid", {{"X", {"mul_out"}}}, {{"Out", {"y"}}}});
  return ops;
}

}  // namespace

// elementwise_mul by a constant 1.0 is an identity and must be eliminated:
// the mul and its fill_constant Y are removed, and the post-op's input is
// rewired from mul_out to relu_out (the pre-op output).
TEST(ElementwiseMulConstantEliminatePass, eliminate_mul_by_one) {
  std::set<std::string> persistable{"mul_y"};
  auto scope = std::make_shared<Scope>();
  SetScalar(scope.get(), "mul_y", 1.0f);

  auto graph = BuildGraph(MakeMulChain(1.0f), persistable, scope.get());
  ASSERT_EQ(CountOp(*graph, "elementwise_mul"), 1);
  ASSERT_EQ(CountOp(*graph, "fill_constant"), 1);

  auto* pass =
      PassManager::Global().LookUp("elementwise_mul_constant_eliminate_pass");
  ASSERT_NE(pass, nullptr);
  pass->Apply(graph);

  ASSERT_EQ(CountOp(*graph, "elementwise_mul"), 0);
  ASSERT_EQ(CountOp(*graph, "fill_constant"), 0);
  ASSERT_EQ(CountOp(*graph, "relu"), 1);
  ASSERT_EQ(CountOp(*graph, "sigmoid"), 1);
}

// elementwise_mul by a constant != 1.0 is NOT an identity and must be
// preserved.
TEST(ElementwiseMulConstantEliminatePass, keep_mul_by_non_one) {
  std::set<std::string> persistable{"mul_y"};
  auto scope = std::make_shared<Scope>();
  SetScalar(scope.get(), "mul_y", 2.0f);

  auto graph = BuildGraph(MakeMulChain(2.0f), persistable, scope.get());

  auto* pass =
      PassManager::Global().LookUp("elementwise_mul_constant_eliminate_pass");
  ASSERT_NE(pass, nullptr);
  pass->Apply(graph);

  ASSERT_EQ(CountOp(*graph, "elementwise_mul"), 1);
  ASSERT_EQ(CountOp(*graph, "fill_constant"), 1);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
