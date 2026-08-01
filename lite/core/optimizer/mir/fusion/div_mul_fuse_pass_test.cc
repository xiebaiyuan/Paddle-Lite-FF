// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include "lite/core/optimizer/mir/fusion/div_mul_fuse_pass.h"
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/optimizer/mir/fusion/fusion_pass_test_util.h"
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

// Pattern: x / c_div * c_mul → x * (c_mul / c_div)
std::vector<TestOpDesc> MakeDivMulChain() {
  std::vector<TestOpDesc> ops;
  // axis=-1 broadcasts per-element; required by elementwise AttachImpl.
  const std::map<std::string, int> kAxisM1{{"axis", -1}};
  ops.push_back({"elementwise_div",
                 {{"X", {"x"}}, {"Y", {"div_y"}}},
                 {{"Out", {"div_out"}}},
                 kAxisM1,
                 {},
                 {},
                 {}});
  ops.push_back({"elementwise_mul",
                 {{"X", {"div_out"}}, {"Y", {"mul_y"}}},
                 {{"Out", {"out"}}},
                 kAxisM1,
                 {},
                 {},
                 {}});
  return ops;
}

TEST(DivMulFusePass, fold_constants) {
  std::set<std::string> persistable{"div_y", "mul_y"};
  auto scope = std::make_shared<Scope>();
  SetScalar(scope.get(), "div_y", 2.0f);
  SetScalar(scope.get(), "mul_y", 3.0f);

  auto graph = BuildGraph(MakeDivMulChain(), persistable, scope.get());
  ASSERT_EQ(CountOp(*graph, "elementwise_div"), 1);
  ASSERT_EQ(CountOp(*graph, "elementwise_mul"), 1);

  DivMulFusePass pass;
  pass.Apply(graph);

  // div folded into the mul's Y constant (3/2 = 1.5), mul preserved.
  ASSERT_EQ(CountOp(*graph, "elementwise_div"), 0);
  ASSERT_EQ(CountOp(*graph, "elementwise_mul"), 1);
  // The mul's Y constant now holds c_mul/c_div = 1.5.
  auto* scope_mul_y = scope->FindMutableTensor("mul_y");
  ASSERT_TRUE(scope_mul_y != nullptr);
  ASSERT_EQ(scope_mul_y->data<float>()[0], 1.5f);
}

TEST(DivMulFusePass, skip_shared_constant) {
  // If the mul constant is shared by another consumer, folding would corrupt it.
  std::set<std::string> persistable{"div_y", "mul_y"};
  auto scope = std::make_shared<Scope>();
  SetScalar(scope.get(), "div_y", 2.0f);
  SetScalar(scope.get(), "mul_y", 3.0f);

  std::vector<TestOpDesc> ops = MakeDivMulChain();
  // another mul consuming mul_y
  const std::map<std::string, int> kAxisM1{{"axis", -1}};
  ops.push_back({"elementwise_mul",
                 {{"X", {"x"}}, {"Y", {"mul_y"}}},
                 {{"Out", {"other_out"}}},
                 kAxisM1,
                 {},
                 {},
                 {}});
  auto graph = BuildGraph(ops, persistable, scope.get());

  DivMulFusePass pass;
  pass.Apply(graph);

  // Shared constant → skip folding, div preserved.
  ASSERT_EQ(CountOp(*graph, "elementwise_div"), 1);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
