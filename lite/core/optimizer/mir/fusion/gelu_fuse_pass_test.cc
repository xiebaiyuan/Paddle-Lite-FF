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
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/optimizer/mir/fusion/fusion_pass_test_util.h"
#include "lite/core/optimizer/mir/fusion/gelu_fuse_pass.h"
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

// GELU(x) = 0.5 * x * (1 + erf(x / √2))
//  x → div(x, √2) → erf → add(1.0) → mul(x) → mul(0.5) → out
std::vector<TestOpDesc> MakeGeluChain(const std::string& suffix = "") {
  std::vector<TestOpDesc> ops;
  // axis=-1 broadcasts per-element; required by elementwise AttachImpl.
  const std::map<std::string, int> kAxisM1{{"axis", -1}};
  ops.push_back({"elementwise_div",
                 {{"X", {"x"}}, {"Y", {"div_y" + suffix}}},
                 {{"Out", {"div_out" + suffix}}},
                 kAxisM1,
                 {},
                 {},
                 {}});
  ops.push_back({"erf",
                 {{"X", {"div_out" + suffix}}},
                 {{"Out", {"erf_out" + suffix}}}});
  ops.push_back({"elementwise_add",
                 {{"X", {"erf_out" + suffix}}, {"Y", {"add_y" + suffix}}},
                 {{"Out", {"add_out" + suffix}}},
                 kAxisM1,
                 {},
                 {},
                 {}});
  ops.push_back({"elementwise_mul",
                 {{"X", {"x"}}, {"Y", {"add_out" + suffix}}},
                 {{"Out", {"mul_out" + suffix}}},
                 kAxisM1,
                 {},
                 {},
                 {}});
  ops.push_back({"elementwise_mul",
                 {{"X", {"mul_out" + suffix}}, {"Y", {"scale_y" + suffix}}},
                 {{"Out", {"out" + suffix}}},
                 kAxisM1,
                 {},
                 {},
                 {}});
  return ops;
}

TEST(GeluFusePass, fuse_erf_chain) {
  std::set<std::string> persistable{"div_y", "add_y", "scale_y"};
  auto scope = std::make_shared<Scope>();
  SetScalar(scope.get(), "div_y", 1.41421356237f);  // √2
  SetScalar(scope.get(), "add_y", 1.0f);
  SetScalar(scope.get(), "scale_y", 0.5f);

  auto graph = BuildGraph(MakeGeluChain(), persistable, scope.get());
  ASSERT_EQ(CountOp(*graph, "gelu"), 0);
  ASSERT_EQ(CountOp(*graph, "elementwise_div"), 1);

  GeluFusePass pass;
  pass.Apply(graph);

  // Chain folded into a single gelu op.
  ASSERT_EQ(CountOp(*graph, "gelu"), 1);
  ASSERT_EQ(CountOp(*graph, "elementwise_div"), 0);
  ASSERT_EQ(CountOp(*graph, "erf"), 0);
  ASSERT_EQ(CountOp(*graph, "elementwise_add"), 0);
  ASSERT_EQ(CountOp(*graph, "elementwise_mul"), 0);
}

TEST(GeluFusePass, skip_wrong_constants) {
  // Wrong constants (not √2/1.0/0.5) must not fuse — numerics would be wrong.
  std::set<std::string> persistable{"div_y", "add_y", "scale_y"};
  auto scope = std::make_shared<Scope>();
  SetScalar(scope.get(), "div_y", 2.0f);  // not √2
  SetScalar(scope.get(), "add_y", 1.0f);
  SetScalar(scope.get(), "scale_y", 0.5f);

  auto graph = BuildGraph(MakeGeluChain(), persistable, scope.get());
  GeluFusePass pass;
  pass.Apply(graph);

  ASSERT_EQ(CountOp(*graph, "gelu"), 0);
  ASSERT_EQ(CountOp(*graph, "elementwise_div"), 1);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
