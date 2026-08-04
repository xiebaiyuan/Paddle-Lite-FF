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

// relu(x) -> scale(scale, bias) -> out
std::vector<TestOpDesc> MakeScaleChain(float scale, float bias) {
  std::vector<TestOpDesc> ops;
  ops.push_back({"relu", {{"X", {"x"}}}, {{"Out", {"relu_out"}}}});
  TestOpDesc s;
  s.type = "scale";
  s.inputs = {{"X", {"relu_out"}}};
  s.outputs = {{"Out", {"out"}}};
  s.float_attrs = {{"scale", scale}, {"bias", bias}};
  s.bool_attrs = {{"bias_after_scale", true}};
  ops.push_back(s);
  return ops;
}

}  // namespace

// Identity scale (scale=1, bias=0) must be eliminated: the pre-op's output
// is rewired directly to `out` and the scale node disappears.
TEST(IdentityScaleEliminatePass, eliminate_identity_scale) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeScaleChain(1.0f, 0.0f), {}, scope.get());

  ASSERT_EQ(CountOp(*graph, "relu"), 1);
  ASSERT_EQ(CountOp(*graph, "scale"), 1);

  auto* pass = PassManager::Global().LookUp("identity_scale_eliminate_pass");
  ASSERT_NE(pass, nullptr);
  pass->Apply(graph);

  ASSERT_EQ(CountOp(*graph, "scale"), 0);
  ASSERT_EQ(CountOp(*graph, "relu"), 1);
}

// A non-identity scale (scale != 1) must be preserved.
TEST(IdentityScaleEliminatePass, keep_non_identity_scale) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeScaleChain(2.0f, 0.0f), {}, scope.get());

  ASSERT_EQ(CountOp(*graph, "scale"), 1);

  auto* pass = PassManager::Global().LookUp("identity_scale_eliminate_pass");
  ASSERT_NE(pass, nullptr);
  pass->Apply(graph);

  ASSERT_EQ(CountOp(*graph, "scale"), 1);
  ASSERT_EQ(CountOp(*graph, "relu"), 1);
}

// A scale with scale=1 but non-zero bias is not an identity and must be
// preserved.
TEST(IdentityScaleEliminatePass, keep_scale_with_bias) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeScaleChain(1.0f, 1.0f), {}, scope.get());

  auto* pass = PassManager::Global().LookUp("identity_scale_eliminate_pass");
  ASSERT_NE(pass, nullptr);
  pass->Apply(graph);

  ASSERT_EQ(CountOp(*graph, "scale"), 1);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
