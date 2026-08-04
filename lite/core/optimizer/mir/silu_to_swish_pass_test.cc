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
#include "lite/core/optimizer/mir/fusion/fusion_pass_test_util.h"
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/optimizer/mir/silu_to_swish_pass.h"

namespace paddle {
namespace lite {
namespace mir {

TEST(SiluToSwishPass, convert_silu_to_swish) {
  std::vector<TestOpDesc> ops;
  ops.push_back({"silu", {{"X", {"x"}}}, {{"Out", {"out"}}}});
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(ops, {}, scope.get());
  ASSERT_EQ(CountOp(*graph, "silu"), 1);

  SiluToSwishPass pass;
  pass.Apply(graph);

  ASSERT_EQ(CountOp(*graph, "silu"), 0);
  ASSERT_EQ(CountOp(*graph, "swish"), 1);
}

TEST(SiluToSwishPass, keep_other_ops) {
  // non-silu ops must be untouched.
  std::vector<TestOpDesc> ops;
  ops.push_back({"relu", {{"X", {"x"}}}, {{"Out", {"out"}}}});
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(ops, {}, scope.get());

  SiluToSwishPass pass;
  pass.Apply(graph);

  ASSERT_EQ(CountOp(*graph, "relu"), 1);
  ASSERT_EQ(CountOp(*graph, "swish"), 0);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
