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
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/optimizer/mir/fusion/fusion_pass_test_util.h"
#include "lite/core/optimizer/mir/fusion/sigmoid_elementmul_fuser.h"

namespace paddle {
namespace lite {
namespace mir {

// x → sigmoid → sigmoid_out; x → elementwise_mul(sigmoid_out) → swish
std::vector<TestOpDesc> MakeSigmoidMul() {
  return {TestOpDesc{"sigmoid", {{"X", {"x"}}}, {{"Out", {"sig_out"}}}},
          TestOpDesc{"elementwise_mul",
                     {{"X", {"x"}}, {"Y", {"sig_out"}}},
                     {{"Out", {"out"}}},
                     {{"axis", -1}},
                     {},
                     {},
                     {}}};
}

TEST(SigmoidElementmulFuser, fuse_to_swish) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeSigmoidMul(), {}, scope.get());
  ASSERT_EQ(CountOp(*graph, "sigmoid"), 1);
  ASSERT_EQ(CountOp(*graph, "elementwise_mul"), 1);

  fusion::SigmoidElementmulFuser fuser;
  ASSERT_EQ(fuser(graph.get()), 1u);

  ASSERT_EQ(CountOp(*graph, "sigmoid"), 0);
  ASSERT_EQ(CountOp(*graph, "elementwise_mul"), 0);
  ASSERT_EQ(CountOp(*graph, "swish"), 1);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
