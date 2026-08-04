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
#include "lite/core/optimizer/mir/fusion/elementwise_add_activation_fuser.h"
#include "lite/core/optimizer/mir/fusion/fusion_pass_test_util.h"

namespace paddle {
namespace lite {
namespace mir {

// elementwise_add → relu folds into fusion_elementwise_add_activation.
std::vector<TestOpDesc> MakeEltAct(const std::string& elt_type) {
  return {TestOpDesc{elt_type,
                     {{"X", {"x"}}, {"Y", {"y"}}},
                     {{"Out", {"add_out"}}},
                     {{"axis", -1}},
                     {},
                     {},
                     {}},
          TestOpDesc{"relu", {{"X", {"add_out"}}}, {{"Out", {"out"}}}}};
}

TEST(ElementwiseActivationFuser, fuse_add_relu) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeEltAct("elementwise_add"), {}, scope.get());
  ASSERT_EQ(CountOp(*graph, "elementwise_add"), 1);
  ASSERT_EQ(CountOp(*graph, "relu"), 1);

  fusion::ElementwiseActivationFuser fuser("elementwise_add", "relu");
  ASSERT_EQ(fuser(graph.get()), 1u);

  ASSERT_EQ(CountOp(*graph, "fusion_elementwise_add_activation"), 1);
  ASSERT_EQ(CountOp(*graph, "elementwise_add"), 0);
  ASSERT_EQ(CountOp(*graph, "relu"), 0);
}

TEST(ElementwiseActivationFuser, fuse_sub_relu) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeEltAct("elementwise_sub"), {}, scope.get());
  fusion::ElementwiseActivationFuser fuser("elementwise_sub", "relu");
  ASSERT_EQ(fuser(graph.get()), 1u);
  ASSERT_EQ(CountOp(*graph, "fusion_elementwise_sub_activation"), 1);
}

TEST(ElementwiseActivationFuser, fuse_mul_relu) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeEltAct("elementwise_mul"), {}, scope.get());
  fusion::ElementwiseActivationFuser fuser("elementwise_mul", "relu");
  ASSERT_EQ(fuser(graph.get()), 1u);
  ASSERT_EQ(CountOp(*graph, "fusion_elementwise_mul_activation"), 1);
}

TEST(ElementwiseActivationFuser, skip_wrong_elt_type) {
  // fuser is built for elementwise_add, but the graph has elementwise_mul →
  // the pattern does not match, so no fusion happens.
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeEltAct("elementwise_mul"), {}, scope.get());
  fusion::ElementwiseActivationFuser fuser("elementwise_add", "relu");
  ASSERT_EQ(fuser(graph.get()), 0u);
  ASSERT_EQ(CountOp(*graph, "elementwise_mul"), 1);
  ASSERT_EQ(CountOp(*graph, "relu"), 1);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
