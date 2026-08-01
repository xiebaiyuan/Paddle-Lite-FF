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
#include "lite/core/optimizer/mir/fusion/conv_activation_fuser.h"
#include "lite/core/optimizer/mir/fusion/fusion_pass_test_util.h"

namespace paddle {
namespace lite {
namespace mir {

// conv2d(+bias) → gelu must fold into a conv with act_type=gelu.
std::vector<TestOpDesc> MakeConvGelu(bool with_bias) {
  TestOpDesc conv;
  conv.type = "conv2d";
  conv.inputs = {{"Input", {"x"}}, {"Filter", {"w"}}};
  conv.outputs = {{"Output", {"conv_out"}}};
  conv.int_attrs = {{"groups", 1}};
  conv.int_vector_attrs = {{"strides", {1, 1}},
                           {"paddings", {0, 0}},
                           {"dilations", {1, 1}}};
  if (with_bias) conv.inputs["Bias"] = {"b"};
  std::vector<TestOpDesc> ops{conv};
  ops.push_back({"gelu",
                 {{"X", {"conv_out"}}},
                 {{"Out", {"out"}}},
                 {},
                 {},
                 {{"approximate", false}},
                 {}});
  return ops;
}

TEST(ConvActivationFuser, fuse_gelu) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeConvGelu(false), {}, scope.get());
  ASSERT_EQ(CountOp(*graph, "conv2d"), 1);
  ASSERT_EQ(CountOp(*graph, "gelu"), 1);

  fusion::ConvActivationFuser fuser("conv2d", "gelu", false, false);
  size_t n = fuser(graph.get());
  ASSERT_EQ(n, 1u);

  // gelu folded into conv; no standalone gelu remains.
  ASSERT_EQ(CountOp(*graph, "conv2d"), 1);
  ASSERT_EQ(CountOp(*graph, "gelu"), 0);
}

TEST(ConvActivationFuser, fuse_gelu_with_bias) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeConvGelu(true), {}, scope.get());
  fusion::ConvActivationFuser fuser("conv2d", "gelu", true, false);
  size_t n = fuser(graph.get());
  ASSERT_EQ(n, 1u);
  ASSERT_EQ(CountOp(*graph, "gelu"), 0);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
