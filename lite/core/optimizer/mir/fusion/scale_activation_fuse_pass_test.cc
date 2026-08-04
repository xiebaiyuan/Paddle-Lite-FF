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
#include "lite/core/optimizer/mir/fusion/scale_activation_fuser.h"

namespace paddle {
namespace lite {
namespace mir {

// scale → relu folds into scale with activation_type=relu.
std::vector<TestOpDesc> MakeScaleAct(const std::string& act_type) {
  std::vector<TestOpDesc> ops;
  TestOpDesc scale;
  scale.type = "scale";
  scale.inputs = {{"X", {"x"}}};
  scale.outputs = {{"Out", {"scale_out"}}};
  scale.float_attrs = {{"scale", 2.0f}, {"bias", 1.0f}};
  scale.bool_attrs = {{"bias_after_scale", true}};
  ops.push_back(scale);

  TestOpDesc act;
  act.type = act_type;
  act.inputs = {{"X", {"scale_out"}}};
  act.outputs = {{"Out", {"out"}}};
  if (act_type == "leaky_relu") act.float_attrs = {{"alpha", 0.1f}};
  if (act_type == "relu6") act.float_attrs = {{"threshold", 6.0f}};
  ops.push_back(act);
  return ops;
}

TEST(ScaleActivationFuser, fuse_relu) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeScaleAct("relu"), {}, scope.get());
  ASSERT_EQ(CountOp(*graph, "scale"), 1);
  ASSERT_EQ(CountOp(*graph, "relu"), 1);

  fusion::ScaleActivationFuser fuser("relu");
  ASSERT_EQ(fuser(graph.get()), 1u);

  ASSERT_EQ(CountOp(*graph, "scale"), 1);
  ASSERT_EQ(CountOp(*graph, "relu"), 0);
}

TEST(ScaleActivationFuser, fuse_relu6) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeScaleAct("relu6"), {}, scope.get());
  fusion::ScaleActivationFuser fuser("relu6");
  ASSERT_EQ(fuser(graph.get()), 1u);
  ASSERT_EQ(CountOp(*graph, "relu6"), 0);
}

TEST(ScaleActivationFuser, fuse_leaky_relu) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeScaleAct("leaky_relu"), {}, scope.get());
  fusion::ScaleActivationFuser fuser("leaky_relu");
  ASSERT_EQ(fuser(graph.get()), 1u);
  ASSERT_EQ(CountOp(*graph, "leaky_relu"), 0);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
