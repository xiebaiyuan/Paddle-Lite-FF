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
#include "lite/core/optimizer/mir/fusion/instance_norm_activation_fuser.h"

namespace paddle {
namespace lite {
namespace mir {

// instance_norm → relu folds into instance_norm (activation_type=relu).
std::vector<TestOpDesc> MakeInstanceNormRelu() {
  TestOpDesc in;
  in.type = "instance_norm";
  in.inputs = {{"X", {"x"}}, {"Scale", {"scale"}}, {"Bias", {"bias"}}};
  in.outputs = {{"Y", {"in_out"}},
                {"SavedMean", {"saved_mean"}},
                {"SavedVariance", {"saved_var"}}};
  in.float_attrs = {{"epsilon", 1e-5f}};
  std::vector<TestOpDesc> ops{in};
  ops.push_back({"relu", {{"X", {"in_out"}}}, {{"Out", {"out"}}}});
  return ops;
}

TEST(InstanceNormActivationFuser, fuse_relu) {
  auto scope = std::make_shared<Scope>();
  auto* x = scope->Var("x")->GetMutable<lite::Tensor>();
  x->Resize(DDim({1, 2, 2, 2}));
  auto* scale = scope->Var("scale")->GetMutable<lite::Tensor>();
  scale->Resize(DDim({2}));
  auto* bias = scope->Var("bias")->GetMutable<lite::Tensor>();
  bias->Resize(DDim({2}));
  auto* sm = scope->Var("saved_mean")->GetMutable<lite::Tensor>();
  sm->Resize(DDim({2}));
  auto* sv = scope->Var("saved_var")->GetMutable<lite::Tensor>();
  sv->Resize(DDim({2}));

  auto graph = BuildGraph(MakeInstanceNormRelu(), {}, scope.get());
  ASSERT_EQ(CountOp(*graph, "instance_norm"), 1);
  ASSERT_EQ(CountOp(*graph, "relu"), 1);

  fusion::InstanceNormActivationFuser fuser("relu");
  ASSERT_EQ(fuser(graph.get()), 1u);

  ASSERT_EQ(CountOp(*graph, "instance_norm"), 1);
  ASSERT_EQ(CountOp(*graph, "relu"), 0);
  // instance_norm carries activation_type=relu.
  bool has_act = false;
  for (auto& node : graph->nodes()) {
    if (node.IsStmt() && node.stmt()->op_info()->Type() == "instance_norm" &&
        node.stmt()->op_info()->HasAttr("activation_type") &&
        node.stmt()->op_info()->GetAttr<std::string>("activation_type") ==
            "relu") {
      has_act = true;
    }
  }
  ASSERT_TRUE(has_act);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
