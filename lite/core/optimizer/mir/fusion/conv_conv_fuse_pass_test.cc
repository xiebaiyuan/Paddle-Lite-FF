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
#include "lite/core/optimizer/mir/fusion/conv_conv_fuser.h"
#include "lite/core/optimizer/mir/fusion/fusion_pass_test_util.h"

namespace paddle {
namespace lite {
namespace mir {

// conv2d(1x1) → conv2d(1x1) folds into a single conv2d.
// conv0: [2,1,1,1], conv1: [2,2,1,1] → ic0=1, oc0=2, ic1=2, oc1=2;
// computation check: ic0*(oc1-oc0)=1*0=0 ≤ 0 → must skip!
// Use oc1=4: ic0*(oc1-oc0)=1*2=2 ≤ oc0*oc1=8 → fusable.
std::vector<TestOpDesc> MakeConvConv(bool bias0, bool bias1) {
  TestOpDesc conv0;
  conv0.type = "conv2d";
  conv0.inputs = {{"Input", {"x"}}, {"Filter", {"w0"}}};
  conv0.outputs = {{"Output", {"mid"}}};
  conv0.int_attrs = {{"groups", 1}};
  conv0.int_vector_attrs = {{"strides", {1, 1}},
                            {"paddings", {0, 0}},
                            {"dilations", {1, 1}}};
  if (bias0) conv0.inputs["Bias"] = {"b0"};
  std::vector<TestOpDesc> ops{conv0};

  TestOpDesc conv1;
  conv1.type = "conv2d";
  conv1.inputs = {{"Input", {"mid"}}, {"Filter", {"w1"}}};
  conv1.outputs = {{"Output", {"out"}}};
  conv1.int_attrs = {{"groups", 1}};
  conv1.int_vector_attrs = {{"strides", {1, 1}},
                            {"paddings", {0, 0}},
                            {"dilations", {1, 1}}};
  if (bias1) conv1.inputs["Bias"] = {"b1"};
  ops.push_back(conv1);
  return ops;
}

TEST(ConvConvFuser, fuse_conv_conv) {
  auto scope = std::make_shared<Scope>();
  // w0 [2,1,1,1], w1 [4,2,1,1]
  auto* w0 = scope->Var("w0")->GetMutable<lite::Tensor>();
  w0->Resize(DDim({2, 1, 1, 1}));
  for (int i = 0; i < 2; ++i) w0->mutable_data<float>()[i] = 1.0f;
  auto* w1 = scope->Var("w1")->GetMutable<lite::Tensor>();
  w1->Resize(DDim({4, 2, 1, 1}));
  for (int i = 0; i < 8; ++i) w1->mutable_data<float>()[i] = 1.0f;

  auto graph = BuildGraph(MakeConvConv(false, false), {"w0", "w1"},
                          scope.get());
  ASSERT_EQ(CountOp(*graph, "conv2d"), 2);

  fusion::ConvConvFuser fuser("conv2d", "conv2d", false, false, graph);
  ASSERT_EQ(fuser(graph.get()), 1u);

  ASSERT_EQ(CountOp(*graph, "conv2d"), 1);
}

TEST(ConvConvFuser, fuse_conv_conv_with_bias) {
  // ConvConvFuser::BuildPattern previously read conv0's *first* inlink as the
  // weight (conv0_in.front()). When the conv has a Bias input, the inlink
  // order is not guaranteed to put Filter first, so the weight-dims probe
  // could read a bias tensor and abort the fusion. This test locks the fixed
  // behavior: the pass must find the Filter by argname and fuse the pair.
  auto scope = std::make_shared<Scope>();
  auto* w0 = scope->Var("w0")->GetMutable<lite::Tensor>();
  w0->Resize(DDim({2, 1, 1, 1}));
  for (int i = 0; i < 2; ++i) w0->mutable_data<float>()[i] = 1.0f;
  auto* w1 = scope->Var("w1")->GetMutable<lite::Tensor>();
  w1->Resize(DDim({4, 2, 1, 1}));
  for (int i = 0; i < 8; ++i) w1->mutable_data<float>()[i] = 1.0f;
  auto* b0 = scope->Var("b0")->GetMutable<lite::Tensor>();
  b0->Resize(DDim({2}));
  b0->mutable_data<float>()[0] = 0.5f;
  b0->mutable_data<float>()[1] = 0.5f;
  auto* b1 = scope->Var("b1")->GetMutable<lite::Tensor>();
  b1->Resize(DDim({4}));
  for (int i = 0; i < 4; ++i) b1->mutable_data<float>()[i] = 1.0f;

  auto graph = BuildGraph(MakeConvConv(true, true), {"w0", "w1", "b0", "b1"},
                          scope.get());
  ASSERT_EQ(CountOp(*graph, "conv2d"), 2);

  fusion::ConvConvFuser fuser("conv2d", "conv2d", true, true, graph);
  ASSERT_EQ(fuser(graph.get()), 1u);
  ASSERT_EQ(CountOp(*graph, "conv2d"), 1);
}

TEST(ConvConvFuser, skip_conv_with_activation) {
  // A fused conv0 that already carries an activation (with_act=true, e.g.
  // relu) must not participate in conv+conv fusion: recomputing the weights
  // would silently swallow conv0's activation. The pass must skip the pair.
  auto scope = std::make_shared<Scope>();
  auto* w0 = scope->Var("w0")->GetMutable<lite::Tensor>();
  w0->Resize(DDim({2, 1, 1, 1}));
  w0->mutable_data<float>()[0] = 1.0f;
  w0->mutable_data<float>()[1] = 1.0f;
  auto* w1 = scope->Var("w1")->GetMutable<lite::Tensor>();
  w1->Resize(DDim({4, 2, 1, 1}));
  for (int i = 0; i < 8; ++i) w1->mutable_data<float>()[i] = 1.0f;

  std::vector<TestOpDesc> ops = MakeConvConv(false, false);
  ops[0].bool_attrs = {{"with_act", true}};
  ops[0].str_attrs = {{"act_type", "relu"}};
  auto graph = BuildGraph(ops, {"w0", "w1"}, scope.get());
  fusion::ConvConvFuser fuser("conv2d", "conv2d", false, false, graph);
  ASSERT_EQ(fuser(graph.get()), 0u);
  ASSERT_EQ(CountOp(*graph, "conv2d"), 2);
}

TEST(ConvConvFuser, skip_conv_with_residual) {
  // A conv0 with a ResidualData input feeds its own residual path into the
  // fused output; folding conv0 into conv1 would break the residual semantics.
  auto scope = std::make_shared<Scope>();
  auto* w0 = scope->Var("w0")->GetMutable<lite::Tensor>();
  w0->Resize(DDim({2, 1, 1, 1}));
  w0->mutable_data<float>()[0] = 1.0f;
  w0->mutable_data<float>()[1] = 1.0f;
  auto* w1 = scope->Var("w1")->GetMutable<lite::Tensor>();
  w1->Resize(DDim({4, 2, 1, 1}));
  for (int i = 0; i < 8; ++i) w1->mutable_data<float>()[i] = 1.0f;
  auto* res = scope->Var("res")->GetMutable<lite::Tensor>();
  res->Resize(DDim({1, 2, 1, 1}));
  res->mutable_data<float>()[0] = 0.25f;
  res->mutable_data<float>()[1] = 0.25f;

  std::vector<TestOpDesc> ops = MakeConvConv(false, false);
  ops[0].inputs["ResidualData"] = {"res"};
  auto graph = BuildGraph(ops, {"w0", "w1", "res"}, scope.get());
  fusion::ConvConvFuser fuser("conv2d", "conv2d", false, false, graph);
  ASSERT_EQ(fuser(graph.get()), 0u);
  ASSERT_EQ(CountOp(*graph, "conv2d"), 2);
}

TEST(ConvConvFuser, skip_non_1x1_second_conv) {
  auto scope = std::make_shared<Scope>();
  // second conv 3x3 → BuildPattern never builds (kw != 1).
  auto* w0 = scope->Var("w0")->GetMutable<lite::Tensor>();
  w0->Resize(DDim({2, 1, 1, 1}));
  w0->mutable_data<float>()[0] = 1.0f;
  w0->mutable_data<float>()[1] = 1.0f;
  auto* w1 = scope->Var("w1")->GetMutable<lite::Tensor>();
  w1->Resize(DDim({4, 2, 3, 3}));
  for (int i = 0; i < 36; ++i) w1->mutable_data<float>()[i] = 1.0f;

  std::vector<TestOpDesc> ops = MakeConvConv(false, false);
  ops[1].int_vector_attrs = {{"strides", {1, 1}},
                             {"paddings", {1, 1}},
                             {"dilations", {1, 1}}};
  auto graph = BuildGraph(ops, {"w0", "w1"}, scope.get());
  fusion::ConvConvFuser fuser("conv2d", "conv2d", false, false, graph);
  ASSERT_EQ(fuser(graph.get()), 0u);
  ASSERT_EQ(CountOp(*graph, "conv2d"), 2);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
