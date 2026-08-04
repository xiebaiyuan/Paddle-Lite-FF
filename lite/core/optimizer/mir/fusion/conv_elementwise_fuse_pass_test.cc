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
#include "lite/core/optimizer/mir/fusion/conv_elementwise_fuser.h"
#include "lite/core/optimizer/mir/fusion/fusion_pass_test_util.h"

namespace paddle {
namespace lite {
namespace mir {

// conv2d(+bias) → elementwise_add(Y=persistable bias) folds into conv.
std::vector<TestOpDesc> MakeConvAdd(bool with_bias) {
  TestOpDesc conv;
  conv.type = "conv2d";
  conv.inputs = {{"Input", {"x"}}, {"Filter", {"w"}}};
  conv.outputs = {{"Output", {"conv_out"}}};
  conv.int_attrs = {{"groups", 1}};
  conv.int_vector_attrs = {{"strides", {1, 1}},
                           {"paddings", {0, 0}},
                           {"dilations", {1, 1}}};
  if (with_bias) conv.inputs["Bias"] = {"conv_b"};
  std::vector<TestOpDesc> ops{conv};
  ops.push_back({"elementwise_add",
                 {{"X", {"conv_out"}}, {"Y", {"add_bias"}}},
                 {{"Out", {"out"}}},
                 {{"axis", 1}},
                 {},
                 {},
                 {}});
  return ops;
}

TEST(ConvElementwiseFuser, fuse_without_bias) {
  auto scope = std::make_shared<Scope>();
  // conv weight [1,1,1,1]; elementwise bias [1] (persistable)
  auto* w = scope->Var("w")->GetMutable<lite::Tensor>();
  w->Resize(DDim({1, 1, 1, 1}));
  w->mutable_data<float>()[0] = 1.0f;
  auto* ab = scope->Var("add_bias")->GetMutable<lite::Tensor>();
  ab->Resize(DDim({1}));
  ab->mutable_data<float>()[0] = 3.0f;

  auto graph = BuildGraph(MakeConvAdd(false), {"w", "add_bias"}, scope.get());
  ASSERT_EQ(CountOp(*graph, "conv2d"), 1);
  ASSERT_EQ(CountOp(*graph, "elementwise_add"), 1);

  fusion::ConvElementwiseFuser fuser("conv2d", false);
  ASSERT_EQ(fuser(graph.get()), 1u);

  // add bias became the conv bias; add eliminated.
  ASSERT_EQ(CountOp(*graph, "conv2d"), 1);
  ASSERT_EQ(CountOp(*graph, "elementwise_add"), 0);
  ASSERT_EQ(ab->data<float>()[0], 3.0f);
}

TEST(ConvElementwiseFuser, fuse_with_bias) {
  auto scope = std::make_shared<Scope>();
  auto* w = scope->Var("w")->GetMutable<lite::Tensor>();
  w->Resize(DDim({1, 1, 1, 1}));
  w->mutable_data<float>()[0] = 1.0f;
  auto* cb = scope->Var("conv_b")->GetMutable<lite::Tensor>();
  cb->Resize(DDim({1}));
  cb->mutable_data<float>()[0] = 1.0f;
  auto* ab = scope->Var("add_bias")->GetMutable<lite::Tensor>();
  ab->Resize(DDim({1}));
  ab->mutable_data<float>()[0] = 3.0f;

  auto graph =
      BuildGraph(MakeConvAdd(true), {"w", "conv_b", "add_bias"}, scope.get());
  fusion::ConvElementwiseFuser fuser("conv2d", true);
  ASSERT_EQ(fuser(graph.get()), 1u);

  // add bias += conv bias → 4.0
  ASSERT_EQ(CountOp(*graph, "elementwise_add"), 0);
  ASSERT_EQ(ab->data<float>()[0], 4.0f);
}

TEST(ConvElementwiseFuser, skip_wrong_bias_dims) {
  // add bias rank 2 (not [C] or [1,C,1,1]) must abort the fusion.
  auto scope = std::make_shared<Scope>();
  auto* w = scope->Var("w")->GetMutable<lite::Tensor>();
  w->Resize(DDim({1, 1, 1, 1}));
  w->mutable_data<float>()[0] = 1.0f;
  auto* ab = scope->Var("add_bias")->GetMutable<lite::Tensor>();
  ab->Resize(DDim({1, 1}));
  ab->mutable_data<float>()[0] = 3.0f;

  auto graph = BuildGraph(MakeConvAdd(false), {"w", "add_bias"}, scope.get());
  fusion::ConvElementwiseFuser fuser("conv2d", false);
  // FuseBase::operator() returns the number of *matched* subgraphs; the
  // dims check lives in InsertNewNode, so a match that bails out still
  // counts as 1. Assert on the graph instead.
  ASSERT_EQ(fuser(graph.get()), 1u);
  ASSERT_EQ(CountOp(*graph, "elementwise_add"), 1);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
