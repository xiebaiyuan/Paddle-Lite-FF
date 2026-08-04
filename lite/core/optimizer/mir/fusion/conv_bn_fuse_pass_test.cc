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
#include "lite/core/optimizer/mir/fusion/conv_bn_fuser.h"
#include "lite/core/optimizer/mir/fusion/fusion_pass_test_util.h"

namespace paddle {
namespace lite {
namespace mir {

// conv2d(+bias) → batch_norm folds into a single conv (weight scaled by
// alpha = scale/sqrt(var+eps), bias folded into the bn Bias tensor which
// becomes the conv bias).
std::vector<TestOpDesc> MakeConvBn(bool with_bias, bool is_test = true) {
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

  TestOpDesc bn;
  bn.type = "batch_norm";
  bn.inputs = {{"X", {"conv_out"}},
               {"Scale", {"bn_scale"}},
               {"Bias", {"bn_bias"}},
               {"Mean", {"bn_mean"}},
               {"Variance", {"bn_var"}}};
  bn.outputs = {{"Y", {"bn_out"}},
                {"MeanOut", {"mean_out"}},
                {"VarianceOut", {"var_out"}},
                {"SavedMean", {"saved_mean"}},
                {"SavedVariance", {"saved_var"}}};
  bn.float_attrs = {{"epsilon", 1e-5f}, {"momentum", 0.9f}};
  bn.str_attrs = {{"data_layout", "NCHW"}};
  bn.int_attrs = {{"is_test", is_test ? 1 : 0}};
  bn.bool_attrs = {{"use_global_stats", true}};
  ops.push_back(bn);
  return ops;
}

TEST(ConvBNFuser, fuse_no_bias) {
  auto scope = std::make_shared<Scope>();
  // conv weight [1,1,1,1] = {2.0}; bn scale/bias/mean/var all [1]
  auto* w = scope->Var("w")->GetMutable<lite::Tensor>();
  w->Resize(DDim({1, 1, 1, 1}));
  w->mutable_data<float>()[0] = 2.0f;
  auto* s = scope->Var("bn_scale")->GetMutable<lite::Tensor>();
  s->Resize(DDim({1}));
  s->mutable_data<float>()[0] = 1.0f;
  auto* bb = scope->Var("bn_bias")->GetMutable<lite::Tensor>();
  bb->Resize(DDim({1}));
  bb->mutable_data<float>()[0] = 0.5f;
  auto* m = scope->Var("bn_mean")->GetMutable<lite::Tensor>();
  m->Resize(DDim({1}));
  m->mutable_data<float>()[0] = 0.0f;
  auto* v = scope->Var("bn_var")->GetMutable<lite::Tensor>();
  v->Resize(DDim({1}));
  v->mutable_data<float>()[0] = 1.0f;

  auto graph = BuildGraph(
      MakeConvBn(false), {"w", "bn_scale", "bn_bias", "bn_mean", "bn_var"},
      scope.get());
  ASSERT_EQ(CountOp(*graph, "conv2d"), 1);
  ASSERT_EQ(CountOp(*graph, "batch_norm"), 1);

  fusion::ConvBNFuser fuser("conv2d", "batch_norm", false);
  ASSERT_EQ(fuser(graph.get()), 1u);

  // alpha = scale/sqrt(var+eps) = 1/sqrt(1.00001) ≈ 0.999995; weight scaled.
  ASSERT_EQ(CountOp(*graph, "conv2d"), 1);
  ASSERT_EQ(CountOp(*graph, "batch_norm"), 0);
  ASSERT_NEAR(w->data<float>()[0], 2.0f / 1.000005f, 1e-3f);
  // bn bias became the conv bias: 0.5 + alpha*0 (no old conv bias) = 0.5
  ASSERT_NEAR(bb->data<float>()[0], 0.5f, 1e-5f);
}

TEST(ConvBNFuser, fuse_with_bias) {
  auto scope = std::make_shared<Scope>();
  auto* w = scope->Var("w")->GetMutable<lite::Tensor>();
  w->Resize(DDim({1, 1, 1, 1}));
  w->mutable_data<float>()[0] = 2.0f;
  auto* cb = scope->Var("conv_b")->GetMutable<lite::Tensor>();
  cb->Resize(DDim({1}));
  cb->mutable_data<float>()[0] = 1.0f;
  auto* s = scope->Var("bn_scale")->GetMutable<lite::Tensor>();
  s->Resize(DDim({1}));
  s->mutable_data<float>()[0] = 1.0f;
  auto* bb = scope->Var("bn_bias")->GetMutable<lite::Tensor>();
  bb->Resize(DDim({1}));
  bb->mutable_data<float>()[0] = 0.5f;
  auto* m = scope->Var("bn_mean")->GetMutable<lite::Tensor>();
  m->Resize(DDim({1}));
  m->mutable_data<float>()[0] = 0.0f;
  auto* v = scope->Var("bn_var")->GetMutable<lite::Tensor>();
  v->Resize(DDim({1}));
  v->mutable_data<float>()[0] = 1.0f;

  auto graph = BuildGraph(
      MakeConvBn(true),
      {"w", "conv_b", "bn_scale", "bn_bias", "bn_mean", "bn_var"},
      scope.get());

  fusion::ConvBNFuser fuser("conv2d", "batch_norm", true);
  ASSERT_EQ(fuser(graph.get()), 1u);

  ASSERT_EQ(CountOp(*graph, "batch_norm"), 0);
  // bias = 0.5 + alpha * 1.0 ≈ 1.5
  ASSERT_NEAR(bb->data<float>()[0], 1.5f, 1e-3f);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
