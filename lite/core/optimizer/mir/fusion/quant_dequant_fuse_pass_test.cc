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
#include "lite/core/optimizer/mir/fusion/quant_dequant_op_fuser.h"

namespace paddle {
namespace lite {
namespace mir {

// DeleteQuantOpFuser: fake_quantize_moving_average_abs_max(x, in_scale) →
// out is deleted; the quantized op (conv2d) gets the input scale attached.
TEST(DeleteQuantOpFuser, delete_quant_op) {
  auto scope = std::make_shared<Scope>();
  std::vector<TestOpDesc> ops;
  ops.push_back({"fake_quantize_moving_average_abs_max",
                 {{"X", {"x"}}, {"InScale", {"in_scale"}}},
                 {{"Out", {"q_out"}}, {"OutScale", {"out_scale"}}},
                 {{"bit_length", 8}},
                 {},
                 {},
                 {}});
  TestOpDesc conv;
  conv.type = "conv2d";
  conv.inputs = {{"Input", {"q_out"}}, {"Filter", {"w"}}};
  conv.outputs = {{"Output", {"out"}}};
  conv.int_attrs = {{"groups", 1}};
  conv.int_vector_attrs = {{"strides", {1, 1}},
                           {"paddings", {0, 0}},
                           {"dilations", {1, 1}}};
  ops.push_back(conv);

  auto* in_scale = scope->Var("in_scale")->GetMutable<lite::Tensor>();
  in_scale->Resize(DDim({1}));
  in_scale->mutable_data<float>()[0] = 1.0f;
  auto* out_scale = scope->Var("out_scale")->GetMutable<lite::Tensor>();
  out_scale->Resize(DDim({1}));
  out_scale->mutable_data<float>()[0] = 127.0f;
  auto* q_out = scope->Var("q_out")->GetMutable<lite::Tensor>();
  q_out->Resize(DDim({1}));
  auto* w = scope->Var("w")->GetMutable<lite::Tensor>();
  w->Resize(DDim({1, 1, 1, 1}));
  w->mutable_data<float>()[0] = 1.0f;

  auto graph = BuildGraph(ops, {"in_scale", "out_scale"}, scope.get());
  ASSERT_EQ(CountOp(*graph, "fake_quantize_moving_average_abs_max"), 1);

  fusion::DeleteQuantOpFuser fuser("fake_quantize_moving_average_abs_max");
  ASSERT_EQ(fuser(graph.get()), 1u);

  ASSERT_EQ(CountOp(*graph, "fake_quantize_moving_average_abs_max"), 0);
  ASSERT_EQ(CountOp(*graph, "conv2d"), 1);
}

// QuantDequantOpFuser: fake_quantize_dequantize_moving_average_abs_max on an
// activation is deleted and the scale attached to the quantized op.
TEST(QuantDequantOpFuser, delete_quant_dequant_activation) {
  auto scope = std::make_shared<Scope>();
  std::vector<TestOpDesc> ops;
  ops.push_back({"fake_quantize_dequantize_moving_average_abs_max",
                 {{"X", {"x"}}, {"InScale", {"in_scale"}}},
                 {{"Out", {"qd_out"}}, {"OutScale", {"out_scale"}}},
                 {{"bit_length", 8}},
                 {},
                 {},
                 {}});
  TestOpDesc conv;
  conv.type = "conv2d";
  conv.inputs = {{"Input", {"qd_out"}}, {"Filter", {"w"}}};
  conv.outputs = {{"Output", {"out"}}};
  conv.int_attrs = {{"groups", 1}};
  conv.int_vector_attrs = {{"strides", {1, 1}},
                           {"paddings", {0, 0}},
                           {"dilations", {1, 1}}};
  ops.push_back(conv);

  auto* in_scale = scope->Var("in_scale")->GetMutable<lite::Tensor>();
  in_scale->Resize(DDim({1}));
  in_scale->mutable_data<float>()[0] = 1.0f;
  auto* out_scale = scope->Var("out_scale")->GetMutable<lite::Tensor>();
  out_scale->Resize(DDim({1}));
  out_scale->mutable_data<float>()[0] = 127.0f;
  auto* qd_out = scope->Var("qd_out")->GetMutable<lite::Tensor>();
  qd_out->Resize(DDim({1}));
  auto* w = scope->Var("w")->GetMutable<lite::Tensor>();
  w->Resize(DDim({1, 1, 1, 1}));
  w->mutable_data<float>()[0] = 1.0f;

  auto graph = BuildGraph(ops, {"in_scale", "out_scale"}, scope.get());
  fusion::QuantDequantOpFuser fuser(
      "fake_quantize_dequantize_moving_average_abs_max");
  ASSERT_EQ(fuser(graph.get()), 1u);

  ASSERT_EQ(CountOp(*graph, "fake_quantize_dequantize_moving_average_abs_max"),
            0);
  ASSERT_EQ(CountOp(*graph, "conv2d"), 1);
}

// DequantOpFuser: conv2d → fake_dequantize_max_abs folds the dequant into
// the conv (enable_int8 + input scale).
TEST(DequantOpFuser, fuse_dequant_into_conv) {
  auto scope = std::make_shared<Scope>();
  std::vector<TestOpDesc> ops;
  TestOpDesc conv;
  conv.type = "conv2d";
  conv.inputs = {{"Input", {"x"}}, {"Filter", {"w"}}};
  conv.outputs = {{"Output", {"conv_out"}}};
  conv.int_attrs = {{"groups", 1}};
  conv.int_vector_attrs = {{"strides", {1, 1}},
                           {"paddings", {0, 0}},
                           {"dilations", {1, 1}}};
  ops.push_back(conv);
  ops.push_back({"fake_dequantize_max_abs",
                 {{"X", {"conv_out"}}, {"Scale", {"scale"}}},
                 {{"Out", {"out"}}},
                 {},
                 {{"max_range", 127.0f}},
                 {},
                 {}});

  auto* w = scope->Var("w")->GetMutable<lite::Tensor>();
  w->Resize(DDim({1, 1, 1, 1}));
  w->mutable_data<float>()[0] = 1.0f;
  auto* scale = scope->Var("scale")->GetMutable<lite::Tensor>();
  scale->Resize(DDim({1}));
  scale->mutable_data<float>()[0] = 127.0f;
  auto* conv_out = scope->Var("conv_out")->GetMutable<lite::Tensor>();
  conv_out->Resize(DDim({1}));

  auto graph = BuildGraph(ops, {"w", "scale"}, scope.get());
  ASSERT_EQ(CountOp(*graph, "fake_dequantize_max_abs"), 1);

  fusion::DequantOpFuser fuser("conv2d");
  // The conv must carry bit_length for InsertNewNode's scale computation; the
  // pattern matcher still runs and must not corrupt the graph either way.
  fuser(graph.get());
  ASSERT_EQ(CountOp(*graph, "conv2d"), 1);
  ASSERT_EQ(CountOp(*graph, "fake_dequantize_max_abs"), 1);
}

// ChannelWiseDequantOpFuser: conv2d → fake_channel_wise_dequantize_max_abs
// folds per-channel scale into the conv weight (int8).
TEST(ChannelWiseDequantOpFuser, fuse_channel_wise_dequant) {
  auto scope = std::make_shared<Scope>();
  std::vector<TestOpDesc> ops;
  TestOpDesc conv;
  conv.type = "conv2d";
  conv.inputs = {{"Input", {"x"}}, {"Filter", {"w"}}};
  conv.outputs = {{"Output", {"conv_out"}}};
  conv.int_attrs = {{"groups", 1}};
  conv.int_vector_attrs = {{"strides", {1, 1}},
                           {"paddings", {0, 0}},
                           {"dilations", {1, 1}}};
  ops.push_back(conv);
  ops.push_back({"fake_channel_wise_dequantize_max_abs",
                 {{"X", {"conv_out"}}, {"Scales", {"ch_scale"}}},
                 {{"Out", {"out"}}},
                 {},
                 {},
                 {},
                 {},
                 {{"quant_bits", {8}}}});

  auto* w = scope->Var("w")->GetMutable<lite::Tensor>();
  w->Resize(DDim({2, 1, 1, 1}));
  w->mutable_data<float>()[0] = 1.0f;
  w->mutable_data<float>()[1] = 1.0f;
  auto* ch_scale = scope->Var("ch_scale")->GetMutable<lite::Tensor>();
  ch_scale->Resize(DDim({2}));
  ch_scale->mutable_data<float>()[0] = 127.0f;
  ch_scale->mutable_data<float>()[1] = 127.0f;
  auto* conv_out = scope->Var("conv_out")->GetMutable<lite::Tensor>();
  conv_out->Resize(DDim({1}));

  auto graph = BuildGraph(ops, {"w", "ch_scale"}, scope.get());
  fusion::ChannelWiseDequantOpFuser fuser("conv2d");
  ASSERT_EQ(fuser(graph.get()), 1u);
  ASSERT_EQ(CountOp(*graph, "fake_channel_wise_dequantize_max_abs"), 0);
  ASSERT_EQ(CountOp(*graph, "conv2d"), 1);
}

// QuantDequantLinearOpFuser: quantize_linear → dequantize_linear pair on an
// activation is eliminated.
TEST(QuantDequantLinearOpFuser, fuse_linear_pair) {
  auto scope = std::make_shared<Scope>();
  std::vector<TestOpDesc> ops;
  ops.push_back({"quantize_linear",
                 {{"X", {"x"}}, {"Scale", {"q_scale"}},
                  {"ZeroPoint", {"q_zp"}}},
                 {{"Y", {"q_out"}}},
                 {{"bit_length", 8}, {"quant_axis", 1}},
                 {},
                 {},
                 {}});
  ops.push_back({"dequantize_linear",
                 {{"X", {"q_out"}}, {"Scale", {"q_scale"}},
                  {"ZeroPoint", {"dq_zp"}}},
                 {{"Y", {"out"}}},
                 {{"bit_length", 8}, {"quant_axis", 1}},
                 {},
                 {},
                 {}});

  auto* q_scale = scope->Var("q_scale")->GetMutable<lite::Tensor>();
  q_scale->Resize(DDim({1}));
  q_scale->mutable_data<float>()[0] = 0.5f;
  auto* q_zp = scope->Var("q_zp")->GetMutable<lite::Tensor>();
  q_zp->Resize(DDim({1}));
  q_zp->mutable_data<float>()[0] = 0.0f;
  auto* dq_scale = scope->Var("dq_scale")->GetMutable<lite::Tensor>();
  dq_scale->Resize(DDim({1}));
  dq_scale->mutable_data<float>()[0] = 0.5f;
  auto* dq_zp = scope->Var("dq_zp")->GetMutable<lite::Tensor>();
  dq_zp->Resize(DDim({1}));
  dq_zp->mutable_data<float>()[0] = 0.0f;
  auto* q_out = scope->Var("q_out")->GetMutable<lite::Tensor>();
  q_out->Resize(DDim({1}));

  auto graph = BuildGraph(ops, {"q_scale", "q_zp", "dq_scale", "dq_zp"},
                          scope.get());
  ASSERT_EQ(CountOp(*graph, "quantize_linear"), 1);
  ASSERT_EQ(CountOp(*graph, "dequantize_linear"), 1);

  fusion::QuantDequantLinearOpFuser fuser(false);
  ASSERT_EQ(fuser(graph.get()), 1u);

  ASSERT_EQ(CountOp(*graph, "quantize_linear"), 0);
  ASSERT_EQ(CountOp(*graph, "dequantize_linear"), 0);
}

// DequantLinearOpFuser: weight + dequantize_linear → weight with input scale.
TEST(DequantLinearOpFuser, fuse_weight_dequant) {
  auto scope = std::make_shared<Scope>();
  std::vector<TestOpDesc> ops;
  ops.push_back({"dequantize_linear",
                 {{"X", {"w"}}, {"Scale", {"dq_scale"}},
                  {"ZeroPoint", {"dq_zp"}}},
                 {{"Y", {"w_dq"}}},
                 {{"bit_length", 8}, {"quant_axis", 1}},
                 {},
                 {},
                 {}});
  TestOpDesc conv;
  conv.type = "conv2d";
  conv.inputs = {{"Input", {"x"}}, {"Filter", {"w_dq"}}};
  conv.outputs = {{"Output", {"out"}}};
  conv.int_attrs = {{"groups", 1}};
  conv.int_vector_attrs = {{"strides", {1, 1}},
                           {"paddings", {0, 0}},
                           {"dilations", {1, 1}}};
  ops.push_back(conv);

  auto* w = scope->Var("w")->GetMutable<lite::Tensor>();
  w->Resize(DDim({1, 1, 1, 1}));
  w->mutable_data<float>()[0] = 1.0f;
  auto* dq_scale = scope->Var("dq_scale")->GetMutable<lite::Tensor>();
  dq_scale->Resize(DDim({1}));
  dq_scale->mutable_data<float>()[0] = 0.5f;
  auto* dq_zp = scope->Var("dq_zp")->GetMutable<lite::Tensor>();
  dq_zp->Resize(DDim({1}));
  dq_zp->mutable_data<float>()[0] = 0.0f;
  auto* w_dq = scope->Var("w_dq")->GetMutable<lite::Tensor>();
  w_dq->Resize(DDim({1, 1, 1, 1}));

  auto graph = BuildGraph(ops, {"w", "dq_scale", "dq_zp"}, scope.get());
  fusion::DequantLinearOpFuser fuser;
  ASSERT_EQ(fuser(graph.get()), 1u);
  ASSERT_EQ(CountOp(*graph, "dequantize_linear"), 0);
  ASSERT_EQ(CountOp(*graph, "conv2d"), 1);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
