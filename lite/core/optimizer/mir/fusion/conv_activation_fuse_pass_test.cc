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
#include "lite/core/optimizer/mir/fusion/conv_activation_fuser.h"
#include "lite/core/optimizer/mir/fusion/fusion_pass_test_util.h"

namespace paddle {
namespace lite {
namespace mir {

// True if any stmt node of `op_type` binds the var `input_name` under the
// input slot `param`. Used to lock fused-op input bindings (e.g. prelu's
// Prelu_alpha) that a generic attribute refactor could silently drop.
bool OpHasInput(const SSAGraph& graph,
                const std::string& op_type,
                const std::string& param,
                const std::string& input_name) {
  for (auto& node : graph.nodes()) {
    if (!node.IsStmt()) continue;
    auto* info = node.stmt()->op_info();
    if (info->Type() != op_type) continue;
    if (!info->HasInput(param)) continue;
    for (const auto& name : info->Input(param)) {
      if (name == input_name) return true;
    }
  }
  return false;
}

// conv2d(+bias) → act must fold into a single conv with the activation
// attributes attached (act_type / with_act + per-act attrs).
std::vector<TestOpDesc> MakeConvAct(const std::string& act_type,
                                    bool with_bias,
                                    bool with_alpha = false,
                                    bool conv_has_prelu_alpha = false) {
  TestOpDesc conv;
  conv.type = "conv2d";
  conv.inputs = {{"Input", {"x"}}, {"Filter", {"w"}}};
  conv.outputs = {{"Output", {"conv_out"}}};
  conv.int_attrs = {{"groups", 1}};
  conv.int_vector_attrs = {{"strides", {1, 1}},
                           {"paddings", {0, 0}},
                           {"dilations", {1, 1}}};
  if (with_bias) conv.inputs["Bias"] = {"b"};
  // When the act is prelu, the *fused* conv needs a Prelu_alpha input for
  // ConvOp::AttachImpl to succeed; the fuser reuses the act's Alpha input.
  if (conv_has_prelu_alpha) conv.inputs["Prelu_alpha"] = {"alpha"};
  std::vector<TestOpDesc> ops{conv};

  TestOpDesc act;
  act.type = act_type;
  act.inputs = {{"X", {"conv_out"}}};
  act.outputs = {{"Out", {"out"}}};
  if (act_type == "leaky_relu") act.float_attrs = {{"alpha", 0.1f}};
  if (act_type == "relu6") act.float_attrs = {{"threshold", 6.0f}};
  if (act_type == "hard_swish") {
    act.float_attrs = {{"threshold", 6.0f}, {"scale", 6.0f}, {"offset", 3.0f}};
  }
  if (act_type == "hard_sigmoid") {
    act.float_attrs = {{"slope", 0.2f}, {"offset", 0.5f}};
  }
  if (act_type == "prelu") {
    act.str_attrs = {{"mode", "all"}};
    if (with_alpha) act.inputs["Alpha"] = {"alpha"};
  }
  if (act_type == "swish") act.float_attrs = {{"beta", 1.0f}};
  if (act_type == "gelu") act.bool_attrs = {{"approximate", false}};
  ops.push_back(act);
  return ops;
}

TEST(ConvActivationFuser, fuse_relu) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeConvAct("relu", true), {}, scope.get());
  ASSERT_EQ(CountOp(*graph, "conv2d"), 1);
  ASSERT_EQ(CountOp(*graph, "relu"), 1);

  fusion::ConvActivationFuser fuser("conv2d", "relu", true, false);
  ASSERT_EQ(fuser(graph.get()), 1u);

  ASSERT_EQ(CountOp(*graph, "conv2d"), 1);
  ASSERT_EQ(CountOp(*graph, "relu"), 0);
  // conv carries the fused activation marker (bool attrs).
  ASSERT_TRUE(OpHasBoolAttr(*graph, "conv2d", "with_act", true) ||
              OpHasBoolAttr(*graph, "conv2d", "fuse_relu", true));
}

TEST(ConvActivationFuser, fuse_leaky_relu) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeConvAct("leaky_relu", false), {}, scope.get());
  fusion::ConvActivationFuser fuser("conv2d", "leaky_relu", false, false);
  ASSERT_EQ(fuser(graph.get()), 1u);
  ASSERT_EQ(CountOp(*graph, "conv2d"), 1);
  ASSERT_EQ(CountOp(*graph, "leaky_relu"), 0);
}

TEST(ConvActivationFuser, fuse_relu6) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeConvAct("relu6", true), {}, scope.get());
  fusion::ConvActivationFuser fuser("conv2d", "relu6", true, false);
  ASSERT_EQ(fuser(graph.get()), 1u);
  ASSERT_EQ(CountOp(*graph, "conv2d"), 1);
  ASSERT_EQ(CountOp(*graph, "relu6"), 0);
}

TEST(ConvActivationFuser, fuse_hard_swish) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeConvAct("hard_swish", true), {}, scope.get());
  fusion::ConvActivationFuser fuser("conv2d", "hard_swish", true, false);
  ASSERT_EQ(fuser(graph.get()), 1u);
  ASSERT_EQ(CountOp(*graph, "conv2d"), 1);
  ASSERT_EQ(CountOp(*graph, "hard_swish"), 0);
}

TEST(ConvActivationFuser, fuse_hard_sigmoid) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeConvAct("hard_sigmoid", false), {}, scope.get());
  fusion::ConvActivationFuser fuser("conv2d", "hard_sigmoid", false, false);
  ASSERT_EQ(fuser(graph.get()), 1u);
  ASSERT_EQ(CountOp(*graph, "conv2d"), 1);
  ASSERT_EQ(CountOp(*graph, "hard_sigmoid"), 0);
}

TEST(ConvActivationFuser, fuse_prelu_with_alpha) {
  auto scope = std::make_shared<Scope>();
  // prelu needs an Alpha input tensor.
  auto* alpha_t = scope->Var("alpha")->GetMutable<lite::Tensor>();
  alpha_t->Resize(DDim({1}));
  alpha_t->mutable_data<float>()[0] = 0.25f;
  // The conv already carries Prelu_alpha (as the fused form requires), and
  // the prelu op reuses the same Alpha var.
  auto graph =
      BuildGraph(MakeConvAct("prelu", true, true, false), {}, scope.get());
  fusion::ConvActivationFuser fuser("conv2d", "prelu", true, true);
  ASSERT_EQ(fuser(graph.get()), 1u);
  ASSERT_EQ(CountOp(*graph, "conv2d"), 1);
  ASSERT_EQ(CountOp(*graph, "prelu"), 0);
  // The fused conv's OpDesc must bind the alpha tensor as its Prelu_alpha
  // input — the fused kernel resolves its per-channel alpha from that input.
  // Regression: the ApplyActivationAttributes refactor dropped this binding,
  // leaving the fused op referencing no alpha tensor (empty/uninitialized
  // alpha at runtime).
  ASSERT_TRUE(OpHasInput(*graph, "conv2d", "Prelu_alpha", "alpha"))
      << "fused conv must carry Prelu_alpha input bound to the alpha var";
}

TEST(ConvActivationFuser, fuse_sigmoid) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeConvAct("sigmoid", true), {}, scope.get());
  fusion::ConvActivationFuser fuser("conv2d", "sigmoid", true, false);
  ASSERT_EQ(fuser(graph.get()), 1u);
  ASSERT_EQ(CountOp(*graph, "sigmoid"), 0);
}

TEST(ConvActivationFuser, fuse_tanh) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeConvAct("tanh", false), {}, scope.get());
  fusion::ConvActivationFuser fuser("conv2d", "tanh", false, false);
  ASSERT_EQ(fuser(graph.get()), 1u);
  ASSERT_EQ(CountOp(*graph, "tanh"), 0);
}

TEST(ConvActivationFuser, fuse_swish) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeConvAct("swish", true), {}, scope.get());
  fusion::ConvActivationFuser fuser("conv2d", "swish", true, false);
  ASSERT_EQ(fuser(graph.get()), 1u);
  ASSERT_EQ(CountOp(*graph, "swish"), 0);
}

TEST(ConvActivationFuser, fuse_gelu) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeConvAct("gelu", true), {}, scope.get());
  fusion::ConvActivationFuser fuser("conv2d", "gelu", true, false);
  ASSERT_EQ(fuser(graph.get()), 1u);
  ASSERT_EQ(CountOp(*graph, "gelu"), 0);
}

TEST(ConvActivationFuser, skip_wrong_conv_type) {
  // Pattern is built for conv2d; a gelu without conv must not match.
  auto scope = std::make_shared<Scope>();
  std::vector<TestOpDesc> ops;
  ops.push_back({"gelu",
                 {{"X", {"x"}}},
                 {{"Out", {"out"}}},
                 {},
                 {},
                 {{"approximate", false}},
                 {}});
  auto graph = BuildGraph(ops, {}, scope.get());
  fusion::ConvActivationFuser fuser("conv2d", "gelu", false, false);
  ASSERT_EQ(fuser(graph.get()), 0u);
  ASSERT_EQ(CountOp(*graph, "gelu"), 1);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
