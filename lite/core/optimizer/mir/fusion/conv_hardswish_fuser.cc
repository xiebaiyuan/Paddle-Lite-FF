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

#include "lite/core/optimizer/mir/fusion/conv_hardswish_fuser.h"
#include <memory>
#include <vector>
#include "lite/core/optimizer/mir/subgraph_matcher.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void ConvHardSwishFuser::BuildPattern() {
  // HardSwish(x) = x * clip(x + offset, min, max) / scale
  //
  // Composite pattern as exported by Paddle 2.x / x2paddle:
  //   conv2d → conv_out
  //            ├→ add(offset=3) → clip(min=0,max=6) ─┐
  //            └─────────────────────────────────────→ mul
  //             mul_out → div(scale=6)  OR  mul(1/scale)
  //
  // Note the graph often reuses the intermediate variable names (SSA), so we
  // only assert on op-IO relationships, not on unique var names.

  auto* input =
      VarNode("input")->assert_is_op_input(conv_type_, "Input")->AsInput();
  auto* filter =
      VarNode("filter")->assert_is_op_input(conv_type_, "Filter")->AsInput();
  PMNode* bias = nullptr;
  if (has_bias_) {
    bias = VarNode("bias")->assert_is_op_input(conv_type_, "Bias")->AsInput();
  }

  auto* conv_op = OpNode("conv2d", conv_type_)->AsIntermediate();
  auto* conv_out = VarNode("conv_out")
                       ->assert_is_op_output(conv_type_, "Output")
                       ->AsIntermediate();

  // conv_out → add(y=offset) → clip → mul ← conv_out
  auto* add_op = OpNode("add", "elementwise_add")->AsIntermediate();
  auto* add_y = VarNode("add_y")
                    ->assert_is_op_input("elementwise_add", "Y")
                    ->AsInput();
  auto* add_out = VarNode("add_out")
                      ->assert_is_op_output("elementwise_add", "Out")
                      ->assert_is_op_input("clip", "X")
                      ->AsIntermediate();

  auto* clip_op = OpNode("clip", "clip")->AsIntermediate();
  auto* clip_out = VarNode("clip_out")
                       ->assert_is_op_output("clip", "Out")
                       ->assert_is_op_input("elementwise_mul", "Y")
                       ->AsIntermediate();

  auto* mul_op = OpNode("mul", "elementwise_mul")->AsIntermediate();
  auto* mul_out = VarNode("mul_out")
                      ->assert_is_op_output("elementwise_mul", "Out")
                      ->AsIntermediate();

  // Tail: mul_out → mul(1/scale) → output  (x2paddle style, scale folded
  // into the constant as 1/scale). Paddle 2.6 exports use div(scale); we
  // match the mul form here since it is what x2paddle produces.
  auto* tail_mul_op = OpNode("tail_mul", "elementwise_mul")->AsIntermediate();
  auto* tail_mul_y = VarNode("tail_mul_y")
                         ->assert_is_op_input("elementwise_mul", "Y")
                         ->assert_is_persistable_var()
                         ->AsInput();
  auto* output = VarNode("output")
                     ->assert_is_op_output("elementwise_mul", "Out")
                     ->AsOutput();

  // topology
  std::vector<PMNode*> conv_inputs{filter, input};
  conv_inputs >> *conv_op >> *conv_out;
  if (has_bias_) {
    *bias >> *conv_op;
  }

  // branch 1: add(y) → clip → mul
  *conv_out >> *add_op >> *add_out >> *clip_op >> *clip_out >> *mul_op;
  *add_y >> *add_op;
  *conv_out >> *mul_op;

  // tail: mul_out → mul(1/scale) → output
  *mul_op >> *mul_out >> *tail_mul_op >> *output;
  *tail_mul_y >> *tail_mul_op;
}

void ConvHardSwishFuser::InsertNewNode(SSAGraph* graph,
                                        const key2nodes_t& matched) {
  auto op_desc = GenOpDesc(matched);
  auto conv_op = LiteOpRegistry::Global().Create(conv_type_);
  auto conv_old = matched.at("conv2d")->stmt()->op();
  auto* scope = conv_old->scope();
  auto& valid_places = conv_old->valid_places();
  conv_op->Attach(op_desc, scope);

  auto* new_op_node = graph->GraphCreateInstructNode(conv_op, valid_places);

  IR_NODE_LINK_TO(matched.at("input"), new_op_node);
  IR_NODE_LINK_TO(matched.at("filter"), new_op_node);
  if (has_bias_) {
    IR_NODE_LINK_TO(matched.at("bias"), new_op_node);
  }
  IR_NODE_LINK_TO(new_op_node, matched.at("output"));
}

cpp::OpDesc ConvHardSwishFuser::GenOpDesc(const key2nodes_t& matched) {
  cpp::OpDesc op_desc = *matched.at("conv2d")->stmt()->op_info();
  op_desc.SetOutput("Output", {matched.at("output")->arg()->name});

  // Set hard_swish activation attributes using standard naming
  op_desc.SetAttr("with_act", true);
  op_desc.SetAttr("act_type", std::string("hard_swish"));

  // Extract hard_swish parameters
  float offset = 3.0f;
  float scale = 6.0f;
  float threshold = 6.0f;

  auto* clip_op_info = matched.at("clip")->stmt()->op_info();
  if (clip_op_info) {
    if (clip_op_info->HasAttr("min")) {
      offset = -clip_op_info->GetAttr<float>("min");
    }
    if (clip_op_info->HasAttr("max")) {
      threshold = clip_op_info->GetAttr<float>("max");
    }
  }

  op_desc.SetAttr("hard_swish_threshold", threshold);
  op_desc.SetAttr("hard_swish_scale", scale);
  op_desc.SetAttr("hard_swish_offset", offset);

  return op_desc;
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
