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

bool ConvHardSwishFuser::ValidateMatch(SSAGraph* graph,
                                       const key2nodes_t& matched) {
  // Only fuse the canonical hard_swish constants, otherwise the fused conv
  // would silently change numerics. offset comes from the add bias (3.0),
  // clip must be (0, 6), and the tail mul scale must be 1/6.
  auto add_bias_op = matched.at("add")->stmt()->op();
  auto* scope = add_bias_op->scope();

  float offset = 3.0f;
  bool have_offset = false;
  auto* add_y_node = matched.at("add_y");
  for (auto* producer : add_y_node->inlinks) {
    if (producer->IsStmt() &&
        producer->stmt()->op_info()->Type() == "fill_constant") {
      auto* op_info = producer->stmt()->op_info();
      if (op_info->HasAttr("value")) {
        auto t = op_info->GetAttrType("value");
        if (t == paddle::lite::OpDescAPI::AttrType::FLOAT) {
          offset = op_info->GetAttr<float>("value");
          have_offset = true;
        } else if (t == paddle::lite::OpDescAPI::AttrType::INT) {
          offset = static_cast<float>(op_info->GetAttr<int>("value"));
          have_offset = true;
        }
      }
    }
  }
  if (!have_offset) {
    auto* add_y_t = scope->FindMutableTensor(add_y_node->arg()->name);
    if (add_y_t != nullptr && add_y_t->numel() == 1 &&
        add_y_t->data<float>() != nullptr) {
      offset = add_y_t->data<float>()[0];
      have_offset = true;
    }
  }
  if (!have_offset || std::fabs(offset - 3.0f) > 1e-4f) {
    LOG(WARNING) << "conv_hardswish_fuse: non-canonical offset " << offset
                 << " (expected 3.0), skip";
    return false;
  }

  auto* clip_op_info = matched.at("clip")->stmt()->op_info();
  if (clip_op_info->HasAttr("min") &&
      std::fabs(clip_op_info->GetAttr<float>("min") - 0.0f) > 1e-4f) {
    LOG(WARNING) << "conv_hardswish_fuse: non-canonical clip min, skip";
    return false;
  }
  if (clip_op_info->HasAttr("max") &&
      std::fabs(clip_op_info->GetAttr<float>("max") - 6.0f) > 1e-4f) {
    LOG(WARNING) << "conv_hardswish_fuse: non-canonical clip max, skip";
    return false;
  }

  // tail_mul_y must be 1/6.
  auto* tail_t = scope->FindMutableTensor(matched.at("tail_mul_y")->arg()->name);
  if (tail_t == nullptr || tail_t->numel() != 1 ||
      tail_t->data<float>() == nullptr ||
      std::fabs(tail_t->data<float>()[0] - 1.0f / 6.0f) > 1e-4f) {
    LOG(WARNING) << "conv_hardswish_fuse: non-canonical tail scale, skip";
    return false;
  }

  // conv_out is an intermediate node removed after fusion; it must be
  // consumed only by add and mul (the two pattern branches). Any other
  // consumer would be orphaned.
  auto* conv_out_node = matched.at("conv_out");
  for (auto* consumer : conv_out_node->outlinks) {
    if (!consumer->IsStmt()) continue;
    if (consumer != matched.at("add") && consumer != matched.at("mul")) {
      LOG(WARNING) << "conv_hardswish_fuse: conv output "
                   << conv_out_node->arg()->name
                   << " has extra consumer "
                   << consumer->stmt()->op_type() << ", skip";
      return false;
    }
  }

  // add_out feeds only clip; clip_out feeds only mul; mul_out feeds only
  // tail_mul. These are all intermediates that get deleted.
  const std::pair<const char*, const char*> chain[] = {
      {"add_out", "clip"},
      {"clip_out", "mul"},
      {"mul_out", "tail_mul"},
  };
  for (const auto& kv : chain) {
    auto* var_node = matched.at(kv.first);
    for (auto* consumer : var_node->outlinks) {
      if (!consumer->IsStmt()) continue;
      if (consumer != matched.at(kv.second)) {
        LOG(WARNING) << "conv_hardswish_fuse: " << kv.first << " var "
                     << var_node->arg()->name << " has extra consumer "
                     << consumer->stmt()->op_type() << ", skip";
        return false;
      }
    }
  }
  return true;
}

cpp::OpDesc ConvHardSwishFuser::GenOpDesc(const key2nodes_t& matched) {
  cpp::OpDesc op_desc = *matched.at("conv2d")->stmt()->op_info();
  op_desc.SetOutput("Output", {matched.at("output")->arg()->name});

  // Set hard_swish activation attributes using standard naming
  op_desc.SetAttr("with_act", true);
  op_desc.SetAttr("act_type", std::string("hard_swish"));

  // Extract hard_swish parameters. HardSwish(x) = x * clip(x + offset, 0,
  // threshold) / scale, so the offset is the *bias constant of the
  // elementwise_add*, not derived from the clip range (offset = -clip.min is
  // only correct when the add bias happens to equal -min, which is not the
  // canonical pattern). Read the add_y constant like the layer_norm fuser:
  // prefer the fill_constant producer's attr, fall back to the tensor.
  float offset = 3.0f;
  float scale = 6.0f;
  float threshold = 6.0f;

  auto add_bias_op = matched.at("add")->stmt()->op();
  auto* scope = add_bias_op->scope();
  auto* add_y_node = matched.at("add_y");
  bool have_offset = false;
  for (auto* producer : add_y_node->inlinks) {
    if (producer->IsStmt() &&
        producer->stmt()->op_info()->Type() == "fill_constant") {
      auto* op_info = producer->stmt()->op_info();
      if (op_info->HasAttr("value")) {
        auto t = op_info->GetAttrType("value");
        if (t == paddle::lite::OpDescAPI::AttrType::FLOAT) {
          offset = op_info->GetAttr<float>("value");
          have_offset = true;
        } else if (t == paddle::lite::OpDescAPI::AttrType::INT) {
          offset = static_cast<float>(op_info->GetAttr<int>("value"));
          have_offset = true;
        }
      }
    }
  }
  if (!have_offset) {
    auto* add_y_t = scope->FindMutableTensor(add_y_node->arg()->name);
    if (add_y_t != nullptr && add_y_t->numel() == 1 &&
        add_y_t->data<float>() != nullptr) {
      offset = add_y_t->data<float>()[0];
      have_offset = true;
    }
  }
  // If the offset is not recoverable, keep the default 3.0 (the canonical
  // hard_swish offset); a future ValidateMatch tightening can reject instead.

  auto* clip_op_info = matched.at("clip")->stmt()->op_info();
  if (clip_op_info && clip_op_info->HasAttr("max")) {
    threshold = clip_op_info->GetAttr<float>("max");
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
