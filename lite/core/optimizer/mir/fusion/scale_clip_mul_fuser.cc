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

#include "lite/core/optimizer/mir/fusion/scale_clip_mul_fuser.h"
#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void ScaleClipMulFuser::BuildPattern() {
  // Pattern: x → scale(slope, 0.5) → clip(0, 1) → mul(y) → out
  //
  // clip(scale*x + bias, 0, 1) == hard_sigmoid(x; slope=scale, offset=bias),
  // so the scale+clip pair collapses into a single `hard_sigmoid` node.
  // The mul gate (x * hard_sigmoid(gate)) is preserved.

  auto* scale_in = VarNode("scale_in")
                       ->assert_is_op_input("scale", "X")
                       ->AsInput();
  auto* scale_op = OpNode("scale", "scale")->AsIntermediate();

  // clip must be (0, 1): the canonical hard_sigmoid clamp range.
  auto* clip_op = OpNode("clip", "clip")
                      ->assert_op_attr_satisfied<float>("min", [](float v) {
                        return std::fabs(v - 0.0f) < 1e-4f;
                      })
                      ->assert_op_attr_satisfied<float>("max", [](float v) {
                        return std::fabs(v - 1.0f) < 1e-4f;
                      })
                      ->AsIntermediate();

  auto* scale_out = VarNode("scale_out")
                        ->assert_is_op_output("scale", "Out")
                        ->assert_is_op_input("clip", "X")
                        ->AsIntermediate();
  auto* clip_out = VarNode("clip_out")
                       ->assert_is_op_output("clip", "Out")
                       ->assert_is_op_input("elementwise_mul", "Y");

  // The main branch feeding the mul gate.
  auto* x_in = VarNode("x_in")
                   ->assert_is_op_input("elementwise_mul", "X")
                   ->AsInput();
  // The mul gate (x * hard_sigmoid(gate)) is preserved after fusion, so it
  // must NOT be an intermediate node (DeleteInterNodes would remove it).
  auto* mul_op = OpNode("mul", "elementwise_mul");
  auto* output = VarNode("output")
                     ->assert_is_op_output("elementwise_mul", "Out")
                     ->AsOutput();

  // topology
  *scale_in >> *scale_op >> *scale_out >> *clip_op >> *clip_out >> *mul_op >>
      *output;
  *x_in >> *mul_op;
}

bool ScaleClipMulFuser::ValidateMatch(SSAGraph* graph,
                                      const key2nodes_t& matched) {
  // Validate the scale parameters before fusing. hard_sigmoid is
  //   clip(slope * x + offset, 0, 1)
  // so the scale bias must be 0.5 and the scale factor (the slope) must be
  // non-zero. Fusing without these checks would silently change numerics for
  // any scale→clip subgraph that merely looks like hard_sigmoid.
  auto* scale_op_info = matched.at("scale")->stmt()->op_info();
  float slope = scale_op_info->GetAttr<float>("scale");
  float bias = scale_op_info->GetAttr<float>("bias");
  // scale(x) = x*slope + bias when bias_after_scale=true, but
  //          = (x + bias)*slope when bias_after_scale=false. hard_sigmoid is
  // clip(slope*x + offset, 0, 1), i.e. always "bias after scale". With
  // bias_after_scale=false the effective offset is bias*slope, so the 0.5
  // check must apply to bias (for slope==1) or to bias*slope otherwise. The
  // cleanest rule: require bias_after_scale=true (the canonical PaddlePaddle
  // hard_sigmoid export) so offset == bias directly.
  bool bias_after_scale = true;
  if (scale_op_info->HasAttr("bias_after_scale")) {
    bias_after_scale = scale_op_info->GetAttr<bool>("bias_after_scale");
  }
  if (!bias_after_scale) {
    // With bias-after-scale disabled the effective offset is bias*slope; the
    // fusion is only exact when slope==1 (offset == bias == 0.5). Rather than
    // special-casing, reject the non-canonical form.
    if (std::fabs(slope - 1.0f) > 1e-4f) {
      LOG(WARNING) << "scale_clip_mul_fuse: bias_after_scale=false with "
                      "slope != 1 changes the offset, skip";
      return false;
    }
  }
  if (std::fabs(bias - 0.5f) > 1e-4f) {
    LOG(WARNING) << "scale_clip_mul_fuse: unexpected bias " << bias
                 << " (expected 0.5), skip";
    return false;
  }
  if (std::fabs(slope) < 1e-6f) {
    LOG(WARNING) << "scale_clip_mul_fuse: zero slope, skip";
    return false;
  }

  // The clip output must be consumed exclusively by this mul: after folding,
  // other consumers of the clip output would read a hard_sigmoid result where
  // they expected a clipped value.
  auto* clip_out_node = matched.at("clip_out");
  for (auto* consumer : clip_out_node->outlinks) {
    if (consumer != matched.at("mul")) {
      LOG(WARNING) << "scale_clip_mul_fuse: clip output "
                   << clip_out_node->arg()->name
                   << " is shared by multiple ops, skip";
      return false;
    }
  }
  return true;
}

void ScaleClipMulFuser::InsertNewNode(SSAGraph* graph,
                                      const key2nodes_t& matched) {
  auto scale_old = matched.at("scale")->stmt()->op();
  auto* scope = scale_old->scope();

  auto op_desc = GenOpDesc(matched);
  auto hs_op = LiteOpRegistry::Global().Create("hard_sigmoid");
  auto& valid_places = scale_old->valid_places();
  hs_op->Attach(op_desc, scope);

  auto* new_op_node = graph->GraphCreateInstructNode(hs_op, valid_places);

  // hard_sigmoid replaces scale+clip on the *gate* branch only. It writes
  // its output to the clip_out variable, which is the mul's Y operand, so
  // the mul (x * hard_sigmoid(gate)) keeps its original input/output edges.
  IR_NODE_LINK_TO(matched.at("scale_in"), new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("clip_out"));
}

cpp::OpDesc ScaleClipMulFuser::GenOpDesc(const key2nodes_t& matched) {
  auto* scale_op_info = matched.at("scale")->stmt()->op_info();
  float slope = scale_op_info->GetAttr<float>("scale");
  float bias = scale_op_info->GetAttr<float>("bias");
  bool bias_after_scale = true;
  if (scale_op_info->HasAttr("bias_after_scale")) {
    bias_after_scale = scale_op_info->GetAttr<bool>("bias_after_scale");
  }
  // scale(x) = x*slope + bias_eff, where bias_eff = bias (bias after scale)
  // or bias*slope (bias before scale). hard_sigmoid always applies the offset
  // after scaling, so express the same affine transform in that form.
  float offset = bias_after_scale ? bias : bias * slope;

  cpp::OpDesc op_desc;
  op_desc.SetType("hard_sigmoid");
  op_desc.SetInput("X", {matched.at("scale_in")->arg()->name});
  op_desc.SetOutput("Out", {matched.at("clip_out")->arg()->name});
  op_desc.SetAttr("slope", slope);
  op_desc.SetAttr("offset", offset);
  return op_desc;
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
