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

#include "lite/core/optimizer/mir/fusion/div_mul_fuser.h"
#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void DivMulFuser::BuildPattern() {
  // Pattern: div(constant) → mul(constant)
  //
  // Optimizes: x / c_div * c_mul
  //         → x * (c_mul / c_div)

  auto* input = VarNode("input")
                    ->assert_is_op_input("elementwise_div", "X")
                    ->AsInput();
  auto* div_y = VarNode("div_y")
                    ->assert_is_op_input("elementwise_div", "Y")
                    ->assert_is_persistable_var()
                    ->AsInput();

  auto* div_op = OpNode("div", "elementwise_div")->AsIntermediate();
  auto* div_out = VarNode("div_out")
                      ->assert_is_op_output("elementwise_div", "Out")
                      ->assert_is_op_input("elementwise_mul", "X")
                      ->AsIntermediate();

  auto* mul_y = VarNode("mul_y")
                    ->assert_is_op_input("elementwise_mul", "Y")
                    ->assert_is_persistable_var()
                    ->AsInput();
  auto* mul_op = OpNode("mul", "elementwise_mul")->AsIntermediate();
  auto* output = VarNode("output")
                     ->assert_is_op_output("elementwise_mul", "Out")
                     ->AsOutput();

  // topology
  *input >> *div_op >> *div_out >> *mul_op >> *output;
  *div_y >> *div_op;
  *mul_y >> *mul_op;
}

bool DivMulFuser::ValidateMatch(SSAGraph* graph, const key2nodes_t& matched) {
  // Safety: we mutate the mul_y tensor in place. Only fold when mul_y is
  // consumed exclusively by this mul; otherwise another op sharing the same
  // constant would read the overwritten value.
  auto* mul_y_node = matched.at("mul_y");
  for (auto* consumer : mul_y_node->outlinks) {
    if (consumer != matched.at("mul")) {
      LOG(WARNING) << "div_mul_fuse: constant " << mul_y_node->arg()->name
                   << " is shared by multiple ops, skip";
      return false;
    }
  }

  auto mul_old = matched.at("mul")->stmt()->op();
  auto* scope = mul_old->scope();
  auto* div_y_t = scope->FindMutableTensor(matched.at("div_y")->arg()->name);
  auto* mul_y_t = scope->FindMutableTensor(matched.at("mul_y")->arg()->name);
  if (div_y_t == nullptr || mul_y_t == nullptr) {
    LOG(WARNING) << "div_mul_fuse: cannot find constant tensors, skip";
    return false;
  }

  auto div_dims = div_y_t->dims();
  auto mul_dims = mul_y_t->dims();
  // Both constants must be broadcast-compatible scalars or vectors of
  // identical shape for this fold to be valid.
  if (div_dims.production() != mul_dims.production()) {
    LOG(WARNING) << "div_mul_fuse: constant shapes mismatch, skip";
    return false;
  }

  const float* div_data = div_y_t->data<float>();
  const float* mul_data = mul_y_t->data<float>();
  auto numel = div_y_t->numel();
  for (int64_t i = 0; i < numel; ++i) {
    if (div_data[i] == 0.0f) {
      LOG(WARNING) << "div_mul_fuse: division by zero constant, skip";
      return false;
    }
  }
  return true;
}

void DivMulFuser::InsertNewNode(SSAGraph* graph,
                                 const key2nodes_t& matched) {
  // Fold constants: out = (x / c_div) * c_mul = x * (c_mul / c_div)
  // Read both constant tensors from scope and write the combined scale
  // back into the mul's Y tensor.
  auto mul_old = matched.at("mul")->stmt()->op();
  auto* scope = mul_old->scope();

  auto* div_y_t = scope->FindMutableTensor(matched.at("div_y")->arg()->name);
  auto* mul_y_t = scope->FindMutableTensor(matched.at("mul_y")->arg()->name);
  auto numel = div_y_t->numel();

  const float* div_data = div_y_t->data<float>();
  const float* mul_data = mul_y_t->data<float>();
  float* combined_data = mul_y_t->mutable_data<float>();
  for (int64_t i = 0; i < numel; ++i) {
    combined_data[i] = mul_data[i] / div_data[i];
  }

  // Rewire the graph: div is eliminated, input feeds directly into mul.
  auto op_desc = GenOpDesc(matched);
  auto mul_op = LiteOpRegistry::Global().Create("elementwise_mul");
  auto& valid_places = mul_old->valid_places();
  mul_op->Attach(op_desc, scope);

  auto* new_op_node = graph->GraphCreateInstructNode(mul_op, valid_places);

  IR_NODE_LINK_TO(matched.at("input"), new_op_node);
  IR_NODE_LINK_TO(matched.at("mul_y"), new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("output"));
}

cpp::OpDesc DivMulFuser::GenOpDesc(const key2nodes_t& matched) {
  cpp::OpDesc op_desc = *matched.at("mul")->stmt()->op_info();
  op_desc.SetInput("X", {matched.at("input")->arg()->name});
  op_desc.SetOutput("Out", {matched.at("output")->arg()->name});
  return op_desc;
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
