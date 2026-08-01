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

#include "lite/core/optimizer/mir/fusion/gelu_fuser.h"
#include <cmath>
#include <memory>
#include <vector>
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void GeluFuser::BuildPattern() {
  // GELU(x) = 0.5 * x * (1 + erf(x / √2))
  //
  // Pattern (Paddle 2.x expansion):
  //   x → div(x, c1) → erf → add(1.0) → mul(x) → mul(0.5) → out
  //
  //   div: x / c1            (c1 = √2, or 1/√2 depending on export)
  //   erf: erf(div_out)
  //   add: erf_out + 1.0
  //   mul: x * add_out
  //   mul: mul_out * 0.5

  auto* input = VarNode("input")
                    ->assert_is_op_input("elementwise_div", "X")
                    ->assert_is_op_input("elementwise_mul", "X")
                    ->AsInput();

  // div(x, c1)
  auto* div_y = VarNode("div_y")
                    ->assert_is_op_input("elementwise_div", "Y")
                    ->assert_is_persistable_var()
                    ->AsInput();
  auto* div_op = OpNode("div", "elementwise_div")->AsIntermediate();
  auto* div_out = VarNode("div_out")
                      ->assert_is_op_output("elementwise_div", "Out")
                      ->assert_is_op_input("erf", "X")
                      ->AsIntermediate();

  // erf(div_out)
  auto* erf_op = OpNode("erf", "erf")->AsIntermediate();
  auto* erf_out = VarNode("erf_out")
                      ->assert_is_op_output("erf", "Out")
                      ->assert_is_op_input("elementwise_add", "X")
                      ->AsIntermediate();

  // add(erf_out + 1.0)
  auto* add_y = VarNode("add_y")
                    ->assert_is_op_input("elementwise_add", "Y")
                    ->assert_is_persistable_var()
                    ->AsInput();
  auto* add_op = OpNode("add", "elementwise_add")->AsIntermediate();
  auto* add_out = VarNode("add_out")
                      ->assert_is_op_output("elementwise_add", "Out")
                      ->assert_is_op_input("elementwise_mul", "Y")
                      ->AsIntermediate();

  // mul(x * add_out)
  auto* mul_op = OpNode("mul", "elementwise_mul")->AsIntermediate();
  auto* mul_out = VarNode("mul_out")
                      ->assert_is_op_output("elementwise_mul", "Out")
                      ->assert_is_op_input("elementwise_mul", "X")
                      ->AsIntermediate();

  // mul(mul_out * 0.5)
  auto* scale_y = VarNode("scale_y")
                      ->assert_is_op_input("elementwise_mul", "Y")
                      ->assert_is_persistable_var()
                      ->AsInput();
  auto* scale_op = OpNode("scale", "elementwise_mul")->AsIntermediate();
  auto* output = VarNode("output")
                     ->assert_is_op_output("elementwise_mul", "Out")
                     ->AsOutput();

  // topology
  *input >> *div_op >> *div_out >> *erf_op >> *erf_out >> *add_op >>
      *add_out >> *mul_op >> *mul_out >> *scale_op >> *output;
  *div_y >> *div_op;
  *add_y >> *add_op;
  *input >> *mul_op;
  *scale_y >> *scale_op;
}

bool GeluFuser::ValidateMatch(SSAGraph* graph, const key2nodes_t& matched) {
  // Validate the GELU constants before fusing. Exact GELU is
  //   0.5 * x * (1 + erf(x / √2))
  // so the div divisor must be √2 (or equivalently 1/√2 folded into the
  // multiply), the add bias must be 1.0, and the final scale must be 0.5.
  // Fusing without checking would silently produce wrong numerics for any
  // erf-based subgraph that merely *looks* like GELU.
  auto mul_old = matched.at("scale")->stmt()->op();
  auto* scope = mul_old->scope();

  const float kInvSqrt2 = 0.70710678118f;
  const float kSqrt2 = 1.41421356237f;
  const float kTolerance = 1e-4f;

  auto* div_y_t = scope->FindMutableTensor(matched.at("div_y")->arg()->name);
  auto* add_y_t = scope->FindMutableTensor(matched.at("add_y")->arg()->name);
  auto* scale_y_t = scope->FindMutableTensor(matched.at("scale_y")->arg()->name);
  if (div_y_t == nullptr || add_y_t == nullptr || scale_y_t == nullptr) {
    LOG(WARNING) << "gelu_fuse: cannot find constant tensors, skip";
    return false;
  }

  auto check_scalar = [kTolerance](const lite::Tensor* t, float expected,
                                   const char* what) {
    if (t->numel() != 1) {
      LOG(WARNING) << "gelu_fuse: " << what << " is not a scalar, skip";
      return false;
    }
    float v = t->data<float>()[0];
    if (std::fabs(v - expected) > kTolerance &&
        std::fabs(1.0f / v - expected) > kTolerance) {
      LOG(WARNING) << "gelu_fuse: unexpected " << what << " value " << v
                   << " (expected " << expected << "), skip";
      return false;
    }
    return true;
  };

  // div(x, c1): c1 must be √2 (x/√2). Accept both √2 and its reciprocal.
  if (!check_scalar(div_y_t, kSqrt2, "div divisor")) return false;
  // add: erf_out + 1.0
  if (!check_scalar(add_y_t, 1.0f, "add bias")) return false;
  // final mul scale: 0.5
  if (!check_scalar(scale_y_t, 0.5f, "final scale")) return false;
  return true;
}

void GeluFuser::InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) {
  auto mul_old = matched.at("scale")->stmt()->op();
  auto* scope = mul_old->scope();

  auto op_desc = GenOpDesc(matched);
  auto gelu_op = LiteOpRegistry::Global().Create("gelu");
  auto& valid_places = mul_old->valid_places();
  gelu_op->Attach(op_desc, scope);

  auto* new_op_node = graph->GraphCreateInstructNode(gelu_op, valid_places);

  IR_NODE_LINK_TO(matched.at("input"), new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("output"));
}

cpp::OpDesc GeluFuser::GenOpDesc(const key2nodes_t& matched) {
  cpp::OpDesc op_desc;
  op_desc.SetType("gelu");
  op_desc.SetInput("X", {matched.at("input")->arg()->name});
  op_desc.SetOutput("Out", {matched.at("output")->arg()->name});
  op_desc.SetAttr("approximate", false);
  return op_desc;
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
