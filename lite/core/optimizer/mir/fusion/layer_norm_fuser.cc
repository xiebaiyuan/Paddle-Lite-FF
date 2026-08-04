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

#include "lite/core/optimizer/mir/fusion/layer_norm_fuser.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void LayerNormFuser::BuildPattern() {
  // LayerNorm(x) = (x - mean) / sqrt(var + eps) * scale + bias
  //
  // x2paddle / PIR expansion:
  //   x → reduce_mean(x) → mean
  //   x → sub(x, mean) → centered
  //   centered → pow(2.0) → centered²
  //   centered² → reduce_mean → var
  //   var → add(eps) → var_eps
  //   var_eps → sqrt → std
  //   centered → div(std) → normalized
  //   normalized → mul(scale) → scaled
  //   scaled → add(bias) → output
  //
  // Fused: x → layer_norm(scale, bias, begin_norm_axis, epsilon) → output

  // Input x feeds both the mean branch (reduce_mean) and the center branch
  // (elementwise_sub).
  auto* input = VarNode("input")
                    ->assert_is_op_input("elementwise_sub", "X")
                    ->assert_is_op_input("reduce_mean", "X")
                    ->AsInput();

  // mean = reduce_mean(x)
  auto* mean_op = OpNode("mean_op", "reduce_mean")->AsIntermediate();
  auto* mean = VarNode("mean")
                   ->assert_is_op_output("reduce_mean", "Out")
                   ->assert_is_op_input("elementwise_sub", "Y")
                   ->AsIntermediate();

  // centered = sub(x, mean)
  auto* sub_op = OpNode("sub_op", "elementwise_sub")->AsIntermediate();
  auto* centered = VarNode("centered")
                       ->assert_is_op_output("elementwise_sub", "Out")
                       ->assert_is_op_input("elementwise_pow", "X")
                       ->assert_is_op_input("elementwise_div", "X")
                       ->AsIntermediate();

  // pow(centered, 2.0)
  auto* pow_y = VarNode("pow_y")
                    ->assert_is_op_input("elementwise_pow", "Y")
                    ->AsInput();
  auto* pow_op = OpNode("pow_op", "elementwise_pow")->AsIntermediate();
  auto* centered_sq = VarNode("centered_sq")
                          ->assert_is_op_output("elementwise_pow", "Out")
                          ->assert_is_op_input("reduce_mean", "X")
                          ->AsIntermediate();

  // var = reduce_mean(centered²)
  auto* var_op = OpNode("var_op", "reduce_mean")->AsIntermediate();
  auto* var = VarNode("var")
                  ->assert_is_op_output("reduce_mean", "Out")
                  ->assert_is_op_input("elementwise_add", "X")
                  ->AsIntermediate();

  // var_eps = add(var, eps)
  auto* eps = VarNode("eps")
                  ->assert_is_op_input("elementwise_add", "Y")
                  ->AsInput();
  auto* add_eps_op = OpNode("add_eps_op", "elementwise_add")->AsIntermediate();
  auto* var_eps = VarNode("var_eps")
                      ->assert_is_op_output("elementwise_add", "Out")
                      ->assert_is_op_input("sqrt", "X")
                      ->AsIntermediate();

  // std = sqrt(var + eps)
  auto* sqrt_op = OpNode("sqrt_op", "sqrt")->AsIntermediate();
  auto* std = VarNode("std")
                  ->assert_is_op_output("sqrt", "Out")
                  ->assert_is_op_input("elementwise_div", "Y")
                  ->AsIntermediate();

  // normalized = div(centered, std)
  auto* div_op = OpNode("div_op", "elementwise_div")->AsIntermediate();
  auto* normalized = VarNode("normalized")
                         ->assert_is_op_output("elementwise_div", "Out")
                         ->assert_is_op_input("elementwise_mul", "X")
                         ->AsIntermediate();

  // scaled = mul(normalized, scale)
  auto* scale = VarNode("scale")
                    ->assert_is_op_input("elementwise_mul", "Y")
                    ->assert_is_persistable_var()
                    ->AsInput();
  auto* mul_op = OpNode("mul_op", "elementwise_mul")->AsIntermediate();
  auto* scaled = VarNode("scaled")
                     ->assert_is_op_output("elementwise_mul", "Out")
                     ->assert_is_op_input("elementwise_add", "X")
                     ->AsIntermediate();

  // output = add(scaled, bias)
  auto* bias = VarNode("bias")
                   ->assert_is_op_input("elementwise_add", "Y")
                   ->assert_is_persistable_var()
                   ->AsInput();
  auto* add_bias_op = OpNode("add_bias_op", "elementwise_add")->AsIntermediate();
  auto* output = VarNode("output")
                     ->assert_is_op_output("elementwise_add", "Out")
                     ->AsOutput();

  // Topology: x forks into the mean branch and the center branch, and
  // `centered` is consumed by both pow and div.
  *input >> *mean_op >> *mean >> *sub_op;
  *input >> *sub_op;
  *sub_op >> *centered;
  *centered >> *pow_op >> *centered_sq >> *var_op >> *var >> *add_eps_op >>
      *var_eps >> *sqrt_op >> *std >> *div_op >> *normalized >> *mul_op >>
      *scaled >> *add_bias_op >> *output;
  *pow_y >> *pow_op;
  *eps >> *add_eps_op;
  *centered >> *div_op;
  *scale >> *mul_op;
  *bias >> *add_bias_op;
}

bool LayerNormFuser::ValidateMatch(SSAGraph* graph,
                                   const key2nodes_t& matched) {
  auto* pow_node = matched.at("pow_op");
  auto* scope = pow_node->stmt()->op()->scope();

  // pow exponent must be scalar 2.0. Read it from the fill_constant producer's
  // op attrs (the output tensor may not be materialized at opt time), falling
  // back to the tensor when available.
  auto check_scalar_constant = [&](const std::string& key, float expected,
                                   const char* what) -> bool {
    auto* var_node = matched.at(key);
    // Find the fill_constant producer of this var.
    for (auto* producer : var_node->inlinks) {
      if (producer->IsStmt() &&
          producer->stmt()->op_info()->Type() == "fill_constant") {
        auto* op_info = producer->stmt()->op_info();
        float v = 0.0f;
        bool have_v = false;
        if (op_info->HasAttr("value")) {
          auto t = op_info->GetAttrType("value");
          if (t == paddle::lite::OpDescAPI::AttrType::FLOAT) {
            v = op_info->GetAttr<float>("value");
            have_v = true;
          } else if (t == paddle::lite::OpDescAPI::AttrType::INT) {
            v = static_cast<float>(op_info->GetAttr<int>("value"));
            have_v = true;
          }
        }
        if (have_v && std::fabs(v - expected) < 1e-4f) return true;
        if (have_v) {
          LOG(WARNING) << "layer_norm_fuse: " << what << " value " << v
                       << " != " << expected << ", skip";
          return false;
        }
      }
    }
    // Fallback: read the (already materialized) tensor.
    auto* t = scope->FindMutableTensor(var_node->arg()->name);
    if (t == nullptr || t->numel() != 1) {
      LOG(WARNING) << "layer_norm_fuse: cannot read constant " << what
                   << ", skip";
      return false;
    }
    float v = t->data<float>()[0];
    if (std::fabs(v - expected) > 1e-4f) {
      LOG(WARNING) << "layer_norm_fuse: unexpected " << what << " value " << v
                   << " (expected " << expected << "), skip";
      return false;
    }
    return true;
  };
  if (!check_scalar_constant("pow_y", 2.0f, "pow exponent")) return false;

  // scale/bias must exist.
  auto* scale_t = scope->FindMutableTensor(matched.at("scale")->arg()->name);
  auto* bias_t = scope->FindMutableTensor(matched.at("bias")->arg()->name);
  if (scale_t == nullptr || bias_t == nullptr) {
    LOG(WARNING) << "layer_norm_fuse: missing scale/bias tensor, skip";
    return false;
  }

  // The fused op writes Mean/Variance as outputs. Both reduce_mean outputs
  // must be consumed only inside this pattern: if some other op reads the old
  // mean/var, collapsing would change what it observes.
  auto mean_consumers_ok = [&]() {
    auto* node = matched.at("mean");
    for (auto* consumer : node->outlinks) {
      if (consumer->IsStmt() && consumer != matched.at("sub_op")) {
        LOG(WARNING) << "layer_norm_fuse: mean var " << node->arg()->name
                     << " has extra consumer "
                     << consumer->stmt()->op_type() << ", skip";
        return false;
      }
    }
    return true;
  };
  auto var_consumers_ok = [&]() {
    auto* node = matched.at("var");
    for (auto* consumer : node->outlinks) {
      if (consumer->IsStmt() && consumer != matched.at("add_eps_op")) {
        LOG(WARNING) << "layer_norm_fuse: var " << node->arg()->name
                     << " has extra consumer "
                     << consumer->stmt()->op_type() << ", skip";
        return false;
      }
    }
    return true;
  };
  if (!mean_consumers_ok() || !var_consumers_ok()) return false;

  // Intermediate vars must be consumed only inside the pattern, otherwise
  // deleting them would orphan other consumers. `centered` is consumed by
  // exactly two ops (pow and div), both inside the pattern; everything else
  // must have exactly one consumer (the next op in the chain).
  auto consumers_ok = [&](const std::string& key,
                          const std::vector<const Node*>& allowed) {
    auto* node = matched.at(key);
    int stmt_consumers = 0;
    for (auto* consumer : node->outlinks) {
      if (!consumer->IsStmt()) continue;
      ++stmt_consumers;
      if (std::find(allowed.begin(), allowed.end(), consumer) ==
          allowed.end()) {
        LOG(WARNING) << "layer_norm_fuse: " << key << " var "
                     << node->arg()->name << " has unexpected consumer "
                     << consumer->stmt()->op_type() << ", skip";
        return false;
      }
    }
    return stmt_consumers == static_cast<int>(allowed.size());
  };
  if (!consumers_ok("centered_sq", {matched.at("var_op")})) return false;
  if (!consumers_ok("var_eps", {matched.at("sqrt_op")})) return false;
  if (!consumers_ok("std", {matched.at("div_op")})) return false;
  if (!consumers_ok("normalized", {matched.at("mul_op")})) return false;
  if (!consumers_ok("scaled", {matched.at("add_bias_op")})) return false;
  if (!consumers_ok("centered",
                    {matched.at("pow_op"), matched.at("div_op")})) {
    return false;
  }

  // both reduce_mean keep_dim must match (layer_norm needs axis semantics)
  auto* mean_op_info = matched.at("mean_op")->stmt()->op_info();
  auto* var_op_info = matched.at("var_op")->stmt()->op_info();
  bool mean_keep = mean_op_info->HasAttr("keep_dim")
                       ? mean_op_info->GetAttr<bool>("keep_dim")
                       : false;
  bool var_keep = var_op_info->HasAttr("keep_dim")
                      ? var_op_info->GetAttr<bool>("keep_dim")
                      : false;
  if (mean_keep != var_keep) {
    LOG(WARNING) << "layer_norm_fuse: keep_dim mismatch, skip";
    return false;
  }

  // begin_norm_axis is derived from the var reduce_mean's `dim` attribute:
  // a reduce over trailing dims [axis, axis+1, ..., rank-1] maps to
  // begin_norm_axis = axis (i.e. the first reduced axis). Any other `dim`
  // shape (non-contiguous, or not ending at the last axis) cannot be
  // expressed by the fused layer_norm op, so the match is rejected rather
  // than silently computing over the wrong axes.
  auto* var_in_info = matched.at("var_op")->stmt()->op_info();
  if (!var_in_info->HasAttr("dim")) {
    LOG(WARNING) << "layer_norm_fuse: reduce_mean lacks dim attr, skip";
    return false;
  }
  auto dim_attr_type = var_in_info->GetAttrType("dim");
  std::vector<int64_t> dims;
  if (dim_attr_type == paddle::lite::OpDescAPI::AttrType::INTS) {
    auto dims32 = var_in_info->GetAttr<std::vector<int32_t>>("dim");
    dims.assign(dims32.begin(), dims32.end());
  } else if (dim_attr_type == paddle::lite::OpDescAPI::AttrType::LONGS) {
    dims = var_in_info->GetAttr<std::vector<int64_t>>("dim");
  } else {
    LOG(WARNING) << "layer_norm_fuse: unexpected dim attr type, skip";
    return false;
  }
  if (dims.empty()) {
    LOG(WARNING) << "layer_norm_fuse: empty reduce dim, skip";
    return false;
  }
  // dims must be contiguous trailing axes [axis, axis+1, ..., rank-1].
  int64_t begin_norm_axis = dims[0];
  for (size_t i = 1; i < dims.size(); ++i) {
    if (dims[i] != begin_norm_axis + static_cast<int64_t>(i)) {
      LOG(WARNING) << "layer_norm_fuse: reduce dims not contiguous trailing "
                      "axes, skip";
      return false;
    }
  }
  // The mean reduce_mean must reduce the same axes; use its dim attr.
  if (mean_op_info->HasAttr("dim")) {
    auto mean_dim_type = mean_op_info->GetAttrType("dim");
    std::vector<int64_t> mean_dims;
    if (mean_dim_type == paddle::lite::OpDescAPI::AttrType::INTS) {
      auto md32 = mean_op_info->GetAttr<std::vector<int32_t>>("dim");
      mean_dims.assign(md32.begin(), md32.end());
    } else if (mean_dim_type == paddle::lite::OpDescAPI::AttrType::LONGS) {
      mean_dims = mean_op_info->GetAttr<std::vector<int64_t>>("dim");
    }
    if (mean_dims != dims) {
      LOG(WARNING) << "layer_norm_fuse: mean/var reduce dims differ, skip";
      return false;
    }
  }

  // scale/bias must be 1-D vectors (per-channel LN parameters).
  if (scale_t->dims().size() != 1 || bias_t->dims().size() != 1) {
    LOG(WARNING) << "layer_norm_fuse: scale/bias must be 1-D, skip";
    return false;
  }

  // Store epsilon from the eps constant (fill_constant attr or tensor).
  float eps = 0.0f;
  bool have_eps = false;
  auto* eps_node = matched.at("eps");
  for (auto* producer : eps_node->inlinks) {
    if (producer->IsStmt() &&
        producer->stmt()->op_info()->Type() == "fill_constant") {
      auto* op_info = producer->stmt()->op_info();
      if (op_info->HasAttr("value")) {
        auto t = op_info->GetAttrType("value");
        if (t == paddle::lite::OpDescAPI::AttrType::FLOAT) {
          eps = op_info->GetAttr<float>("value");
          have_eps = true;
        } else if (t == paddle::lite::OpDescAPI::AttrType::INT) {
          eps = static_cast<float>(op_info->GetAttr<int>("value"));
          have_eps = true;
        }
      }
    }
  }
  if (!have_eps) {
    auto* eps_t = scope->FindMutableTensor(eps_node->arg()->name);
    if (eps_t != nullptr && eps_t->numel() == 1) {
      eps = eps_t->data<float>()[0];
      have_eps = true;
    }
  }
  if (!have_eps) {
    LOG(WARNING) << "layer_norm_fuse: cannot read eps, skip";
    return false;
  }
  if (eps <= 0.0f) {
    LOG(WARNING) << "layer_norm_fuse: non-positive eps, skip";
    return false;
  }
  return true;
}

void LayerNormFuser::InsertNewNode(SSAGraph* graph,
                                   const key2nodes_t& matched) {
  auto add_bias_op = matched.at("add_bias_op")->stmt()->op();
  auto* scope = add_bias_op->scope();

  auto op_desc = GenOpDesc(matched);
  auto ln_op = LiteOpRegistry::Global().Create("layer_norm");
  auto& valid_places = add_bias_op->valid_places();
  ln_op->Attach(op_desc, scope);

  auto* new_op_node = graph->GraphCreateInstructNode(ln_op, valid_places);

  IR_NODE_LINK_TO(matched.at("input"), new_op_node);
  IR_NODE_LINK_TO(matched.at("scale"), new_op_node);
  IR_NODE_LINK_TO(matched.at("bias"), new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("output"));

  // The old mean/var arg nodes become the layer_norm's Mean/Variance outputs.
  // They were produced by the (now removed) reduce_mean ops; link them to the
  // new op so the kernel can write mean/var there.
  IR_OP_VAR_LINK(new_op_node, matched.at("mean"));
  IR_OP_VAR_LINK(new_op_node, matched.at("var"));
}

cpp::OpDesc LayerNormFuser::GenOpDesc(const key2nodes_t& matched) {
  // Re-derive per-match values directly from the matched subgraph instead of
  // relying on members populated by ValidateMatch: FuseBase::operator() runs
  // ValidateMatch for every match before any InsertNewNode, so a member would
  // hold the *last* validated match's values for all fused ops. Reading here
  // keeps every fused layer_norm independent.

  // epsilon = eps constant (fill_constant attr or tensor).
  auto add_bias_op = matched.at("add_bias_op")->stmt()->op();
  auto* scope = add_bias_op->scope();
  float eps = 1e-5f;
  bool have_eps = false;
  auto* eps_node = matched.at("eps");
  for (auto* producer : eps_node->inlinks) {
    if (producer->IsStmt() &&
        producer->stmt()->op_info()->Type() == "fill_constant") {
      auto* op_info = producer->stmt()->op_info();
      if (op_info->HasAttr("value")) {
        auto t = op_info->GetAttrType("value");
        if (t == paddle::lite::OpDescAPI::AttrType::FLOAT) {
          eps = op_info->GetAttr<float>("value");
          have_eps = true;
        } else if (t == paddle::lite::OpDescAPI::AttrType::INT) {
          eps = static_cast<float>(op_info->GetAttr<int>("value"));
          have_eps = true;
        }
      }
    }
  }
  if (!have_eps) {
    auto* eps_t = scope->FindMutableTensor(eps_node->arg()->name);
    if (eps_t != nullptr && eps_t->numel() == 1) {
      eps = eps_t->data<float>()[0];
      have_eps = true;
    }
  }
  if (!have_eps || eps <= 0.0f) {
    // ValidateMatch rejected this match; fall back to a sane default.
    eps = 1e-5f;
  }

  // begin_norm_axis = first reduced axis of the var reduce_mean.
  int64_t begin_norm_axis = 2;
  auto* var_in_info = matched.at("var_op")->stmt()->op_info();
  if (var_in_info->HasAttr("dim")) {
    auto dim_attr_type = var_in_info->GetAttrType("dim");
    if (dim_attr_type == paddle::lite::OpDescAPI::AttrType::INTS) {
      auto dims = var_in_info->GetAttr<std::vector<int32_t>>("dim");
      if (!dims.empty()) begin_norm_axis = dims[0];
    } else if (dim_attr_type == paddle::lite::OpDescAPI::AttrType::LONGS) {
      auto dims = var_in_info->GetAttr<std::vector<int64_t>>("dim");
      if (!dims.empty()) begin_norm_axis = dims[0];
    }
  }

  cpp::OpDesc op_desc;
  op_desc.SetType("layer_norm");
  op_desc.SetInput("X", {matched.at("input")->arg()->name});
  op_desc.SetInput("Scale", {matched.at("scale")->arg()->name});
  op_desc.SetInput("Bias", {matched.at("bias")->arg()->name});
  op_desc.SetOutput("Y", {matched.at("output")->arg()->name});
  op_desc.SetOutput("Mean", {matched.at("mean")->arg()->name});
  op_desc.SetOutput("Variance", {matched.at("var")->arg()->name});
  op_desc.SetAttr("begin_norm_axis", static_cast<int>(begin_norm_axis));
  op_desc.SetAttr("epsilon", eps);
  return op_desc;
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
