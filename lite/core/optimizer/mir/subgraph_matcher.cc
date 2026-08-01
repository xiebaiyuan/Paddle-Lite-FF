// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

#include "lite/core/optimizer/mir/subgraph_matcher.h"
#include <set>

namespace paddle {
namespace lite {
namespace mir {

// ============================================================
// Activation-type helpers
// ============================================================

bool IsFusableActivationOp(const std::string& op_type) {
  static const std::set<std::string> kFusable = {
      "relu",          "relu6",       "leaky_relu",
      "hard_swish",    "hard_sigmoid", "prelu",
      "sigmoid",       "tanh",         "swish",
      // "gelu" removed — gelu fusion temporarily disabled (see
      // conv_activation_fuse_pass.cc). The attribute plumbing below stays
      // so re-enabling is a one-line change.
      "abs" };
  return kFusable.count(op_type) > 0;
}

ActivationAttributes ExtractActivationAttributes(
    const std::string& act_type,
    const cpp::OpDesc* act_op_desc) {
  ActivationAttributes attrs;
  attrs.type = act_type;

  if (act_op_desc == nullptr) return attrs;

  if (act_type == "relu6" || act_type == "hard_swish") {
    attrs.threshold = act_op_desc->GetAttr<float>("threshold");
  }
  if (act_type == "leaky_relu") {
    attrs.alpha = act_op_desc->GetAttr<float>("alpha");
  }
  if (act_type == "swish" && act_op_desc->HasAttr("beta")) {
    attrs.beta = act_op_desc->GetAttr<float>("beta");
  }
  if (act_type == "gelu" && act_op_desc->HasAttr("approximate")) {
    attrs.approximate = act_op_desc->GetAttr<bool>("approximate");
  }
  if (act_type == "hard_swish" || act_type == "hard_sigmoid") {
    attrs.offset = act_op_desc->GetAttr<float>("offset");
    if (act_type == "hard_swish") {
      attrs.scale = act_op_desc->GetAttr<float>("scale");
    } else {
      if (act_op_desc->HasAttr("slope")) {
        attrs.scale = act_op_desc->GetAttr<float>("slope");
      }
    }
  }
  if (act_type == "prelu") {
    attrs.mode = act_op_desc->GetAttr<std::string>("mode");
  }
  return attrs;
}

void ApplyActivationAttributes(cpp::OpDesc* op_desc,
                               const std::string& act_type,
                               const cpp::OpDesc* act_op_desc) {
  if (act_type.empty()) return;

  // Generic activation type marker that many fused kernels check.
  op_desc->SetAttr("with_act", true);
  op_desc->SetAttr("act_type", act_type);

  auto attrs = ExtractActivationAttributes(act_type, act_op_desc);

  // Quantized activation threshold
  if (act_op_desc && act_op_desc->HasAttr("out_threshold")) {
    op_desc->SetAttr("out_threshold",
                     act_op_desc->GetAttr<float>("out_threshold"));
  }

  // Standard attribute names used by most ARM/OpenCL fused kernels
  if (act_type == "relu") {
    op_desc->SetAttr("fuse_relu", true);
  } else if (act_type == "relu6") {
    op_desc->SetAttr("fuse_brelu_threshold", attrs.threshold);
  } else if (act_type == "leaky_relu") {
    op_desc->SetAttr("leaky_relu_alpha", attrs.alpha);
  } else if (act_type == "hard_swish") {
    op_desc->SetAttr("hard_swish_threshold", attrs.threshold);
    op_desc->SetAttr("hard_swish_scale", attrs.scale);
    op_desc->SetAttr("hard_swish_offset", attrs.offset);
  } else if (act_type == "hard_sigmoid") {
    op_desc->SetAttr("slope", attrs.scale);
    op_desc->SetAttr("offset", attrs.offset);
  } else if (act_type == "prelu") {
    op_desc->SetAttr("prelu_mode", attrs.mode);
  } else if (act_type == "sigmoid") {
    op_desc->SetAttr("fuse_sigmoid", true);
  } else if (act_type == "tanh") {
    op_desc->SetAttr("fuse_tanh", true);
  } else if (act_type == "swish") {
    op_desc->SetAttr("swish_scale", attrs.beta);
    op_desc->SetAttr("fuse_swish", true);
  } else if (act_type == "abs") {
    op_desc->SetAttr("fuse_abs", true);
  } else if (act_type == "gelu") {
    op_desc->SetAttr("approximate", attrs.approximate);
  }
}

void ActivationAttributes::ApplyToOpDescScaleLike(cpp::OpDesc* op_desc) const {
  if (type.empty()) return;
  op_desc->SetAttr("activation_type", type);
  if (type == "relu") {
    op_desc->SetAttr("fuse_relu", true);
  } else if (type == "relu6") {
    op_desc->SetAttr("alpha", threshold);  // scale/instance_norm kernel reads "alpha"
  } else if (type == "leaky_relu") {
    op_desc->SetAttr("alpha", alpha);
  } else if (type == "hard_swish") {
    op_desc->SetAttr("threshold", threshold);
    op_desc->SetAttr("scale", scale);
    op_desc->SetAttr("offset", offset);
  } else if (type == "hard_sigmoid") {
    op_desc->SetAttr("slope", scale);
    op_desc->SetAttr("offset", offset);
  } else if (type == "prelu") {
    op_desc->SetAttr("mode", mode);
  } else if (type == "swish") {
    op_desc->SetAttr("beta", beta);
  } else if (type == "gelu") {
    op_desc->SetAttr("approximate", approximate);
  }
  // sigmoid / tanh / abs — scale/instance_norm kernels don't fuse these,
  // so no additional attributes needed beyond activation_type.
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
