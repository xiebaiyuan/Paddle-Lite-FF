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

#pragma once
#include <string>
#include "lite/core/optimizer/mir/pattern_matcher_high_api.h"
#include "lite/model_parser/cpp_desc.h"

namespace paddle {
namespace lite {
namespace mir {

// ============================================================
// Reusable activation-fusion helpers
// ============================================================
//
// The activation-type-to-attributes mapping (relu/relu6/leaky_relu/...)
// is duplicated across at least 5 fusion passes with different attribute
// naming conventions.  These helpers provide:
//
//   1. A canonical source of truth for "which op types are activations".
//   2. A struct that extracts all activation parameters from an op_desc.
//   3. A function that sets the extracted parameters with the caller's
//      choice of attribute names (because different kernels use different
//      naming conventions, e.g. "fuse_swish" vs "activation_type=swish").

// Returns true if |op_type| is a recognized activation operator.
bool IsFusableActivationOp(const std::string& op_type);

// Parsed activation parameters extracted from an activation OpDesc.
struct ActivationAttributes {
  std::string type;   // "relu", "relu6", "leaky_relu", ...
  float alpha    = 0.0f;  // threshold for relu6, alpha for leaky_relu, beta for swish
  float threshold = 0.0f;  // threshold for hard_swish / relu6
  float scale    = 1.0f;  // scale for hard_swish, slope for hard_sigmoid
  float offset   = 0.0f;  // offset for hard_swish / hard_sigmoid
  float beta     = 1.0f;  // swish beta
  std::string mode;       // prelu mode ("channel", "element", "all")

  bool empty() const { return type.empty(); }
};

// Extract activation parameters from an op's OpDesc.
//
// |act_type|  — the operator type string ("relu", "relu6", ...).
// |act_op_desc| — the activation operator's full descriptor (for attributes);
//                 may be nullptr if the caller already has the parameters.
ActivationAttributes ExtractActivationAttributes(
    const std::string& act_type,
    const cpp::OpDesc* act_op_desc = nullptr);

// Apply the activation attributes to |op_desc| using standard naming
// conventions.  This is the fallback that works for most fusion passes.
//
// Individual fusion passes that need non-standard attribute names should
// call ExtractActivationAttributes() and set attributes manually.
void ApplyActivationAttributes(cpp::OpDesc* op_desc,
                               const std::string& act_type,
                               const cpp::OpDesc* act_op_desc);

}  // namespace mir
}  // namespace lite
}  // namespace paddle
