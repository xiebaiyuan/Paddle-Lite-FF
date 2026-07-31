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

#include <memory>
#include <string>
#include "lite/core/optimizer/mir/pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

/// Fuses conv2d + hard_swish composite pattern into a single fused conv.
///
/// HardSwish composite pattern (x2paddle export):
///   conv2d ─→ conv_out
///            ├→ add(offset) → clip(min,max) ─┐
///            └───────────────────────────────→ mul ─→ mul(1/scale) ─→ output
///
/// Corresponds to: hard_swish(x) = x * clip(x + offset, min, max) / scale
/// Default values: offset=3, min=0, max=6, scale=6
class ConvHardSwishFuser : public FuseBase {
 public:
  explicit ConvHardSwishFuser(const std::string& conv_type, bool has_bias)
      : conv_type_(conv_type), has_bias_(has_bias) {}

  void BuildPattern() override;
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override;

 private:
  cpp::OpDesc GenOpDesc(const key2nodes_t& matched) override;
  std::string conv_type_;
  bool has_bias_;
};

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
