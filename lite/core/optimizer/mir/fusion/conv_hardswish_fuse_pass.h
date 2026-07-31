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
#include "lite/core/optimizer/mir/pass.h"

namespace paddle {
namespace lite {
namespace mir {

/// Fuses conv2d/depthwise_conv2d + hard_swish composite pattern into a single
/// fused conv operator with hard_swish activation.
///
/// Matches: conv → elementwise_add(offset) → clip → elementwise_mul →
/// elementwise_mul and replaces with a fused conv with hard_swish activation.
///
/// Note: currently NOT registered in the default optimizer pipeline — kept as
/// a candidate for models that export hard_swish as a composite chain.
class ConvHardSwishFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
