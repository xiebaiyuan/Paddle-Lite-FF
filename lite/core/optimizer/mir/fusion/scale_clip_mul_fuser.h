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

/// Folds the gate branch `scale(slope, offset) → clip(0, 1)` of an
/// elementwise_mul into a native `hard_sigmoid` operator.
///
/// Pattern (SE-style gating, as exported by Paddle 2.6 OCR models):
///   x → scale(slope, 0.5) → clip(0, 1) → mul(y, ...) → out
///
/// clip(scale * x + bias, 0, 1) is exactly the definition of hard_sigmoid
/// (slope = scale, offset = bias).  The two ops collapse into one native
/// `hard_sigmoid` node, leaving the mul gate untouched.
///
/// Safety: only fuses when the clip output is consumed exclusively by the
/// matched mul (so no other consumer observes a hard_sigmoid where a clip
/// was expected), the clip range is (0, 1), and the scale bias is 0.5.
class ScaleClipMulFuser : public FuseBase {
 public:
  explicit ScaleClipMulFuser() = default;

  void BuildPattern() override;
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override;

 private:
  cpp::OpDesc GenOpDesc(const key2nodes_t& matched) override;
};

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
