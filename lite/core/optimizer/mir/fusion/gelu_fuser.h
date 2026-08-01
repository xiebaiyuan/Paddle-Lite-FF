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

/// Folds the erf-based GELU composite pattern into a single `gelu` operator.
///
/// Matches the standard Paddle 2.x GELU expansion:
///   x → div(x, 1/√2) → erf → add(1.0) → mul(x) → mul(0.5) → out
///
/// i.e. out = 0.5 * x * (1 + erf(x / √2))
///
/// which is exactly the exact (non-approximate) GELU. The whole chain is
/// replaced by a single native `gelu` operator (approximate=false), matching
/// ONNX Runtime's GeluFusion (Level2) transformer.
class GeluFuser : public FuseBase {
 public:
  GeluFuser() = default;

  void BuildPattern() override;
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override;

 private:
  bool ValidateMatch(SSAGraph* graph, const key2nodes_t& matched) override;
  cpp::OpDesc GenOpDesc(const key2nodes_t& matched) override;
};

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
