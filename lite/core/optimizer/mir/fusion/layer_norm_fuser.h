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

#pragma once

#include <memory>
#include <string>
#include "lite/core/optimizer/mir/pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

/// Folds the elementwise expansion of a LayerNorm back into a native
/// `layer_norm` op so the specialized ARM kernel (matrix_norm_row) is used.
///
/// x2paddle / PIR exports LayerNorm as a ~10-op elementwise chain:
///
///   x ── reduce_mean ──────────────→ mean
///   x ── sub(mean) ────────────────→ centered = x - mean
///   centered ── pow(2.0) ──────────→ centered²
///   centered² ── reduce_mean ──────→ var
///   var ── add(eps) ───────────────→ var + eps
///   var+eps ── sqrt ───────────────→ std
///   centered ── div(std) ──────────→ normalized
///   normalized ── mul(scale) ──────→ scaled
///   scaled ── add(bias) ───────────→ output
///
/// After fusion: x → layer_norm(scale, bias, begin_norm_axis=2, epsilon)
class LayerNormFuser : public FuseBase {
 public:
  LayerNormFuser() = default;

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
