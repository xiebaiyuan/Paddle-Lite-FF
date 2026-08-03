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
#include "lite/core/optimizer/mir/pass_v2.h"
#include "lite/core/optimizer/mir/ssa_graph.h"

namespace paddle {
namespace lite {
namespace mir {

/*
 * Reshape2CastEliminatePass — eliminate x2paddle dynamic-shape residue.
 *
 * x2paddle-exported models (PaddleOCR rec) contain reshape2/cast chains that
 * shuffle shape tensors at runtime for dynamic batch. One deterministically
 * redundant pattern appears repeatedly:
 *
 *   reshape2(shape=[4]) identity — a shape tensor already has 4 elements,
 *   so `reshape2(shape=[4])` is a no-op; drop it, feeding the shape tensor
 *   directly to the consumer (usually the next cast/reshape2 in the chain).
 *
 * The pattern is pure shape-value manipulation; the rewritten graph computes
 * the same runtime shape (verified bit-identical on inference).
 */
class Reshape2CastEliminatePass : public PassV2 {
 public:
  Reshape2CastEliminatePass() : PassV2(Pass::Kind::kProgramWise) {}
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;
  bool ShouldOnlyApplyOnce() const override { return true; }
  int OptimizationLevel() const override { return PassV2::kLevelBasic; }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
