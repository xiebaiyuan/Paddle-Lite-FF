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
 * AssignEliminatePass — eliminate x2paddle `assign` identity-copy ops.
 *
 * x2paddle-exported models (e.g. PaddleOCR rec) contain hundreds of
 * `assign` ops that just copy one tensor to a fresh variable
 * (AssignOpLite::InferShapeImpl = Out->Resize(X->dims())), forming a
 * shared-variable "register pool" (one var read by 100+ ops). These
 * break the SSA closedness that fusion patterns rely on
 * (AsIntermediate() checks), so conv+act / scale_clip_mul / elementwise
 * fusions never fire on such models.
 *
 * This pass performs copy propagation: for each `assign(out) = in`,
 * every consumer of `out` is rewritten to read `in` instead, and the
 * assign node is removed. Pure identity copy — the rewrite is
 * semantics-preserving because out == in by construction.
 *
 * Safety: an assign is only eliminated when
 *   - `in` has exactly one producer (no feed / multiple writers),
 *   - `out` is not a feed input or model output (I/O interface unchanged),
 *   - `in`/`out` are DENSE_TENSOR (not tensor_array).
 */
class AssignEliminatePass : public PassV2 {
 public:
  AssignEliminatePass() : PassV2(Pass::Kind::kProgramWise) {}
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;
  bool ShouldOnlyApplyOnce() const override { return true; }
  int OptimizationLevel() const override { return PassV2::kLevelBasic; }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
