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
 * IdentityTransposeEliminatePass — collapse consecutive transpose2 ops whose
 * permutations compose to the identity.
 *
 *   x -> transpose2(axis0) -> mid -> transpose2(axis1) -> out
 *
 * becomes
 *
 *   x -> transpose2(axis0) -> out
 *
 * when axis1[axis0[i]] == i for every i (the two permutations cancel). The
 * second transpose is a pure no-op; removing it is semantics-preserving.
 *
 * Safety: a pair is only collapsed when
 *   - the mid var is consumed by exactly one stmt (a single transpose2),
 *   - both ops carry an "axis" attribute,
 *   - the two permutations compose to the identity,
 *   - the mid var is written by exactly one stmt (no register-pool sharing),
 *   - the second transpose's output is a chain tail (no following transpose).
 *
 * NOTE: this pass is NOT in the default opt pass chain. On x2paddle rec
 * models the mid var is a register-pool variable whose name is partially
 * rewritten by upstream passes, so "unique consumer" cannot be trusted there
 * and elimination would corrupt the graph (verified: tiny_rec output shape
 * changes). It is kept for clean graphs (e.g. hand-written or non-x2paddle
 * models) and for its unit tests; enable it explicitly only when the graph
 * is known to be free of shared-variable aliasing.
 */
class IdentityTransposeEliminatePass : public PassV2 {
 public:
  IdentityTransposeEliminatePass() : PassV2(Pass::Kind::kProgramWise) {}
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;
  bool ShouldOnlyApplyOnce() const override { return true; }
  int OptimizationLevel() const override { return PassV2::kLevelBasic; }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
