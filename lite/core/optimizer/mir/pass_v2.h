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
#include "lite/core/optimizer/mir/pass.h"

namespace paddle {
namespace lite {
namespace mir {

// PassV2 extends the existing Pass interface with optional convergence-loop
// and level-aware capabilities. All existing passes are source-compatible —
// they inherit from Pass and can be used with PassManagerV2 without any
// modification. New passes can optionally override the new methods to declare
// whether they only need one application or which optimization level they
// belong to.
class PassV2 : public Pass {
 public:
  using Pass::Pass;  // Inherit constructors (Kind enum)

  // Optimization levels, mirroring ONNX Runtime's TransformerLevel.
  //
  // kLevelBasic     — safety guarantees. Eliminations, constant folding,
  //                   dead-node removal. Never changes model semantics.
  // kLevelExtended  — fusion and layout transformations that may change op
  //                   counts but preserve precision. Default for most passes.
  // kLevelAggressive — layout rewrites (NCHW→NHWC), quantization transforms,
  //                     precision conversions. May trade accuracy for speed.
  static constexpr int kLevelBasic = 1;
  static constexpr int kLevelExtended = 2;
  static constexpr int kLevelAggressive = 3;

  // Returns whether this pass modified the graph.
  //
  // The default returns true (conservative — assume modification) so that
  // the convergence loop keeps iterating. Passes that can cheaply determine
  // whether they changed anything should override this to return false when
  // no modifications were made, enabling early exit from the convergence loop.
  virtual bool ModifiedGraph() const { return true; }

  // Returns whether this pass should only be applied on the first step of
  // the convergence loop.
  //
  // Passes like constant folding and dead-node elimination only need one
  // application because their result is idempotent. Returning true here
  // skips them on subsequent steps, saving unnecessary graph traversals.
  virtual bool ShouldOnlyApplyOnce() const { return false; }

  // Optimization level this pass belongs to.
  //
  // PassManagerV2::Run() accepts a `max_level` parameter and will skip
  // passes whose OptimizationLevel() exceeds it. This enables progressive
  // optimization strategies (e.g. only basic passes during development, full
  // pipeline during production build).
  virtual int OptimizationLevel() const { return kLevelExtended; }

  // Apply the pass to the graph. Subclasses must set |modified| to true
  // iff the graph topology or node content was changed.
  //
  // The default implementation delegates to Apply() and queries
  // ModifiedGraph(). Subclasses that override this should set |modified|
  // directly based on what happened during this invocation, rather than
  // relying on the post-hoc ModifiedGraph() query.
  virtual void ApplyV2(const std::unique_ptr<SSAGraph>& graph,
                       bool& modified) {
    // Default fallback: call existing Apply(), and assume modification
    // (conservative).
    Apply(graph);
    modified = ModifiedGraph();
  }

  virtual ~PassV2() = default;
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
