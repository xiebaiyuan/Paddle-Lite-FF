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
#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "lite/core/optimizer/mir/pass.h"
#include "lite/core/optimizer/mir/pass_manager.h"
#include "lite/core/optimizer/mir/pass_v2.h"

namespace paddle {
namespace lite {
namespace mir {

// PassManagerV2 extends PassManager with two key features:
//
// 1. CONVERGENCE LOOP — passes are re-run up to `steps_` times. After each
//    step, if no pass modified the graph, the loop terminates early. Passes
//    that set ShouldOnlyApplyOnce() are skipped after the first step.
//
// 2. PROGRESSIVE LEVELS — passes are grouped by OptimizationLevel (Basic /
//    Extended / Aggressive). `Run()` accepts a `max_level` to control
//    optimization aggressiveness per invocation.
//
// Usage (backward compatible):
//   PassManagerV2::Global().Run(graph);  // defaults: steps=3, max_level=3
//   PassManagerV2::Global().Run(graph, /*steps=*/1, /*max_level=*/1);
//
// All existing passes registered via PassManager will be transparently
// promoted to PassV2-compatible during Run(), so zero code changes are
// required in existing passes.
class PassManagerV2 : public PassManager {
 public:
  static PassManagerV2& Global() {
    static PassManagerV2 x;
    return x;
  }

  // Run all registered passes with convergence loop.
  //
  // |steps|     — maximum iterations of the convergence loop (default 3).
  // |max_level| — only run passes with OptimizationLevel() <= max_level
  //               (default 3 = all levels). Set to 1 for basic-only mode.
  void Run(const std::unique_ptr<SSAGraph>& graph,
           int steps = 3,
           int max_level = 3) {
    LOG(INFO) << "Running MIR passes (v2) with steps=" << steps
              << " max_level=" << max_level;

    for (int step = 0; step < steps; ++step) {
      bool graph_changed = false;

      for (auto it = passes_begin(); it != passes_end(); ++it) {
        Pass* pass = it->get();
        int level = GetPassLevel(pass);
        if (level > max_level) {
          VLOG(4) << "Skipping pass " << pass->name()
                  << " (level=" << level << " > max=" << max_level << ")";
          continue;
        }

        if (step > 0 && ShouldOnlyApplyOnceForPass(pass)) {
          VLOG(4) << "Skipping pass " << pass->name()
                  << " on step " << step << " (apply-once)";
          continue;
        }

        VLOG(4) << "Step " << step << ", running pass: " << pass->name();
        bool modified = false;
        ApplyPass(pass, graph, modified);
        if (modified) {
          graph_changed = true;
          VLOG(4) << "Pass " << pass->name() << " modified graph";
        }
      }

      if (!graph_changed) {
        LOG(INFO) << "Convergence reached after " << (step + 1) << " step(s)";
        break;
      }
    }
  }

  // Register a pass by name (delegates to PassManager).
  // The pass pointer is stored internally; PassManagerV2 will query its
  // PassV2 interface at runtime.
  bool AddNewPass(const std::string& name, Pass* pass) {
    // Try to promote Pass* to PassV2* for metadata queries.
    auto* pass_v2 = dynamic_cast<PassV2*>(pass);
    if (pass_v2 != nullptr) {
      promoted_passes_[name] = pass_v2;
    }
    return PassManager::AddNewPass(name, pass);
  }

  // Opt-in: force a specific pass to behave as apply-once.
  void SetApplyOnce(const std::string& name, bool flag = true) {
    apply_once_overrides_[name] = flag;
  }

  // Opt-in: force a specific pass level override.
  void SetLevel(const std::string& name, int level) {
    level_overrides_[name] = level;
  }

  // Clear all passes and metadata (overrides PassManager::Clear).
  void ClearAll() {
    PassManager::Clear();
    promoted_passes_.clear();
    apply_once_overrides_.clear();
    level_overrides_.clear();
  }

  // Allow temporary instances for testing and local use.
  PassManagerV2() = default;

  // Query a pass's effective level (metadata or override). Returns -1 if
  // the pass is not registered.
  int GetPassLevel(const std::string& name) const {
    auto it = level_overrides_.find(name);
    if (it != level_overrides_.end()) return it->second;
    auto pit = promoted_passes_.find(name);
    if (pit != promoted_passes_.end()) return pit->second->OptimizationLevel();
    return PassV2::kLevelExtended;
  }

 private:
  int GetPassLevel(Pass* pass) const { return GetPassLevel(pass->name()); }

  bool ShouldOnlyApplyOnceForPass(Pass* pass) const {
    auto name_it = apply_once_overrides_.find(pass->name());
    if (name_it != apply_once_overrides_.end()) return name_it->second;
    auto promoted_it = promoted_passes_.find(pass->name());
    if (promoted_it != promoted_passes_.end())
      return promoted_it->second->ShouldOnlyApplyOnce();
    return false;
  }

  void ApplyPass(Pass* pass,
                 const std::unique_ptr<SSAGraph>& graph,
                 bool& modified) {
    auto* pass_v2 = dynamic_cast<PassV2*>(pass);
    if (pass_v2 != nullptr) {
      pass_v2->ApplyV2(graph, modified);
      return;
    }
    // Legacy pass — always assume modification (conservative).
    pass->Apply(graph);
    modified = true;
  }

  std::map<std::string, PassV2*> promoted_passes_;
  std::map<std::string, bool> apply_once_overrides_;
  std::map<std::string, int> level_overrides_;
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
