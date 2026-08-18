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

#include "lite/core/optimizer/mir/elimination/identity_transpose_eliminate_pass.h"
#include <map>
#include <set>
#include <string>
#include <vector>
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/optimizer/mir/pattern_matcher.h"
#include "lite/core/optimizer/mir/ssa_graph.h"
#include "lite/model_parser/cpp_desc.h"

namespace paddle {
namespace lite {
namespace mir {

namespace {

// Two permutations compose to the identity iff axis1[axis0[i]] == i for every
// index i. Both permutations act on the same rank.
bool AxesComposeToIdentity(const std::vector<int>& axis0,
                           const std::vector<int>& axis1) {
  if (axis0.size() != axis1.size()) return false;
  for (size_t i = 0; i < axis0.size(); ++i) {
    int a0 = axis0[i];
    if (a0 < 0 || static_cast<size_t>(a0) >= axis1.size()) return false;
    if (axis1[a0] != static_cast<int>(i)) return false;
  }
  return true;
}

}  // namespace

void IdentityTransposeEliminatePass::Apply(
    const std::unique_ptr<SSAGraph>& graph) {
  std::set<const Node*> to_remove;

  // Count stmt producers per arg var. A var written by several ops (an
  // x2paddle register-pool var, e.g. batch_norm_0.tmp_3 read by 30 ops) is
  // NOT a candidate even if its SSAGraph outlinks are small after copy
  // propagation: its other consumers logically depend on the mid transpose.
  std::map<std::string, int> var_producer_count;
  for (auto& node : graph->mutable_nodes()) {
    if (!node.IsArg()) continue;
    int producers = 0;
    for (auto* inlink : node.inlinks) {
      if (inlink->IsStmt()) {
        ++producers;
      }
    }
    var_producer_count[node.arg()->name] = producers;
  }

  for (auto* t0_node : graph->StmtTopologicalOrder()) {
    if (!t0_node->IsStmt() || t0_node->stmt()->op_type() != "transpose2") {
      continue;
    }
    auto* t0_op_info = t0_node->stmt()->op_info();
    if (!t0_op_info->HasAttr("axis")) continue;
    auto axis0 = t0_op_info->GetAttr<std::vector<int>>("axis");

    // t0's Out var must be consumed by exactly one stmt — a single
    // transpose2 — and must be produced by exactly one stmt. A var written
    // by several ops (register pool) is never matched: rewiring would change
    // what its other consumers observe.
    const auto out_names = t0_op_info->Output("Out");
    if (out_names.size() != 1) continue;
    const std::string& mid_name = out_names.front();
    auto it = var_producer_count.find(mid_name);
    if (it == var_producer_count.end() || it->second != 1) {
      VLOG(4) << "identity_transpose_eliminate: skip, mid " << mid_name
              << " producers="
              << (it == var_producer_count.end() ? -1 : it->second);
      continue;
    }
    auto* mid_arg = graph->RetrieveArgument(mid_name);
    if (mid_arg == nullptr) continue;

    // A register-pool var (x2paddle) is referenced by many ops in their
    // op_info inputs even after copy propagation rewires the SSAGraph edges.
    // Count logical consumers across every stmt's op_info: only a mid var
    // referenced by exactly one op is safe to eliminate.
    int logical_consumers = 0;
    for (auto& node : graph->mutable_nodes()) {
      if (!node.IsStmt()) continue;
      auto* oi = node.stmt()->op_info();
      bool references_mid = false;
      for (const auto& argname : oi->input_argnames()) {
        for (const auto& v : oi->Input(argname)) {
          if (v == mid_name) {
            references_mid = true;
            break;
          }
        }
        if (references_mid) break;
      }
      if (references_mid) ++logical_consumers;
    }
    if (logical_consumers != 1) {
      VLOG(4) << "identity_transpose_eliminate: skip, mid " << mid_name
              << " logical consumers=" << logical_consumers;
      continue;
    }

    // A register-pool var is written by many ops under the same name (e.g.
    // batch_norm_0.tmp_3 written by op 81 and read by 30 ops). Even after
    // copy propagation rewires consumers, the var stays as an Output of
    // several stmts. If any stmt other than t0 writes mid, mid is shared and
    // must not be eliminated.
    int other_writers = 0;
    for (auto& node : graph->mutable_nodes()) {
      if (!node.IsStmt() || &node == t0_node) continue;
      auto* oi = node.stmt()->op_info();
      for (const auto& argname : oi->output_argnames()) {
        for (const auto& v : oi->Output(argname)) {
          if (v == mid_name) {
            ++other_writers;
            break;
          }
        }
        if (other_writers > 0) break;
      }
      if (other_writers > 0) break;
    }
    if (other_writers > 0) {
      VLOG(4) << "identity_transpose_eliminate: skip, mid " << mid_name
              << " is written by other stmts";
      continue;
    }

    std::vector<Node*> mid_consumers;
    for (auto* consumer : mid_arg->outlinks) {
      if (consumer->IsStmt()) {
        mid_consumers.push_back(consumer);
      }
    }
    if (mid_consumers.size() != 1) continue;
    auto* t1_node = mid_consumers.front();
    if (t1_node->stmt()->op_type() != "transpose2") continue;
    auto* t1_op_info = t1_node->stmt()->op_info();
    if (!t1_op_info->HasAttr("axis")) continue;
    auto axis1 = t1_op_info->GetAttr<std::vector<int>>("axis");

    if (!AxesComposeToIdentity(axis0, axis1)) {
      continue;  // not a no-op pair, keep both
    }

    // t1's Out becomes t0's Out; t1 (and its XShape output var) are removed.
    const auto t1_out_names = t1_op_info->Output("Out");
    if (t1_out_names.size() != 1) continue;
    const std::string& out_name = t1_out_names.front();

    // Only eliminate a pair whose t1 output is a chain tail: either it has
    // no stmt consumers (model output / end of graph) or its single consumer
    // is not another transpose2. Eliminating a mid-chain pair in a longer
    // transpose run (t0 -> t1 -> t2 -> ...) rewires a var that a later
    // transpose still feeds, which is not semantics-preserving without also
    // folding the whole run.
    auto* out_arg = graph->RetrieveArgument(out_name);
    if (out_arg == nullptr) continue;
    std::vector<Node*> out_consumers;
    for (auto* consumer : out_arg->outlinks) {
      if (consumer->IsStmt()) {
        out_consumers.push_back(consumer);
      }
    }
    if (out_consumers.size() > 1) continue;
    if (out_consumers.size() == 1 &&
        out_consumers.front()->stmt()->op_type() == "transpose2") {
      VLOG(4) << "identity_transpose_eliminate: skip, t1 output feeds "
                 "another transpose (chain, not tail)";
      continue;
    }

    auto new_op_info = *t0_op_info;
    new_op_info.UpdateAllOutputs(mid_name, out_name);
    t0_node->stmt()->ResetOp(new_op_info, graph->valid_places());

    // Re-link t0 -> final out, dropping t0 -> mid.
    if (out_arg != nullptr) {
      RemoveDirectedLink(t0_node, mid_arg);
      DirectedLink(t0_node, out_arg);
    }

    to_remove.insert(t1_node);
    // Also remove t1's XShape output var if it exists and is now orphaned.
    if (t1_op_info->HasOutput("XShape")) {
      for (const auto& xs : t1_op_info->Output("XShape")) {
        auto* xs_arg = graph->RetrieveArgument(xs);
        if (xs_arg != nullptr) {
          to_remove.insert(xs_arg);
        }
      }
    }
  }

  if (!to_remove.empty()) {
    GraphSafeRemoveNodes(graph.get(), to_remove);
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(identity_transpose_eliminate_pass,
                  paddle::lite::mir::IdentityTransposeEliminatePass)
    .BindTargets({TARGET(kAny)});
