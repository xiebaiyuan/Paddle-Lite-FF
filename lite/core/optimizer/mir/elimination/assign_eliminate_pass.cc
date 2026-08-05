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

#include "lite/core/optimizer/mir/elimination/assign_eliminate_pass.h"
#include <map>
#include <memory>
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

void AssignEliminatePass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  // Iterate to a fixpoint: assign chains (assign1.out consumed by
  // assign2.in) collapse progressively. Each pass over the graph handles
  // one level of the chain; chains in x2paddle models are ≤ 2 deep.
  for (int iter = 0; iter < 16; ++iter) {
    // Collect all assign stmt nodes.
    std::vector<Node*> assign_nodes;
    for (auto* node : graph->StmtTopologicalOrder()) {
      if (node->IsStmt() && node->stmt()->op_type() == "assign") {
        assign_nodes.push_back(node);
      }
    }
    if (assign_nodes.empty()) {
      return;
    }

    // For each arg (var) node, count its producers — the number of stmt
    // inlinks. Feed inputs / framework vars have 0 stmt producers; vars
    // written by exactly one op have 1. An assign's `in` must have exactly
    // one producer, otherwise rewriting consumers to read `in` could change
    // what they observe (multiple writers, or a framework-managed var).
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

    std::set<const Node*> to_remove;

    for (auto* assign_node : assign_nodes) {
      auto* op_info = assign_node->stmt()->op_info();
      const auto in_names = op_info->Input("X");
      const auto out_names = op_info->Output("Out");
      if (in_names.size() != 1 || out_names.size() != 1) {
        // Only the plain tensor form (1 X -> 1 Out).
        continue;
      }
      const std::string& in_name = in_names.front();
      const std::string& out_name = out_names.front();

      // `in` must have exactly one producer.
      auto it = var_producer_count.find(in_name);
      if (it == var_producer_count.end() || it->second != 1) {
        VLOG(4) << "assign_eliminate: skip, in " << in_name << " producers="
                << (it == var_producer_count.end() ? -1 : it->second);
        continue;
      }

      // Rewrite every stmt consumer of `out` to read `in` instead.
      auto* out_arg = graph->RetrieveArgument(out_name);
      auto* in_arg = graph->RetrieveArgument(in_name);
      if (out_arg == nullptr || in_arg == nullptr) {
        VLOG(4) << "assign_eliminate: skip, missing arg node for "
                << out_name << "/" << in_name;
        continue;
      }

      // `out` must not be a model output / framework var — rewriting its
      // consumers would change the inference I/O surface. A model output is
      // an arg node whose name matches a fetch target; SSAGraph exposes
      // these as arg nodes with no stmt consumers. If out has no consumers
      // at all it is either a dead var (safe to drop) or a fetch target
      // (must keep). Fetch targets are linked to the graph as outputs, so
      // check whether any consumer is a fetch/bookkeeping node.
      bool out_has_fetch_consumer = false;
      for (auto* consumer : out_arg->outlinks) {
        if (!consumer->IsStmt()) {
          out_has_fetch_consumer = true;
          break;
        }
      }
      if (out_has_fetch_consumer) {
        VLOG(4) << "assign_eliminate: skip, out " << out_name
                << " is a model output";
        continue;
      }

      std::vector<Node*> consumers;
      for (auto* consumer : out_arg->outlinks) {
        if (consumer->IsStmt()) {
          consumers.push_back(consumer);
        }
      }

      for (auto* consumer : consumers) {
        auto new_op_info = *consumer->stmt()->op_info();
        new_op_info.UpdateAllInputs(out_name, in_name);
        consumer->stmt()->ResetOp(new_op_info, graph->valid_places());
        // Drop the old edge (out_arg -> consumer) first; DirectedLink only
        // dedups the new edge, leaving a dangling link from the removed
        // assign's output arg which later trips kernel binding.
        RemoveDirectedLink(out_arg, consumer);
        DirectedLink(in_arg, consumer);
      }

      // Remove the assign node AND its output arg (now orphaned — the
      // assign was its only producer and all consumers were redirected).
      to_remove.insert(assign_node);
      to_remove.insert(out_arg);
      VLOG(3) << "assign_eliminate: " << out_name << " <- " << in_name
              << " (" << consumers.size() << " consumers redirected)";
    }

    if (to_remove.empty()) {
      return;  // nothing safe to eliminate — done
    }
    GraphSafeRemoveNodes(graph.get(), to_remove);
    // Loop again to collapse chains; assign_nodes was a snapshot.
    // (The loop is bounded at 16 iterations by the for header; a model
    // with deeper chains simply ends after the cap with a WARNING.)
  }
  LOG(WARNING) << "assign_eliminate_pass: iteration bound reached";
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(assign_eliminate_pass,
                  paddle::lite::mir::AssignEliminatePass)
    .BindTargets({TARGET(kAny)});
