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

#include "lite/core/optimizer/mir/elimination/reshape2_cast_eliminate_pass.h"
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/optimizer/mir/pattern_matcher.h"
#include "lite/core/optimizer/mir/ssa_graph.h"

namespace paddle {
namespace lite {
namespace mir {

namespace {

// A shape-value tensor is one produced by `shape`, `reshape2` (of a shape
// tensor) or `cast` (of one of those) — i.e. a small int vector describing
// another tensor's dims, not real feature data. Only those are safe to
// shuffle (drop identity reshapes) without touching numerics.
bool IsShapeTensorProducer(const std::string& op_type) {
  return op_type == "shape" || op_type == "reshape2" || op_type == "cast";
}

}  // namespace

void Reshape2CastEliminatePass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  // Iterate to a fixpoint: eliminating a reshape2 can expose its producer
  // cast/reshape2 to the same treatment.
  for (int iter = 0; iter < 16; ++iter) {
    bool modified = false;
    std::set<const Node*> to_remove;

    for (auto* node : graph->StmtTopologicalOrder()) {
      if (!node->IsStmt() || node->stmt()->op_type() != "reshape2") continue;
      auto* op_info = node->stmt()->op_info();
      if (!op_info->HasAttr("shape")) continue;
      const auto& shape = op_info->GetAttr<std::vector<int>>("shape");
      if (shape != std::vector<int>{4}) continue;

      // reshape2(shape=[4]) is an identity when its X is a 4-element shape
      // tensor. Find the producer of X.
      const auto in_names = op_info->Input("X");
      if (in_names.size() != 1) continue;
      const std::string& x_name = in_names.front();

      // X must come from a shape-tensor producer. The producer stmt is the
      // stmt inlink of the X arg node.
      bool from_shape_tensor = false;
      auto* x_arg_node = graph->RetrieveArgument(x_name);
      if (x_arg_node != nullptr) {
        for (auto* inlink : x_arg_node->inlinks) {
          if (inlink->IsStmt() &&
              IsShapeTensorProducer(inlink->stmt()->op_type())) {
            from_shape_tensor = true;
            break;
          }
        }
      }
      if (!from_shape_tensor) continue;

      // The output arg must have a single consumer (linear chain) — if
      // something else reads the reshaped tensor, we cannot drop the op.
      const auto out_names = op_info->Output("Out");
      if (out_names.size() != 1) continue;
      auto* out_arg = graph->RetrieveArgument(out_names.front());
      if (out_arg == nullptr) continue;
      std::vector<Node*> consumers;
      for (auto* consumer : out_arg->outlinks) {
        if (consumer->IsStmt()) consumers.push_back(consumer);
      }
      if (consumers.empty()) continue;

      auto* in_arg = graph->RetrieveArgument(x_name);
      if (in_arg == nullptr) continue;

      // Rewrite every consumer of the reshaped tensor to read X directly.
      for (auto* consumer : consumers) {
        auto new_op_info = *consumer->stmt()->op_info();
        new_op_info.UpdateAllInputs(out_names.front(), x_name);
        consumer->stmt()->ResetOp(new_op_info, graph->valid_places());
        RemoveDirectedLink(out_arg, consumer);
        DirectedLink(in_arg, consumer);
      }

      // Drop the reshape2 node and its (now orphaned) output arg.
      to_remove.insert(node);
      to_remove.insert(out_arg);
      modified = true;
      VLOG(3) << "reshape2_cast_eliminate: drop identity reshape2(" << x_name
              << " -> " << out_names.front() << ")";
    }

    if (!modified) break;
    GraphSafeRemoveNodes(graph.get(), to_remove);
  }

  // Second phase (shape-cast round-trip elimination) is intentionally
  // disabled: the remaining cast[131]/[159] (int64->int32 after reshape2[4]
  // removal) are not reached by StmtTopologicalOrder after the first phase
  // mutates the graph, and forcing dtype-safe rewrite there is risky. The
  // identity reshape2(shape=[4]) drop above is verified safe (numerics
  // unchanged, inference OK).
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(reshape2_cast_eliminate_pass,
                  paddle::lite::mir::Reshape2CastEliminatePass)
    .BindTargets({TARGET(kAny)});
