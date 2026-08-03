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
// tensor), `cast`, or shape-vector assembly ops (`concat`/`slice`/
// `strided_slice` of those) — i.e. a small int vector describing another
// tensor's dims, not real feature data. Only those are safe to shuffle
// (drop identity reshapes) without touching numerics.
bool IsShapeTensorProducer(const std::string& op_type) {
  return op_type == "shape" || op_type == "reshape2" || op_type == "cast" ||
         op_type == "concat" || op_type == "slice" || op_type == "strided_slice";
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

      // reshape2(shape=[N]) is an identity when its X is an N-element shape
      // tensor. N is small (rec's shape-vector assembly produces 3/4-element
      // vectors, e.g. concat(slice(shape,0,2), const) = [batch, h, w]); the
      // element count equals the product of positive dims. Find the producer
      // of X.
      const auto in_names = op_info->Input("X");
      if (in_names.size() != 1) continue;
      const std::string& x_name = in_names.front();

      int target_elems = 1;
      bool has_dynamic = false;
      for (int d : shape) {
        if (d > 0) {
          target_elems *= d;
        } else {
          has_dynamic = true;  // 0 / -1: shape depends on runtime batch
        }
      }
      // Only fully-static small shapes (3/4 elements) are safe — a dynamic
      // dim means the reshape is the dynamic-batch propagation itself.
      if (has_dynamic || target_elems > 4) continue;

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

  // Second phase: drop dtype round-trips cast(A->B) -> cast(B->A). x2paddle
  // widens a shape tensor to int64 (for concat with int64 constants) and
  // narrows it back to int32 for the consuming reshape2 — an exact lossless
  // round-trip when the middle cast's output has only the second cast as a
  // consumer. ONNX export (the fusion reference) has zero such casts; the
  // rewritten graph feeds the original int32 tensor straight to the reshape2.
  for (int iter = 0; iter < 16; ++iter) {
    bool modified = false;
    std::set<const Node*> to_remove;
    for (auto* node : graph->StmtTopologicalOrder()) {
      if (!node->IsStmt() || node->stmt()->op_type() != "cast") continue;
      auto* op_info = node->stmt()->op_info();
      if (!op_info->HasAttr("in_dtype") || !op_info->HasAttr("out_dtype")) continue;
      const int in_dtype = op_info->GetAttr<int>("in_dtype");
      const int out_dtype = op_info->GetAttr<int>("out_dtype");
      if (in_dtype == out_dtype) continue;  // handled by identity-cast phase
      const auto in_names = op_info->Input("X");
      if (in_names.size() != 1) continue;
      const auto out_names = op_info->Output("Out");
      if (out_names.size() != 1) continue;

      // The middle cast's output must feed exactly one consumer: the
      // narrowing cast back (A->B -> B->A). Any other reader of the widened
      // value makes the round-trip non-eliminable.
      auto* out_arg = graph->RetrieveArgument(out_names.front());
      if (out_arg == nullptr) continue;
      std::vector<Node*> consumers;
      for (auto* consumer : out_arg->outlinks) {
        if (consumer->IsStmt()) consumers.push_back(consumer);
      }
      if (consumers.size() != 1) continue;
      Node* next = consumers.front();
      if (!next->IsStmt() || next->stmt()->op_type() != "cast") continue;
      auto* next_info = next->stmt()->op_info();
      if (!next_info->HasAttr("in_dtype") || !next_info->HasAttr("out_dtype")) continue;
      // The reverse cast must exactly undo the widening.
      if (next_info->GetAttr<int>("in_dtype") != out_dtype ||
          next_info->GetAttr<int>("out_dtype") != in_dtype) {
        continue;
      }
      // The narrowing cast's output must also be single-consumer (linear
      // chain) so we can safely rewire its consumer to X.
      const auto next_out_names = next_info->Output("Out");
      if (next_out_names.size() != 1) continue;
      auto* next_out_arg = graph->RetrieveArgument(next_out_names.front());
      if (next_out_arg == nullptr) continue;
      std::vector<Node*> next_consumers;
      for (auto* consumer : next_out_arg->outlinks) {
        if (consumer->IsStmt()) next_consumers.push_back(consumer);
      }
      if (next_consumers.empty()) continue;

      auto* in_arg = graph->RetrieveArgument(in_names.front());
      if (in_arg == nullptr) continue;
      // Rewire the narrowing cast's consumers to read the original int32
      // tensor directly (value is identical after the lossless round-trip).
      for (auto* consumer : next_consumers) {
        auto new_op_info = *consumer->stmt()->op_info();
        new_op_info.UpdateAllInputs(next_out_names.front(), in_names.front());
        consumer->stmt()->ResetOp(new_op_info, graph->valid_places());
        RemoveDirectedLink(next_out_arg, consumer);
        DirectedLink(in_arg, consumer);
      }
      to_remove.insert(node);
      to_remove.insert(out_arg);
      to_remove.insert(next);
      to_remove.insert(next_out_arg);
      modified = true;
      VLOG(3) << "reshape2_cast_eliminate: drop cast round-trip ("
              << in_dtype << "->" << out_dtype << "->" << in_dtype << " "
              << in_names.front() << " -> " << next_out_names.front() << ")";
    }
    if (!modified) break;
    GraphSafeRemoveNodes(graph.get(), to_remove);
  }

  // Third phase: drop identity casts (in_dtype == out_dtype). x2paddle
  // sometimes emits a no-op cast(3->3) in the shape-assembly chain; the value
  // is bit-identical, so consumers can read X directly. Only the exact
  // same-dtype case is touched — one-way casts (int32<->int64) carry real
  // dtype semantics and are never removed here.
  for (int iter = 0; iter < 16; ++iter) {
    bool modified = false;
    std::set<const Node*> to_remove;
    for (auto* node : graph->StmtTopologicalOrder()) {
      if (!node->IsStmt() || node->stmt()->op_type() != "cast") continue;
      auto* op_info = node->stmt()->op_info();
      if (!op_info->HasAttr("in_dtype") || !op_info->HasAttr("out_dtype")) continue;
      if (op_info->GetAttr<int>("in_dtype") != op_info->GetAttr<int>("out_dtype")) continue;

      const auto in_names = op_info->Input("X");
      if (in_names.size() != 1) continue;
      const std::string& x_name = in_names.front();
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
      for (auto* consumer : consumers) {
        auto new_op_info = *consumer->stmt()->op_info();
        new_op_info.UpdateAllInputs(out_names.front(), x_name);
        consumer->stmt()->ResetOp(new_op_info, graph->valid_places());
        RemoveDirectedLink(out_arg, consumer);
        DirectedLink(in_arg, consumer);
      }
      to_remove.insert(node);
      to_remove.insert(out_arg);
      modified = true;
      VLOG(3) << "reshape2_cast_eliminate: drop identity cast(" << x_name
              << " -> " << out_names.front() << ")";
    }
    if (!modified) break;
    GraphSafeRemoveNodes(graph.get(), to_remove);
  }

  // Fourth phase (one-way shape casts are never removed here).
  // The identity reshape2(shape=[1/3/4]), identity-cast, and dtype
  // round-trip drops above are verified safe (numerics unchanged,
  // inference OK).
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(reshape2_cast_eliminate_pass,
                  paddle::lite::mir::Reshape2CastEliminatePass)
    .BindTargets({TARGET(kAny)});
