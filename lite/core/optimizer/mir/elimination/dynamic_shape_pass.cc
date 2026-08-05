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

#include "lite/core/optimizer/mir/elimination/dynamic_shape_pass.h"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/optimizer/mir/ssa_graph.h"

namespace paddle {
namespace lite {
namespace mir {

namespace {

// A width-derived value in a reshape2 shape attr is a positive multiple of 8
// (the conv-downsampling stride family: W/8, W*3/8, W/64, ...). Fixed channel
// dims like 192/120/15 happen to also be multiples of 8 in some models — so
// the multiple-of-8 test alone is NOT sufficient; see the element-count check
// below which is the real discriminator.
bool IsWidthDerivedCandidate(int v) { return v > 0 && v % 8 == 0; }

// Check whether replacing shape[idx] (a positive value) with -1 yields a
// valid reshape, given the input's element count. Mirrors the arithmetic in
// reshape_op.cc ValidateShape:
//   output_size = product of the OTHER known dims (0 → copy input dim, -1 →
//   unknown). The unknown dim must satisfy: unk * known_product == input_size.
// Returns true iff a unique positive integer resolves the unknown dim.
bool CanMakeDynamic(const std::vector<int>& shape,
                    int idx,
                    int64_t input_size) {
  // input_size must be positive and fully determined for the check to make
  // sense (if the input has a -1 dim, its production is negative and we
  // cannot prove divisibility — skip).
  if (input_size <= 0) return false;

  int64_t known_product = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i == static_cast<size_t>(idx)) continue;
    const int d = shape[i];
    if (d == 0) {
      // 0 copies the input's dim at this rank — its value is unknown at
      // compile time, so we cannot compute the known product. Skip: a 0-copy
      // + -1 combination cannot be proven safe statically.
      return false;
    }
    if (d < 0) {
      // Another -1 in the same shape: two unknown dims — invalid for -1
      // inference. Cannot make this one dynamic without ambiguity.
      return false;
    }
    known_product *= d;
  }
  if (known_product <= 0) return false;

  // unk = input_size / known_product must be a positive integer.
  const int64_t unk = input_size / known_product;
  return unk > 0 && unk * known_product == input_size;
}

}  // namespace

void DynamicShapePass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  // Pass 1: make feed input dims dynamic — but ONLY the dims that models
  // conventionally leave flexible: batch (dim 0) and the last dim (width /
  // sequence length). The intermediate dims (channel, height) are structural
  // constants that the downstream conv/downsample chain depends on; flipping
  // them to -1 (as the first revision did) produced all-(-1) inputs like
  // [-1,-1,-1,-1] and silently dropped the height=48 / channel=3 contract.
  //
  // Rule: for a fully-static input [1,3,48,320] we flip dims 0 and last →
  // [-1,3,48,-1]. For an input that already carries -1 in the flexible slots
  // (det: [-1,3,-1,-1], rec: [-1,3,48,-1]) we leave it untouched — the model
  // already declares which dims are dynamic. A partially-annotated input
  // ([1,3,-1,320]) gets the fixed batch/width slots flipped to match.
  for (auto* node : graph->StmtTopologicalOrder()) {
    if (!node->IsStmt() || node->stmt()->op_type() != "feed") continue;
    auto* op_info = node->stmt()->op_info();
    const auto out_names = op_info->Output("Out");
    if (out_names.size() != 1) continue;
    const std::string& out_name = out_names.front();

    auto* scope = node->stmt()->op()->scope();
    if (scope == nullptr) continue;
    auto* var = scope->FindVar(out_name);
    if (var == nullptr) continue;
    auto* tensor = var->GetMutable<lite::Tensor>();
    if (tensor == nullptr) continue;
    auto dims = tensor->dims();
    if (dims.size() == 0) {
      // Opt-time scope tensor is empty (dims not filled until inference);
      // nothing to make dynamic here — the reshape2 pass below is the real
      // fix, and the input dims stay as declared in the ProgramDesc.
      continue;
    }
    // Only the flexible slots: dim 0 (batch) and the last dim (width).
    std::vector<int> flexible;
    flexible.push_back(0);
    if (dims.size() > 1) flexible.push_back(static_cast<int>(dims.size()) - 1);
    bool modified = false;
    std::vector<int64_t> new_dims;
    new_dims.reserve(dims.size());
    for (int i = 0; i < dims.size(); ++i) {
      const int64_t d = dims[i];
      const bool is_flexible =
          std::find(flexible.begin(), flexible.end(), i) != flexible.end();
      new_dims.push_back((is_flexible && d > 0) ? -1 : d);
      if (is_flexible && d > 0) modified = true;
    }
    if (modified) {
      tensor->Resize(new_dims);
      VLOG(3) << "dynamic_shape_pass: feed " << out_name << " dims "
              << dims << " -> [" << new_dims[0] << ",...,"
              << new_dims.back() << "]";
    }
  }

  // Pass 2: make width-derived reshape2 shape attrs dynamic.
  for (auto* node : graph->StmtTopologicalOrder()) {
    if (!node->IsStmt()) continue;
    const std::string op_type = node->stmt()->op_type();
    if (op_type != "reshape" && op_type != "reshape2") continue;
    auto* op_info = node->stmt()->op_info();
    if (!op_info->HasAttr("shape")) continue;
    const auto shape = op_info->GetAttr<std::vector<int>>("shape");

    // Input element count from the scope tensor of X.
    const auto in_names = op_info->Input("X");
    if (in_names.size() != 1) continue;
    auto* scope = node->stmt()->op()->scope();
    if (scope == nullptr) continue;
    auto* in_var = scope->FindVar(in_names.front());
    if (in_var == nullptr) continue;
    auto* in_tensor = in_var->GetMutable<lite::Tensor>();
    if (in_tensor == nullptr) continue;
    const int64_t input_size = in_tensor->numel();
    if (input_size <= 0) continue;  // unknown dims — cannot prove safety

    // Find the LAST position that is a width-derived candidate AND can be
    // made dynamic. reshape/reshape2 allows at most one -1 (ValidateShape
    // CHECKs a single unknown dim), so when several values could be width-
    // derived we flip exactly one — the last one — keeping the earlier dims
    // as fixed anchors that pin down the unknown value at runtime. This
    // matches the observed patch semantics ([1,120,40] -> [1,120,-1],
    // [1,8,40,15] -> [1,8,-1,15]: the last candidate wins).
    int flip_idx = -1;
    for (size_t i = 0; i < shape.size(); ++i) {
      if (!IsWidthDerivedCandidate(shape[i])) continue;
      if (!CanMakeDynamic(shape, static_cast<int>(i), input_size)) continue;
      flip_idx = static_cast<int>(i);
    }
    if (flip_idx >= 0) {
      std::vector<int> new_shape = shape;
      new_shape[flip_idx] = -1;
      // op_info is const; copy-and-ResetOp is the pattern used by the other
      // elimination passes to mutate an op's attrs.
      auto new_op_info = *node->stmt()->op_info();
      new_op_info.SetAttr<std::vector<int>>("shape", new_shape);
      node->stmt()->ResetOp(new_op_info, graph->valid_places());
      VLOG(3) << "dynamic_shape_pass: " << op_type << " shape -> -1";
    }
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(dynamic_shape_pass, paddle::lite::mir::DynamicShapePass)
    .BindTargets({TARGET(kAny)});
