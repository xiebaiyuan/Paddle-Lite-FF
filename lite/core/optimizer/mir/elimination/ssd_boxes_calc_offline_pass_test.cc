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

#include <gtest/gtest.h>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/optimizer/mir/elimination/ssd_boxes_calc_offline_pass.h"
#include "lite/core/optimizer/mir/fusion/fusion_pass_test_util.h"
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

namespace {

// Pre-fill a persistable float tensor in the root scope with `value`.
void FillTensor(Scope* scope,
                const std::string& name,
                const std::vector<int64_t>& dims,
                float value = 0.f) {
  auto* t = scope->Var(name)->GetMutable<lite::Tensor>();
  t->Resize(dims);
  t->set_precision(PRECISION(kFloat));
  float* data = t->mutable_data<float>();
  for (int64_t i = 0; i < t->numel(); ++i) {
    data[i] = value;
  }
}

// The intermediate/output tensors of the SSD chain are non-persistable, so
// they live in the exec scope (a child of the test's root scope) created by
// BuildGraph. Resize them there so the offline computation produces the
// expected dims (ComputeReshape/ComputeFlatten restore `out`'s pre-existing
// dims after CopyDataFrom).
void ResizeExecTensor(Scope* exec_scope,
                      const std::string& name,
                      const std::vector<int64_t>& dims) {
  auto* t = exec_scope->FindVar(name)->GetMutable<lite::Tensor>();
  t->Resize(dims);
  t->set_precision(PRECISION(kFloat));
}

// The real SSD subgraph (see ssd_boxes_calc_offline_pass.h):
//   conv_feat --prior_box--> boxes/variances
//   boxes --reshape2--> boxes_reshaped --concat--> all_boxes --sigmoid-->
// The sigmoid consumer survives the pass and is used to reach the exec scope
// (which holds the non-persistable intermediate tensors).
std::vector<TestOpDesc> MakeSSDReshapeChain() {
  std::vector<TestOpDesc> ops;
  ops.push_back({"prior_box",
                 {{"Input", {"conv_feat"}}, {"Image", {"img"}}},
                 {{"Boxes", {"boxes"}}, {"Variances", {"variances"}}},
                 {},
                 {{"step_h", 0.f}, {"step_w", 0.f}, {"offset", 0.5f}},
                 {{"clip", true}, {"flip", true}},
                 {},
                 {},
                 {{"min_sizes", std::vector<float>{16.f}},
                  {"max_sizes", std::vector<float>{32.f}},
                  {"aspect_ratios", std::vector<float>{2.f}},
                  {"variances", std::vector<float>{0.1f, 0.1f, 0.2f, 0.2f}}},
                 {{"order", std::vector<std::string>{"min", "max"}}}});
  ops.push_back({"reshape2",
                 {{"X", {"boxes"}}},
                 {{"Out", {"boxes_reshaped"}}, {"XShape", {"boxes_xshape"}}},
                 {},
                 {},
                 {{"inplace", false}},
                 {},
                 {{"shape", std::vector<int>{0, -1}}}});
  ops.push_back({"concat",
                 {{"X", {"boxes_reshaped"}}},
                 {{"Out", {"all_boxes"}}},
                 {{"axis", 0}},
                 {},
                 {},
                 {},
                 {}});
  ops.push_back({"sigmoid",
                 {{"X", {"all_boxes"}}},
                 {{"Out", {"prob"}}},
                 {},
                 {},
                 {},
                 {}});
  return ops;
}

// Same subgraph but with `flatten` in place of `reshape2`.
std::vector<TestOpDesc> MakeSSDFlattenChain() {
  std::vector<TestOpDesc> ops = MakeSSDReshapeChain();
  TestOpDesc flat;
  flat.type = "flatten";
  flat.inputs = {{"X", {"boxes"}}};
  flat.outputs = {{"Out", {"boxes_flat"}}};
  flat.int_attrs = {{"axis", 1}};
  flat.bool_attrs = {{"inplace", false}};
  // concat now consumes the flattened boxes.
  ops[1] = flat;
  ops[2].inputs["X"] = {"boxes_flat"};
  return ops;
}

// Returns the exec scope held by the first surviving stmt node.
Scope* GetExecScope(const SSAGraph& graph) {
  for (auto& node : graph.nodes()) {
    if (node.IsStmt()) {
      return node.stmt()->op()->scope();
    }
  }
  return nullptr;
}

// Pre-set the dims of the exec-scope tensors used by the reshape chain.
void ResizeExecTensors(Scope* exec_scope) {
  // prior_num = aspect_ratios_vec.size() * min_sizes.size() + max_sizes.size()
  //           = 3 * 1 + 1 = 4, so the boxes tensor is 4x4x4x4 (numel 256).
  ResizeExecTensor(exec_scope, "boxes", {4, 4, 4, 4});
  ResizeExecTensor(exec_scope, "variances", {4, 4, 4, 4});
  ResizeExecTensor(exec_scope, "boxes_reshaped", {4, 64});
  ResizeExecTensor(exec_scope, "all_boxes", {4, 64});
}

}  // namespace

// Positive: prior_box -> reshape2 -> concat chain is computed offline and
// every op removed. The output tensor holds a real prior box
// (aspect-ratio-1, size = min_size for the first prior in the ordering).
TEST(SSDBoxesCalcOfflinePass, eliminate_priorbox_reshape_concat) {
  auto scope = std::make_shared<Scope>();
  // feature map 1x4x4x4 and image 1x3x32x32. step is derived from
  // img_w/feat_w = 8; center of first cell = (0.5*8, 0.5*8) = (4, 4).
  FillTensor(scope.get(), "conv_feat", {1, 4, 4, 4});
  FillTensor(scope.get(), "img", {1, 3, 32, 32});

  std::set<std::string> persistable{"conv_feat", "img"};
  auto graph = BuildGraph(MakeSSDReshapeChain(), persistable, scope.get());

  Scope* exec_scope = GetExecScope(*graph);
  ASSERT_NE(exec_scope, nullptr);
  ResizeExecTensors(exec_scope);

  ASSERT_EQ(CountOp(*graph, "prior_box"), 1);
  ASSERT_EQ(CountOp(*graph, "reshape2"), 1);
  ASSERT_EQ(CountOp(*graph, "concat"), 1);

  SSDBoxesCalcOfflinePass pass;
  pass.Apply(graph);

  ASSERT_EQ(CountOp(*graph, "prior_box"), 0);
  ASSERT_EQ(CountOp(*graph, "reshape2"), 0);
  ASSERT_EQ(CountOp(*graph, "concat"), 0);
  ASSERT_EQ(CountOp(*graph, "sigmoid"), 1);

  // Offline-computed prior box: first cell (w=0,h=0), aspect_ratio=1,
  // min_size=16. center=(4,4), box width/height=16.
  //   xmin = (4 - 8)/32 = -0.125  -> clipped to 0
  //   ymin = (4 - 8)/32 = -0.125  -> clipped to 0
  //   xmax = (4 + 8)/32 = 0.375
  //   ymax = (4 + 8)/32 = 0.375
  auto* boxes = exec_scope->FindVar("all_boxes")->GetMutable<lite::Tensor>();
  ASSERT_EQ(boxes->numel(), 4 * 64);
  const float* b = boxes->data<float>();
  ASSERT_NEAR(b[0], 0.0f, 1e-5f);
  ASSERT_NEAR(b[1], 0.0f, 1e-5f);
  ASSERT_NEAR(b[2], 0.375f, 1e-5f);
  ASSERT_NEAR(b[3], 0.375f, 1e-5f);

  // Variances are constant 0.1, 0.1, 0.2, 0.2.
  auto* var = exec_scope->FindVar("variances")->GetMutable<lite::Tensor>();
  ASSERT_EQ(var->numel(), 4 * 64);
  const float* v = var->data<float>();
  ASSERT_NEAR(v[0], 0.1f, 1e-6f);
  ASSERT_NEAR(v[1], 0.1f, 1e-6f);
  ASSERT_NEAR(v[2], 0.2f, 1e-6f);
  ASSERT_NEAR(v[3], 0.2f, 1e-6f);
}

// Positive: the flatten variant (prior_box -> flatten -> concat) is also
// computed offline.
TEST(SSDBoxesCalcOfflinePass, eliminate_priorbox_flatten_concat) {
  auto scope = std::make_shared<Scope>();
  FillTensor(scope.get(), "conv_feat", {1, 4, 4, 4});
  FillTensor(scope.get(), "img", {1, 3, 32, 32});

  std::set<std::string> persistable{"conv_feat", "img"};
  auto graph = BuildGraph(MakeSSDFlattenChain(), persistable, scope.get());

  Scope* exec_scope = GetExecScope(*graph);
  ASSERT_NE(exec_scope, nullptr);
  ResizeExecTensor(exec_scope, "boxes", {4, 4, 4, 4});
  ResizeExecTensor(exec_scope, "variances", {4, 4, 4, 4});
  ResizeExecTensor(exec_scope, "boxes_flat", {4, 64});
  ResizeExecTensor(exec_scope, "all_boxes", {4, 64});

  ASSERT_EQ(CountOp(*graph, "prior_box"), 1);
  ASSERT_EQ(CountOp(*graph, "flatten"), 1);
  ASSERT_EQ(CountOp(*graph, "concat"), 1);

  SSDBoxesCalcOfflinePass pass;
  pass.Apply(graph);

  ASSERT_EQ(CountOp(*graph, "prior_box"), 0);
  ASSERT_EQ(CountOp(*graph, "flatten"), 0);
  ASSERT_EQ(CountOp(*graph, "concat"), 0);
}

// Negative: when the feature-map height is negative (dynamic shape) the
// prior_box cannot be computed offline and must be preserved. The downstream
// reshape2/concat are also preserved because their inputs never get marked
// as prior-box outputs.
TEST(SSDBoxesCalcOfflinePass, keep_priorbox_when_shape_unknown) {
  auto scope = std::make_shared<Scope>();
  FillTensor(scope.get(), "conv_feat", {1, 4, 4, 4});
  FillTensor(scope.get(), "img", {1, 3, 32, 32});

  std::set<std::string> persistable{"conv_feat", "img"};
  auto graph = BuildGraph(MakeSSDReshapeChain(), persistable, scope.get());

  Scope* exec_scope = GetExecScope(*graph);
  ASSERT_NE(exec_scope, nullptr);
  // dynamic height: dims[2] == -1 makes the prior-box calc skip. Resize
  // without allocating (numel is negative) so nothing touches the data.
  auto* feat = exec_scope->FindVar("conv_feat")->GetMutable<lite::Tensor>();
  feat->Resize(DDim({1, 4, -1, 4}));
  feat->set_precision(PRECISION(kFloat));
  ResizeExecTensor(exec_scope, "boxes", {4, 4, 4, 4});
  ResizeExecTensor(exec_scope, "variances", {4, 4, 4, 4});
  ResizeExecTensor(exec_scope, "boxes_reshaped", {4, 64});
  ResizeExecTensor(exec_scope, "all_boxes", {4, 64});

  SSDBoxesCalcOfflinePass pass;
  pass.Apply(graph);

  // Whole chain preserved.
  ASSERT_EQ(CountOp(*graph, "prior_box"), 1);
  ASSERT_EQ(CountOp(*graph, "reshape2"), 1);
  ASSERT_EQ(CountOp(*graph, "concat"), 1);
}

// Positive: two independent prior_box subgraphs that share no variables are
// both eliminated.
TEST(SSDBoxesCalcOfflinePass, eliminate_two_priorbox_chains) {
  auto scope = std::make_shared<Scope>();
  FillTensor(scope.get(), "conv_feat", {1, 4, 4, 4});
  FillTensor(scope.get(), "conv_feat1", {1, 4, 4, 4});
  FillTensor(scope.get(), "img", {1, 3, 32, 32});

  std::vector<TestOpDesc> ops = MakeSSDReshapeChain();
  std::vector<TestOpDesc> ops2 = MakeSSDReshapeChain();
  // Rename the second chain's vars so the two chains are disjoint.
  for (auto& op : ops2) {
    for (auto& kv : op.inputs) {
      for (auto& name : kv.second) {
        if (name == "conv_feat") name += "1";
        if (name == "boxes") name += "1";
        if (name == "boxes_reshaped") name += "1";
        if (name == "all_boxes") name += "1";
        if (name == "prob") name += "1";
      }
    }
    for (auto& kv : op.outputs) {
      for (auto& name : kv.second) {
        if (name == "boxes") name += "1";
        if (name == "variances") name += "1";
        if (name == "boxes_reshaped") name += "1";
        if (name == "boxes_xshape") name += "1";
        if (name == "all_boxes") name += "1";
        if (name == "prob") name += "1";
      }
    }
  }
  ops.insert(ops.end(), ops2.begin(), ops2.end());

  std::set<std::string> persistable{"conv_feat", "conv_feat1", "img"};
  auto graph = BuildGraph(ops, persistable, scope.get());

  Scope* exec_scope = GetExecScope(*graph);
  ASSERT_NE(exec_scope, nullptr);
  ResizeExecTensor(exec_scope, "boxes", {4, 4, 4, 4});
  ResizeExecTensor(exec_scope, "variances", {4, 4, 4, 4});
  ResizeExecTensor(exec_scope, "boxes_reshaped", {4, 64});
  ResizeExecTensor(exec_scope, "all_boxes", {4, 64});
  ResizeExecTensor(exec_scope, "boxes1", {4, 4, 4, 4});
  ResizeExecTensor(exec_scope, "variances1", {4, 4, 4, 4});
  ResizeExecTensor(exec_scope, "boxes_reshaped1", {4, 64});
  ResizeExecTensor(exec_scope, "all_boxes1", {4, 64});

  ASSERT_EQ(CountOp(*graph, "prior_box"), 2);
  ASSERT_EQ(CountOp(*graph, "concat"), 2);

  SSDBoxesCalcOfflinePass pass;
  pass.Apply(graph);

  ASSERT_EQ(CountOp(*graph, "prior_box"), 0);
  ASSERT_EQ(CountOp(*graph, "reshape2"), 0);
  ASSERT_EQ(CountOp(*graph, "concat"), 0);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
