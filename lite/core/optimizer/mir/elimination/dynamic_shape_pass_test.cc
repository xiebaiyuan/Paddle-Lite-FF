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
#include "lite/core/optimizer/mir/elimination/dynamic_shape_pass.h"
#include "lite/core/optimizer/mir/fusion/fusion_pass_test_util.h"
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

namespace {

// Resize the scope tensor `name` to `dims` so the pass can read its numel().
void ResizeTensor(Scope* scope,
                  const std::string& name,
                  const std::vector<int64_t>& dims) {
  auto* tensor = scope->Var(name)->GetMutable<lite::Tensor>();
  tensor->Resize(DDim(dims));
}

// Returns the shape attr of the first reshape/reshape2 stmt in `graph`.
std::vector<int> FirstReshapeShape(const SSAGraph& graph) {
  for (auto& node : graph.nodes()) {
    if (node.IsStmt()) {
      const std::string t = node.stmt()->op_info()->Type();
      if (t == "reshape" || t == "reshape2") {
        return node.stmt()->op_info()->GetAttr<std::vector<int>>("shape");
      }
    }
  }
  return {};
}

// A conv2d producing `out` from a feature input. ConvOp::AttachImpl requires
// strides/paddings/dilations/groups; the output tensor's dims are set by the
// caller via ResizeTensor so the pass can read its numel().
TestOpDesc MakeConvFeat(const std::string& out) {
  return {"conv2d",
          {{"Input", {"feat"}}, {"Filter", {"w"}}},
          {{"Output", {out}}},
          {{"groups", 1}},
          {},
          {},
          {},
          {{"strides", std::vector<int>{1, 1}},
           {"paddings", std::vector<int>{0, 0}},
           {"dilations", std::vector<int>{1, 1}}},
          {}};
}

}  // namespace

// Positive: reshape2 shape=[1, 120, 40] with input size 1*120*40 (W=320
// residue). 40 (= W/8) is a width-derived multiple of 8, and replacing it
// with -1 leaves known product 1*120 which divides 4800 exactly (unk=40).
// The value must become -1.
TEST(DynamicShapePass, make_width_derived_reshape2_dynamic) {
  std::vector<TestOpDesc> ops;
  ops.push_back(MakeConvFeat("feat_out"));
  ops.push_back({"reshape2",
                 {{"X", {"feat_out"}}},
                 {{"Out", {"reshaped"}}, {"XShape", {"reshaped_xshape"}}},
                 {},
                 {},
                 {},
                 {},
                 {{"shape", std::vector<int>{1, 120, 40}}}});
  auto scope = std::make_shared<Scope>();
  // feat_out has 1*120*40 = 4800 elements (W=320 residue).
  ResizeTensor(scope.get(), "feat_out", {1, 120, 40});
  auto graph = BuildGraph(ops, {"w"}, scope.get());

  DynamicShapePass pass;
  pass.Apply(graph);

  const auto shape = FirstReshapeShape(*graph);
  ASSERT_EQ(shape.size(), 3u);
  EXPECT_EQ(shape[0], 1);
  EXPECT_EQ(shape[1], 120);
  EXPECT_EQ(shape[2], -1) << "40 (=W/8) should become -1";
}

// Positive: multiple width-derived values in one shape become -1.
// [1, 8, 40, 15]: both 40 (=W/8) and 15 (=W/64, not a multiple of 8 itself
// but derived) — 40 passes the 8-multiple test; 15 does not. Only 40 flips.
TEST(DynamicShapePass, only_8_multiple_values_flip) {
  std::vector<TestOpDesc> ops;
  ops.push_back(MakeConvFeat("feat_out"));
  ops.push_back({"reshape2",
                 {{"X", {"feat_out"}}},
                 {{"Out", {"reshaped"}}, {"XShape", {"reshaped_xshape"}}},
                 {},
                 {},
                 {},
                 {},
                 {{"shape", std::vector<int>{1, 8, 40, 15}}}});
  auto scope = std::make_shared<Scope>();
  ResizeTensor(scope.get(), "feat_out", {1, 8, 40, 15});
  auto graph = BuildGraph(ops, {"w"}, scope.get());

  DynamicShapePass pass;
  pass.Apply(graph);

  const auto shape = FirstReshapeShape(*graph);
  ASSERT_EQ(shape.size(), 4u);
  EXPECT_EQ(shape[2], -1);
  EXPECT_EQ(shape[3], 15) << "15 is not a multiple of 8; must be preserved";
}

// Negative: a fixed channel dim that is a multiple of 8 but cannot be made
// dynamic without breaking element-count conservation is left alone.
// shape=[1, 192, 40] but input size is 1*192*40*2 (an extra factor 2): making
// 40 -> -1 gives known product 1*192, and 7680/192 = 40 — wait, that still
// divides. Use input size that does NOT divide: 1*192*40 + 1.
TEST(DynamicShapePass, keep_when_unk_does_not_divide) {
  std::vector<TestOpDesc> ops;
  ops.push_back(MakeConvFeat("feat_out"));
  ops.push_back({"reshape2",
                 {{"X", {"feat_out"}}},
                 {{"Out", {"reshaped"}}, {"XShape", {"reshaped_xshape"}}},
                 {},
                 {},
                 {},
                 {},
                 {{"shape", std::vector<int>{1, 192, 40}}}});
  auto scope = std::make_shared<Scope>();
  // 1*192*40 + 1 = 7681 — not divisible by 192, so unk would not be an
  // integer; the reshape cannot be dynamic. Shape must be preserved.
  ResizeTensor(scope.get(), "feat_out", {7681});
  auto graph = BuildGraph(ops, {"w"}, scope.get());

  DynamicShapePass pass;
  pass.Apply(graph);

  const auto shape = FirstReshapeShape(*graph);
  ASSERT_EQ(shape.size(), 3u);
  EXPECT_EQ(shape[0], 1);
  EXPECT_EQ(shape[1], 192);
  EXPECT_EQ(shape[2], 40) << "cannot prove dynamic; must be preserved";
}

// Negative: two -1 dims in the shape make -1 inference ambiguous; the
// remaining positive multiple of 8 must NOT be flipped (CanMakeDynamic bails
// on the second -1).
TEST(DynamicShapePass, keep_when_two_unknown_dims) {
  std::vector<TestOpDesc> ops;
  ops.push_back(MakeConvFeat("feat_out"));
  ops.push_back({"reshape2",
                 {{"X", {"feat_out"}}},
                 {{"Out", {"reshaped"}}, {"XShape", {"reshaped_xshape"}}},
                 {},
                 {},
                 {},
                 {},
                 {{"shape", std::vector<int>{-1, 8, 40, -1}}}});
  auto scope = std::make_shared<Scope>();
  ResizeTensor(scope.get(), "feat_out", {1, 8, 40, 15});
  auto graph = BuildGraph(ops, {"w"}, scope.get());

  DynamicShapePass pass;
  pass.Apply(graph);

  const auto shape = FirstReshapeShape(*graph);
  ASSERT_EQ(shape.size(), 4u);
  EXPECT_EQ(shape[0], -1);
  EXPECT_EQ(shape[1], 8);
  EXPECT_EQ(shape[2], 40) << "two -1s: cannot flip 40";
  EXPECT_EQ(shape[3], -1);
}

// Negative: values that are NOT multiples of 8 are never touched. Here 40
// IS a width-derived multiple of 8 and flips to -1; 7 (not a multiple of 8)
// is preserved.
TEST(DynamicShapePass, keep_non_multiple_of_8) {
  std::vector<TestOpDesc> ops;
  ops.push_back(MakeConvFeat("feat_out"));
  ops.push_back({"reshape2",
                 {{"X", {"feat_out"}}},
                 {{"Out", {"reshaped"}}, {"XShape", {"reshaped_xshape"}}},
                 {},
                 {},
                 {},
                 {},
                 {{"shape", std::vector<int>{1, 40, 7}}}});
  auto scope = std::make_shared<Scope>();
  ResizeTensor(scope.get(), "feat_out", {1, 40, 7});
  auto graph = BuildGraph(ops, {"w"}, scope.get());

  DynamicShapePass pass;
  pass.Apply(graph);

  const auto shape = FirstReshapeShape(*graph);
  ASSERT_EQ(shape.size(), 3u);
  EXPECT_EQ(shape[0], 1);
  EXPECT_EQ(shape[1], -1) << "40 is a width-derived multiple of 8; flips";
  EXPECT_EQ(shape[2], 7) << "7 not a multiple of 8; preserved";
}

// Feed input dims rule: a fully-static input [1,3,48,320] must become
// [-1,3,48,-1] — batch and width flex, channel and height stay fixed.
// Regression (3b3c5fc): the first revision flipped every positive dim,
// producing [-1,-1,-1,-1] and silently dropping the height=48 / channel=3
// contract that the downstream conv/downsample chain depends on.
TEST(DynamicShapePass, feed_only_batch_and_width_become_dynamic) {
  const auto dims = DynamicShapePass::MakeFeedDimsDynamic(DDim({1, 3, 48, 320}));
  ASSERT_EQ(dims.size(), 4u);
  EXPECT_EQ(dims[0], -1) << "batch is flexible";
  EXPECT_EQ(dims[1], 3) << "channel stays fixed";
  EXPECT_EQ(dims[2], 48) << "height stays fixed";
  EXPECT_EQ(dims[3], -1) << "width is flexible";
}

// Feed input dims: a source that already declares dynamic dims (rec:
// [-1,3,48,-1]) must be left untouched — the model already expresses which
// dims are flexible, and the pass must not broaden them.
TEST(DynamicShapePass, feed_already_dynamic_untouched) {
  const auto dims = DynamicShapePass::MakeFeedDimsDynamic(DDim({-1, 3, 48, -1}));
  ASSERT_EQ(dims.size(), 4u);
  EXPECT_EQ(dims[0], -1);
  EXPECT_EQ(dims[1], 3);
  EXPECT_EQ(dims[2], 48);
  EXPECT_EQ(dims[3], -1);
}

// Feed input dims: det-style input [-1,3,-1,-1] (batch/h/w flex, channel
// fixed) is likewise untouched.
TEST(DynamicShapePass, feed_det_style_untouched) {
  const auto dims = DynamicShapePass::MakeFeedDimsDynamic(DDim({-1, 3, -1, -1}));
  ASSERT_EQ(dims.size(), 4u);
  EXPECT_EQ(dims[0], -1);
  EXPECT_EQ(dims[1], 3);
  EXPECT_EQ(dims[2], -1);
  EXPECT_EQ(dims[3], -1);
}

// Feed input dims: a 2-D input [1, 320] (batch, seq-len) becomes [-1, -1].
TEST(DynamicShapePass, feed_2d_both_flexible) {
  const auto dims = DynamicShapePass::MakeFeedDimsDynamic(DDim({1, 320}));
  ASSERT_EQ(dims.size(), 2u);
  EXPECT_EQ(dims[0], -1);
  EXPECT_EQ(dims[1], -1);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
