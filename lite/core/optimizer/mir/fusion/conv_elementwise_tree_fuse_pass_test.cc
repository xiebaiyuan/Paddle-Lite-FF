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
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/optimizer/mir/fusion/conv_elementwise_tree_fuser.h"
#include "lite/core/optimizer/mir/fusion/fusion_pass_test_util.h"

namespace paddle {
namespace lite {
namespace mir {

// conv2d(1x1) → elementwise_add(other_input) folds into conv with a
// SecondInput (fuse_elementwise_op_type=elementwise_add).
std::vector<TestOpDesc> MakeConvTree(bool with_bias) {
  TestOpDesc conv;
  conv.type = "conv2d";
  conv.inputs = {{"Input", {"x"}}, {"Filter", {"w"}}};
  conv.outputs = {{"Output", {"conv_out"}}};
  conv.int_attrs = {{"groups", 1}};
  conv.int_vector_attrs = {{"strides", {1, 1}},
                           {"paddings", {0, 0}},
                           {"dilations", {1, 1}}};
  if (with_bias) conv.inputs["Bias"] = {"conv_b"};
  std::vector<TestOpDesc> ops{conv};
  ops.push_back({"elementwise_add",
                 {{"X", {"x2"}}, {"Y", {"conv_out"}}},
                 {{"Out", {"out"}}},
                 {{"axis", -1}},
                 {},
                 {},
                 {}});
  return ops;
}

// NOTE on the return value of fuser(graph): FuseBase::operator() returns the
// number of *matched* subgraphs, not the number of successful fusions. So
// skip cases assert on node counts (CountOp), not on the fuser's return value.

TEST(ConvElementwiseTreeFuser, fuse_conv1x1_add) {
  auto scope = std::make_shared<Scope>();
  // conv weight must be 1x1 and persistable; both conv out and elementwise
  // out dims must match (1x1 conv keeps shape).
  auto* w = scope->Var("w")->GetMutable<lite::Tensor>();
  w->Resize(DDim({1, 1, 1, 1}));
  w->mutable_data<float>()[0] = 1.0f;
  auto* x = scope->Var("x")->GetMutable<lite::Tensor>();
  x->Resize(DDim({1, 1, 2, 2}));
  x->mutable_data<float>()[0] = 1.0f;
  auto* x2 = scope->Var("x2")->GetMutable<lite::Tensor>();
  x2->Resize(DDim({1, 1, 2, 2}));
  x2->mutable_data<float>()[0] = 1.0f;
  // conv_out / out must exist in scope so InsertNewNode's dims check can read
  // them (missing vars make the check bail out with empty dims).
  auto* conv_out = scope->Var("conv_out")->GetMutable<lite::Tensor>();
  conv_out->Resize(DDim({1, 1, 2, 2}));
  conv_out->mutable_data<float>()[0] = 1.0f;
  auto* out = scope->Var("out")->GetMutable<lite::Tensor>();
  out->Resize(DDim({1, 1, 2, 2}));
  out->mutable_data<float>()[0] = 1.0f;

  auto graph = BuildGraph(MakeConvTree(false), {"w"}, scope.get());
  ASSERT_EQ(CountOp(*graph, "conv2d"), 1);
  ASSERT_EQ(CountOp(*graph, "elementwise_add"), 1);

  fusion::ConvElementwiseTreeFuser fuser("conv2d", false, false, "elementwise_add");
  ASSERT_EQ(fuser.apply_impl(graph.get()), 1u);

  // single conv remains, carrying the fused elementwise marker; the
  // elementwise op is gone and its input x2 is now the conv's SecondInput.
  ASSERT_EQ(CountOp(*graph, "conv2d"), 1);
  ASSERT_EQ(CountOp(*graph, "elementwise_add"), 0);
}

TEST(ConvElementwiseTreeFuser, skip_non_1x1_conv) {
  auto scope = std::make_shared<Scope>();
  // 3x3 conv weight — fusion must be skipped (only 1x1 supported). The
  // pattern still matches, so the fuser returns 1; assert on the graph
  // instead: elementwise_add must survive.
  auto* w = scope->Var("w")->GetMutable<lite::Tensor>();
  w->Resize(DDim({1, 1, 3, 3}));
  w->mutable_data<float>()[0] = 1.0f;
  auto* x = scope->Var("x")->GetMutable<lite::Tensor>();
  x->Resize(DDim({1, 1, 4, 4}));
  x->mutable_data<float>()[0] = 1.0f;
  auto* x2 = scope->Var("x2")->GetMutable<lite::Tensor>();
  x2->Resize(DDim({1, 1, 4, 4}));
  x2->mutable_data<float>()[0] = 1.0f;
  auto* conv_out = scope->Var("conv_out")->GetMutable<lite::Tensor>();
  conv_out->Resize(DDim({1, 1, 4, 4}));
  conv_out->mutable_data<float>()[0] = 1.0f;
  auto* out = scope->Var("out")->GetMutable<lite::Tensor>();
  out->Resize(DDim({1, 1, 4, 4}));
  out->mutable_data<float>()[0] = 1.0f;

  auto graph = BuildGraph(MakeConvTree(false), {"w"}, scope.get());
  fusion::ConvElementwiseTreeFuser fuser("conv2d", false, false, "elementwise_add");
  ASSERT_EQ(fuser.apply_impl(graph.get()), 1u);  // pattern matched...
  // ...but the filter-dims check in InsertNewNode bailed out.
  ASSERT_EQ(CountOp(*graph, "conv2d"), 1);
  ASSERT_EQ(CountOp(*graph, "elementwise_add"), 1);
}

TEST(ConvElementwiseTreeFuser, skip_fused_conv) {
  // A conv already carrying fuse_elementwise_op_type must not re-fuse: the
  // conv teller rejects it, so the pattern does not match at all. The fused
  // conv form requires a SecondInput, which this graph lacks.
  auto scope = std::make_shared<Scope>();
  auto* w = scope->Var("w")->GetMutable<lite::Tensor>();
  w->Resize(DDim({1, 1, 1, 1}));
  w->mutable_data<float>()[0] = 1.0f;
  auto* x = scope->Var("x")->GetMutable<lite::Tensor>();
  x->Resize(DDim({1, 1, 2, 2}));
  x->mutable_data<float>()[0] = 1.0f;
  auto* x2 = scope->Var("x2")->GetMutable<lite::Tensor>();
  x2->Resize(DDim({1, 1, 2, 2}));
  x2->mutable_data<float>()[0] = 1.0f;
  auto* conv_out = scope->Var("conv_out")->GetMutable<lite::Tensor>();
  conv_out->Resize(DDim({1, 1, 2, 2}));
  conv_out->mutable_data<float>()[0] = 1.0f;
  auto* out = scope->Var("out")->GetMutable<lite::Tensor>();
  out->Resize(DDim({1, 1, 2, 2}));
  out->mutable_data<float>()[0] = 1.0f;

  std::vector<TestOpDesc> ops = MakeConvTree(false);
  // A conv already carrying fuse_elementwise_op_type is the *fused* form:
  // it must have a SecondInput (conv_op.h AttachImpl reads it), and the
  // fuser's conv teller rejects it, so the pattern does not match at all.
  ops[0].str_attrs = {{"fuse_elementwise_op_type", "elementwise_add"}};
  ops[0].inputs["SecondInput"] = {"x2"};
  auto graph = BuildGraph(ops, {"w"}, scope.get());

  fusion::ConvElementwiseTreeFuser fuser("conv2d", false, false, "elementwise_add");
  ASSERT_EQ(fuser.apply_impl(graph.get()), 0u);  // conv teller rejects
  ASSERT_EQ(CountOp(*graph, "conv2d"), 1);
  ASSERT_EQ(CountOp(*graph, "elementwise_add"), 1);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
