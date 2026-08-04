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
#include "lite/core/optimizer/mir/fusion/flatten_fc_fuser.h"
#include "lite/core/optimizer/mir/fusion/fusion_pass_test_util.h"

namespace paddle {
namespace lite {
namespace mir {

// flatten_contiguous_range → fc folds into fc (input becomes the flatten's
// input, op_type=mul).
std::vector<TestOpDesc> MakeFlattenFc(bool with_xshape) {
  TestOpDesc flatten;
  flatten.type = "flatten_contiguous_range";
  flatten.inputs = {{"X", {"x"}}};
  flatten.outputs = {{"Out", {"flat"}}};
  flatten.int_attrs = {{"start_axis", 1}, {"stop_axis", -1}};
  if (with_xshape) flatten.outputs["XShape"] = {"flat_xshape"};
  std::vector<TestOpDesc> ops{flatten};

  TestOpDesc fc;
  fc.type = "fc";
  fc.inputs = {{"Input", {"flat"}}, {"W", {"w"}}, {"Bias", {"b"}}};
  fc.outputs = {{"Out", {"out"}}};
  fc.int_attrs = {{"in_num_col_dims", 1}};
  ops.push_back(fc);
  return ops;
}

TEST(FlattenFcFuser, fuse_flatten_fc) {
  auto scope = std::make_shared<Scope>();
  // flatten input rank 4: start_axis=1 → real_start=1, in_num_col_dims=1 < 2 ok.
  auto* x = scope->Var("x")->GetMutable<lite::Tensor>();
  x->Resize(DDim({1, 2, 2, 2}));
  auto* w = scope->Var("w")->GetMutable<lite::Tensor>();
  w->Resize(DDim({8, 4}));
  auto* b = scope->Var("b")->GetMutable<lite::Tensor>();
  b->Resize(DDim({4}));
  auto* flat = scope->Var("flat")->GetMutable<lite::Tensor>();
  flat->Resize(DDim({1, 8}));

  auto graph = BuildGraph(MakeFlattenFc(true), {"w", "b"}, scope.get());
  ASSERT_EQ(CountOp(*graph, "flatten_contiguous_range"), 1);
  ASSERT_EQ(CountOp(*graph, "fc"), 1);

  fusion::FlattenFcFuser fuser(true);
  ASSERT_EQ(fuser(graph.get()), 1u);

  ASSERT_EQ(CountOp(*graph, "flatten_contiguous_range"), 0);
  ASSERT_EQ(CountOp(*graph, "fc"), 1);
}

TEST(FlattenFcFuser, fuse_flatten_fc_no_xshape) {
  auto scope = std::make_shared<Scope>();
  auto* x = scope->Var("x")->GetMutable<lite::Tensor>();
  x->Resize(DDim({1, 2, 2, 2}));
  auto* w = scope->Var("w")->GetMutable<lite::Tensor>();
  w->Resize(DDim({8, 4}));
  auto* b = scope->Var("b")->GetMutable<lite::Tensor>();
  b->Resize(DDim({4}));
  auto* flat = scope->Var("flat")->GetMutable<lite::Tensor>();
  flat->Resize(DDim({1, 8}));

  auto graph = BuildGraph(MakeFlattenFc(false), {"w", "b"}, scope.get());
  fusion::FlattenFcFuser fuser(false);
  ASSERT_EQ(fuser(graph.get()), 1u);
  ASSERT_EQ(CountOp(*graph, "flatten_contiguous_range"), 0);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
