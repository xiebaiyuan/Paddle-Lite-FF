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
#include "lite/core/optimizer/mir/fusion/fusion_pass_test_util.h"
#include "lite/core/optimizer/mir/fusion/matmul_elementwise_add_fuser.h"

namespace paddle {
namespace lite {
namespace mir {

// matmul(W persistable, x) → elementwise_add(b) → (relu)? → out
// folds into fc.
std::vector<TestOpDesc> MakeMatmulAdd(bool with_relu) {
  std::vector<TestOpDesc> ops;
  ops.push_back({"matmul",
                 {{"X", {"x"}}, {"Y", {"w"}}},
                 {{"Out", {"mm_out"}}},
                 {},
                 {{"alpha", 1.0f}},
                 {{"transpose_X", false}, {"transpose_Y", false}},
                 {},
                 {}});
  ops.push_back({"elementwise_add",
                 {{"X", {"mm_out"}}, {"Y", {"b"}}},
                 {{"Out", {"add_out"}}},
                 {{"axis", -1}},
                 {},
                 {},
                 {}});
  if (with_relu) {
    ops.push_back({"relu", {{"X", {"add_out"}}}, {{"Out", {"out"}}}});
  }
  return ops;
}

TEST(MatmulElementwiseAddFuser, fuse_to_fc) {
  auto scope = std::make_shared<Scope>();
  auto* x = scope->Var("x")->GetMutable<lite::Tensor>();
  x->Resize(DDim({2, 3}));
  auto* w = scope->Var("w")->GetMutable<lite::Tensor>();
  w->Resize(DDim({3, 4}));
  auto* b = scope->Var("b")->GetMutable<lite::Tensor>();
  b->Resize(DDim({4}));
  auto* out = scope->Var("add_out")->GetMutable<lite::Tensor>();
  out->Resize(DDim({2, 4}));

  auto graph = BuildGraph(MakeMatmulAdd(false), {"w", "b"}, scope.get());
  ASSERT_EQ(CountOp(*graph, "matmul"), 1);
  ASSERT_EQ(CountOp(*graph, "elementwise_add"), 1);

  fusion::MatmulElementwiseAddFuser fuser(false, graph);
  ASSERT_EQ(fuser(graph.get()), 1u);

  ASSERT_EQ(CountOp(*graph, "matmul"), 0);
  ASSERT_EQ(CountOp(*graph, "elementwise_add"), 0);
  ASSERT_EQ(CountOp(*graph, "fc"), 1);
}

TEST(MatmulElementwiseAddFuser, fuse_to_fc_relu) {
  auto scope = std::make_shared<Scope>();
  auto* x = scope->Var("x")->GetMutable<lite::Tensor>();
  x->Resize(DDim({2, 3}));
  auto* w = scope->Var("w")->GetMutable<lite::Tensor>();
  w->Resize(DDim({3, 4}));
  auto* b = scope->Var("b")->GetMutable<lite::Tensor>();
  b->Resize(DDim({4}));
  auto* out = scope->Var("out")->GetMutable<lite::Tensor>();
  out->Resize(DDim({2, 4}));

  auto graph = BuildGraph(MakeMatmulAdd(true), {"w", "b"}, scope.get());
  fusion::MatmulElementwiseAddFuser fuser(true, graph);
  ASSERT_EQ(fuser(graph.get()), 1u);

  ASSERT_EQ(CountOp(*graph, "relu"), 0);
  ASSERT_EQ(CountOp(*graph, "fc"), 1);
}

TEST(MatmulElementwiseAddFuser, skip_transpose) {
  auto scope = std::make_shared<Scope>();
  auto* x = scope->Var("x")->GetMutable<lite::Tensor>();
  x->Resize(DDim({2, 3}));
  auto* w = scope->Var("w")->GetMutable<lite::Tensor>();
  w->Resize(DDim({3, 4}));
  auto* b = scope->Var("b")->GetMutable<lite::Tensor>();
  b->Resize(DDim({4}));
  auto* out = scope->Var("add_out")->GetMutable<lite::Tensor>();
  out->Resize(DDim({2, 4}));

  std::vector<TestOpDesc> ops = MakeMatmulAdd(false);
  ops[0].bool_attrs = {{"transpose_X", true}, {"transpose_Y", false}};
  auto graph = BuildGraph(ops, {"w", "b"}, scope.get());
  fusion::MatmulElementwiseAddFuser fuser(false, graph);
  ASSERT_EQ(fuser(graph.get()), 0u);
  ASSERT_EQ(CountOp(*graph, "matmul"), 1);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
