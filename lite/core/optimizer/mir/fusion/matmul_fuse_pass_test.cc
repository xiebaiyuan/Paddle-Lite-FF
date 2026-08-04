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
#include "lite/core/optimizer/mir/fusion/matmul_fuser.h"

namespace paddle {
namespace lite {
namespace mir {

// matmul(rank-2, no transpose, alpha=1) → mul.
std::vector<TestOpDesc> MakeMatmul() {
  return {TestOpDesc{"matmul",
                     {{"X", {"x"}}, {"Y", {"y"}}},
                     {{"Out", {"out"}}},
                     {},
                     {{"alpha", 1.0f}},
                     {{"transpose_X", false}, {"transpose_Y", false}},
                     {}}};
}

TEST(MatmulFuser, fuse_matmul_to_mul) {
  auto scope = std::make_shared<Scope>();
  auto* x = scope->Var("x")->GetMutable<lite::Tensor>();
  x->Resize(DDim({2, 3}));
  auto* y = scope->Var("y")->GetMutable<lite::Tensor>();
  y->Resize(DDim({3, 4}));
  auto* out = scope->Var("out")->GetMutable<lite::Tensor>();
  out->Resize(DDim({2, 4}));

  auto graph = BuildGraph(MakeMatmul(), {}, scope.get());
  ASSERT_EQ(CountOp(*graph, "matmul"), 1);

  fusion::MatmulFuser fuser;
  ASSERT_EQ(fuser(graph.get()), 1u);

  ASSERT_EQ(CountOp(*graph, "matmul"), 0);
  ASSERT_EQ(CountOp(*graph, "mul"), 1);
}

TEST(MatmulFuser, skip_non_rank2) {
  auto scope = std::make_shared<Scope>();
  auto* x = scope->Var("x")->GetMutable<lite::Tensor>();
  x->Resize(DDim({1, 2, 3}));  // rank 3
  auto* y = scope->Var("y")->GetMutable<lite::Tensor>();
  y->Resize(DDim({3, 4}));
  auto* out = scope->Var("out")->GetMutable<lite::Tensor>();
  out->Resize(DDim({1, 2, 4}));

  auto graph = BuildGraph(MakeMatmul(), {}, scope.get());
  fusion::MatmulFuser fuser;
  ASSERT_EQ(fuser(graph.get()), 0u);
  ASSERT_EQ(CountOp(*graph, "matmul"), 1);
}

TEST(MatmulFuser, skip_transpose) {
  auto scope = std::make_shared<Scope>();
  auto* x = scope->Var("x")->GetMutable<lite::Tensor>();
  x->Resize(DDim({2, 3}));
  auto* y = scope->Var("y")->GetMutable<lite::Tensor>();
  y->Resize(DDim({3, 4}));
  auto* out = scope->Var("out")->GetMutable<lite::Tensor>();
  out->Resize(DDim({2, 4}));

  std::vector<TestOpDesc> ops = MakeMatmul();
  ops[0].bool_attrs = {{"transpose_X", true}, {"transpose_Y", false}};
  auto graph = BuildGraph(ops, {}, scope.get());
  fusion::MatmulFuser fuser;
  ASSERT_EQ(fuser(graph.get()), 0u);
  ASSERT_EQ(CountOp(*graph, "matmul"), 1);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
