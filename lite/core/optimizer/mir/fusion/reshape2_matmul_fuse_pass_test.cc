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
#include "lite/core/optimizer/mir/fusion/reshape2_matmul_fuser.h"

namespace paddle {
namespace lite {
namespace mir {

// reshape2([N,C,1,1] → [N,C]) → matmul → mul
std::vector<TestOpDesc> MakeReshape2Matmul() {
  std::vector<TestOpDesc> ops;
  ops.push_back({"reshape2",
                 {{"X", {"x"}}},
                 {{"Out", {"reshaped"}}, {"XShape", {"xs"}}},
                 {},
                 {},
                 {},
                 {},
                 {{"shape", {2, 3}}}});
  ops.push_back({"matmul",
                 {{"X", {"reshaped"}}, {"Y", {"y"}}},
                 {{"Out", {"out"}}},
                 {},
                 {{"alpha", 1.0f}},
                 {{"transpose_X", false}, {"transpose_Y", false}},
                 {}});
  return ops;
}

TEST(Reshape2MatmulFuser, fuse_to_mul) {
  auto scope = std::make_shared<Scope>();
  auto* x = scope->Var("x")->GetMutable<lite::Tensor>();
  x->Resize(DDim({2, 3, 1, 1}));  // rank 4, last dims 1
  auto* xs = scope->Var("xs")->GetMutable<lite::Tensor>();
  xs->Resize(DDim({5}));
  auto* reshaped = scope->Var("reshaped")->GetMutable<lite::Tensor>();
  reshaped->Resize(DDim({2, 3}));
  auto* y = scope->Var("y")->GetMutable<lite::Tensor>();
  y->Resize(DDim({3, 4}));
  auto* out = scope->Var("out")->GetMutable<lite::Tensor>();
  out->Resize(DDim({2, 4}));

  auto graph = BuildGraph(MakeReshape2Matmul(), {}, scope.get());
  ASSERT_EQ(CountOp(*graph, "reshape2"), 1);
  ASSERT_EQ(CountOp(*graph, "matmul"), 1);

  fusion::Reshape2MatmulFuser fuser;
  ASSERT_EQ(fuser(graph.get()), 1u);

  ASSERT_EQ(CountOp(*graph, "reshape2"), 0);
  ASSERT_EQ(CountOp(*graph, "matmul"), 0);
  ASSERT_EQ(CountOp(*graph, "mul"), 1);
}

TEST(Reshape2MatmulFuser, skip_wrong_reshape_shape) {
  auto scope = std::make_shared<Scope>();
  auto* x = scope->Var("x")->GetMutable<lite::Tensor>();
  x->Resize(DDim({2, 3, 1, 1}));
  auto* xs = scope->Var("xs")->GetMutable<lite::Tensor>();
  xs->Resize(DDim({5}));
  auto* reshaped = scope->Var("reshaped")->GetMutable<lite::Tensor>();
  reshaped->Resize(DDim({6}));
  auto* y = scope->Var("y")->GetMutable<lite::Tensor>();
  y->Resize(DDim({6, 4}));
  auto* out = scope->Var("out")->GetMutable<lite::Tensor>();
  out->Resize(DDim({2, 4}));

  std::vector<TestOpDesc> ops = MakeReshape2Matmul();
  ops[0].int_vector_attrs = {{"shape", {6}}};  // not rank 2
  auto graph = BuildGraph(ops, {}, scope.get());
  fusion::Reshape2MatmulFuser fuser;
  ASSERT_EQ(fuser(graph.get()), 0u);
  ASSERT_EQ(CountOp(*graph, "reshape2"), 1);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
