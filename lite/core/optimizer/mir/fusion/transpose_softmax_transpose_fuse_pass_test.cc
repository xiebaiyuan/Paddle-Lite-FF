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
#include "lite/core/optimizer/mir/fusion/transpose_softmax_transpose_fuser.h"

namespace paddle {
namespace lite {
namespace mir {

// transpose → softmax(axis=-1) → transpose folds into a single softmax.
std::vector<TestOpDesc> MakeTransposeSoftmaxTranspose(
    const std::string& transpose_type) {
  std::vector<TestOpDesc> ops;
  TestOpDesc t1;
  t1.type = transpose_type;
  t1.inputs = {{"X", {"x"}}};
  t1.outputs = {{"Out", {"t1_out"}}};
  t1.int_vector_attrs = {{"axis", {0, 2, 1, 3}}};
  if (transpose_type == "transpose2") t1.outputs["XShape"] = {"xs1"};
  ops.push_back(t1);

  TestOpDesc sm;
  sm.type = "softmax";
  sm.inputs = {{"X", {"t1_out"}}};
  sm.outputs = {{"Out", {"sm_out"}}};
  sm.int_attrs = {{"axis", -1}};
  ops.push_back(sm);

  TestOpDesc t2;
  t2.type = transpose_type;
  t2.inputs = {{"X", {"sm_out"}}};
  t2.outputs = {{"Out", {"out"}}};
  t2.int_vector_attrs = {{"axis", {0, 2, 1, 3}}};
  if (transpose_type == "transpose2") t2.outputs["XShape"] = {"xs2"};
  ops.push_back(t2);
  return ops;
}

TEST(TransposeSoftmaxTransposeFuser, fuse_transpose2) {
  auto scope = std::make_shared<Scope>();
  auto* x = scope->Var("x")->GetMutable<lite::Tensor>();
  x->Resize(DDim({1, 2, 3, 4}));
  auto* t1 = scope->Var("t1_out")->GetMutable<lite::Tensor>();
  t1->Resize(DDim({1, 3, 2, 4}));
  auto* sm = scope->Var("sm_out")->GetMutable<lite::Tensor>();
  sm->Resize(DDim({1, 3, 2, 4}));
  auto* out = scope->Var("out")->GetMutable<lite::Tensor>();
  out->Resize(DDim({1, 2, 3, 4}));
  for (const auto& n : {"xs1", "xs2"}) {
    auto* xs = scope->Var(n)->GetMutable<lite::Tensor>();
    xs->Resize(DDim({5}));
  }

  auto graph =
      BuildGraph(MakeTransposeSoftmaxTranspose("transpose2"), {}, scope.get());
  ASSERT_EQ(CountOp(*graph, "transpose2"), 2);
  ASSERT_EQ(CountOp(*graph, "softmax"), 1);

  fusion::TransposeSoftmaxTransposeFuser fuser("transpose2", "softmax");
  ASSERT_EQ(fuser(graph.get()), 1u);

  ASSERT_EQ(CountOp(*graph, "transpose2"), 0);
  ASSERT_EQ(CountOp(*graph, "softmax"), 1);
}

TEST(TransposeSoftmaxTransposeFuser, fuse_transpose) {
  auto scope = std::make_shared<Scope>();
  auto* x = scope->Var("x")->GetMutable<lite::Tensor>();
  x->Resize(DDim({1, 2, 3, 4}));
  auto* t1 = scope->Var("t1_out")->GetMutable<lite::Tensor>();
  t1->Resize(DDim({1, 3, 2, 4}));
  auto* sm = scope->Var("sm_out")->GetMutable<lite::Tensor>();
  sm->Resize(DDim({1, 3, 2, 4}));
  auto* out = scope->Var("out")->GetMutable<lite::Tensor>();
  out->Resize(DDim({1, 2, 3, 4}));

  auto graph =
      BuildGraph(MakeTransposeSoftmaxTranspose("transpose"), {}, scope.get());
  fusion::TransposeSoftmaxTransposeFuser fuser("transpose", "softmax");
  ASSERT_EQ(fuser(graph.get()), 1u);
  ASSERT_EQ(CountOp(*graph, "transpose"), 0);
  ASSERT_EQ(CountOp(*graph, "softmax"), 1);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
