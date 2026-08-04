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
#include "lite/core/optimizer/mir/fusion/fc_fuser.h"
#include "lite/core/optimizer/mir/fusion/fusion_pass_test_util.h"

namespace paddle {
namespace lite {
namespace mir {

// mul(x, W) → elementwise_add(+b) → [act] folds into fc.
std::vector<TestOpDesc> MakeMulAdd(const std::string& act_type) {
  TestOpDesc mul;
  mul.type = "mul";
  mul.inputs = {{"X", {"x"}}, {"Y", {"w"}}};
  mul.outputs = {{"Out", {"mul_out"}}};
  mul.int_attrs = {{"x_num_col_dims", 1}, {"y_num_col_dims", 1}};
  std::vector<TestOpDesc> ops{mul};
  ops.push_back({"elementwise_add",
                 {{"X", {"mul_out"}}, {"Y", {"b"}}},
                 {{"Out", {"add_out"}}},
                 {{"axis", -1}},
                 {},
                 {},
                 {}});
  if (!act_type.empty()) {
    ops.push_back({act_type,
                   {{"X", {"add_out"}}},
                   {{"Out", {"out"}}},
                   {},
                   {},
                   {},
                   {}});
  }
  return ops;
}

// The W weight must be rank-2 (inputs_teller0) and the bias rank-1/2 with
// the last dim matching W[1] (inputs_teller1 / InsertNewNode checks).
TEST(FcFuser, fuse_mul_add) {
  auto scope = std::make_shared<Scope>();
  auto* x = scope->Var("x")->GetMutable<lite::Tensor>();
  x->Resize(DDim({2, 3}));
  x->mutable_data<float>()[0] = 1.0f;
  auto* w = scope->Var("w")->GetMutable<lite::Tensor>();
  w->Resize(DDim({3, 4}));
  for (int i = 0; i < 12; ++i) w->mutable_data<float>()[i] = 1.0f;
  auto* b = scope->Var("b")->GetMutable<lite::Tensor>();
  b->Resize(DDim({4}));
  b->mutable_data<float>()[0] = 0.0f;

  auto graph = BuildGraph(MakeMulAdd(""), {"w", "b"}, scope.get());
  ASSERT_EQ(CountOp(*graph, "mul"), 1);
  ASSERT_EQ(CountOp(*graph, "elementwise_add"), 1);

  fusion::FcFuser fuser("mul", "");
  ASSERT_EQ(fuser(graph.get()), 1u);

  ASSERT_EQ(CountOp(*graph, "mul"), 0);
  ASSERT_EQ(CountOp(*graph, "elementwise_add"), 0);
  ASSERT_EQ(CountOp(*graph, "fc"), 1);
}

TEST(FcFuser, fuse_mul_add_relu) {
  auto scope = std::make_shared<Scope>();
  auto* x = scope->Var("x")->GetMutable<lite::Tensor>();
  x->Resize(DDim({2, 3}));
  x->mutable_data<float>()[0] = 1.0f;
  auto* w = scope->Var("w")->GetMutable<lite::Tensor>();
  w->Resize(DDim({3, 4}));
  for (int i = 0; i < 12; ++i) w->mutable_data<float>()[i] = 1.0f;
  auto* b = scope->Var("b")->GetMutable<lite::Tensor>();
  b->Resize(DDim({4}));
  b->mutable_data<float>()[0] = 0.0f;

  auto graph = BuildGraph(MakeMulAdd("relu"), {"w", "b"}, scope.get());
  fusion::FcFuser fuser("mul", "relu");
  ASSERT_EQ(fuser(graph.get()), 1u);
  ASSERT_EQ(CountOp(*graph, "mul"), 0);
  ASSERT_EQ(CountOp(*graph, "relu"), 0);
  ASSERT_EQ(CountOp(*graph, "fc"), 1);
}

TEST(FcFuser, skip_wrong_weight_rank) {
  // rank-3 weight fails inputs_teller0 → pattern does not match.
  auto scope = std::make_shared<Scope>();
  auto* x = scope->Var("x")->GetMutable<lite::Tensor>();
  x->Resize(DDim({2, 3}));
  auto* w = scope->Var("w")->GetMutable<lite::Tensor>();
  w->Resize(DDim({2, 3, 4}));
  auto* b = scope->Var("b")->GetMutable<lite::Tensor>();
  b->Resize(DDim({4}));

  auto graph = BuildGraph(MakeMulAdd(""), {"w", "b"}, scope.get());
  fusion::FcFuser fuser("mul", "");
  ASSERT_EQ(fuser(graph.get()), 0u);
  ASSERT_EQ(CountOp(*graph, "mul"), 1);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
