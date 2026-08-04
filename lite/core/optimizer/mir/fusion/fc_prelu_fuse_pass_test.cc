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
#include "lite/core/optimizer/mir/fusion/fc_prelu_fuser.h"
#include "lite/core/optimizer/mir/fusion/fusion_pass_test_util.h"

namespace paddle {
namespace lite {
namespace mir {

// fc → prelu folds into fc with activation_type=prelu.
std::vector<TestOpDesc> MakeFcPrelu() {
  TestOpDesc fc;
  fc.type = "fc";
  fc.inputs = {{"Input", {"x"}}, {"W", {"w"}}, {"Bias", {"b"}}};
  fc.outputs = {{"Out", {"fc_out"}}};
  fc.int_attrs = {{"in_num_col_dims", 1}};
  std::vector<TestOpDesc> ops{fc};
  ops.push_back({"prelu",
                 {{"X", {"fc_out"}}, {"Alpha", {"alpha"}}},
                 {{"Out", {"out"}}},
                 {},
                 {},
                 {},
                 {{"mode", "all"}},
                 {}});
  return ops;
}

TEST(FcPreluFuser, fuse_fc_prelu) {
  auto scope = std::make_shared<Scope>();
  // fc input must be rank-2.
  auto* x = scope->Var("x")->GetMutable<lite::Tensor>();
  x->Resize(DDim({2, 3}));
  x->mutable_data<float>()[0] = 1.0f;
  auto* w = scope->Var("w")->GetMutable<lite::Tensor>();
  w->Resize(DDim({3, 4}));
  for (int i = 0; i < 12; ++i) w->mutable_data<float>()[i] = 1.0f;
  auto* b = scope->Var("b")->GetMutable<lite::Tensor>();
  b->Resize(DDim({4}));
  b->mutable_data<float>()[0] = 0.0f;
  auto* alpha = scope->Var("alpha")->GetMutable<lite::Tensor>();
  alpha->Resize(DDim({4}));
  alpha->mutable_data<float>()[0] = 0.25f;

  auto graph = BuildGraph(MakeFcPrelu(), {"w", "b", "alpha"}, scope.get());
  ASSERT_EQ(CountOp(*graph, "fc"), 1);
  ASSERT_EQ(CountOp(*graph, "prelu"), 1);

  fusion::FcPreluFuser fuser("prelu");
  ASSERT_EQ(fuser(graph.get()), 1u);

  ASSERT_EQ(CountOp(*graph, "fc"), 1);
  ASSERT_EQ(CountOp(*graph, "prelu"), 0);
}

TEST(FcPreluFuser, skip_non_rank2_input) {
  auto scope = std::make_shared<Scope>();
  // rank-3 fc input fails the inputs_teller → no match.
  auto* x = scope->Var("x")->GetMutable<lite::Tensor>();
  x->Resize(DDim({1, 2, 3}));
  auto* w = scope->Var("w")->GetMutable<lite::Tensor>();
  w->Resize(DDim({3, 4}));
  auto* b = scope->Var("b")->GetMutable<lite::Tensor>();
  b->Resize(DDim({4}));
  auto* alpha = scope->Var("alpha")->GetMutable<lite::Tensor>();
  alpha->Resize(DDim({4}));

  auto graph = BuildGraph(MakeFcPrelu(), {"w", "b", "alpha"}, scope.get());
  fusion::FcPreluFuser fuser("prelu");
  ASSERT_EQ(fuser(graph.get()), 0u);
  ASSERT_EQ(CountOp(*graph, "prelu"), 1);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
