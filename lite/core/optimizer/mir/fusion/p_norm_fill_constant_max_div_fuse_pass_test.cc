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
#include "lite/core/optimizer/mir/fusion/p_norm_fill_constant_max_div_fuser.h"

namespace paddle {
namespace lite {
namespace mir {

// p_norm(2, keepdim) → fill_constant → elementwise_max → elementwise_div(x)
// folds into norm.
std::vector<TestOpDesc> MakePNormChain() {
  std::vector<TestOpDesc> ops;
  ops.push_back({"p_norm",
                 {{"X", {"x"}}},
                 {{"Out", {"pnorm_out"}}},
                 {{"axis", -1}},
                 {{"porder", 2.0f}, {"epsilon", 0.0f}},
                 {{"keepdim", true}, {"asvector", false}},
                 {},
                 {}});
  ops.push_back({"fill_constant",
                 {},
                 {{"Out", {"fc_out"}}},
                 {{"dtype", 5}},
                 {{"value", 1e-5f}},
                 {{"force_cpu", false}},
                 {},
                 {{"shape", {1}}}});
  ops.push_back({"elementwise_max",
                 {{"X", {"fc_out"}}, {"Y", {"pnorm_out"}}},
                 {{"Out", {"max_out"}}},
                 {{"axis", -1}},
                 {},
                 {},
                 {}});
  ops.push_back({"elementwise_div",
                 {{"X", {"x"}}, {"Y", {"max_out"}}},
                 {{"Out", {"out"}}},
                 {{"axis", -1}},
                 {},
                 {},
                 {}});
  return ops;
}

TEST(PNormFillConstantMaxDivFuser, fuse_to_norm) {
  auto scope = std::make_shared<Scope>();
  auto* x = scope->Var("x")->GetMutable<lite::Tensor>();
  x->Resize(DDim({2, 3}));
  auto* fc = scope->Var("fc_out")->GetMutable<lite::Tensor>();
  fc->Resize(DDim({1}));
  fc->mutable_data<float>()[0] = 1e-5f;

  auto graph = BuildGraph(MakePNormChain(), {}, scope.get());
  ASSERT_EQ(CountOp(*graph, "p_norm"), 1);
  ASSERT_EQ(CountOp(*graph, "elementwise_max"), 1);
  ASSERT_EQ(CountOp(*graph, "elementwise_div"), 1);

  fusion::PNormFillConstantMaxDivFuser fuser;
  ASSERT_EQ(fuser(graph.get()), 1u);

  ASSERT_EQ(CountOp(*graph, "p_norm"), 0);
  ASSERT_EQ(CountOp(*graph, "elementwise_max"), 0);
  ASSERT_EQ(CountOp(*graph, "elementwise_div"), 0);
  ASSERT_EQ(CountOp(*graph, "norm"), 1);
}

TEST(PNormFillConstantMaxDivFuser, skip_wrong_porder) {
  auto scope = std::make_shared<Scope>();
  auto* x = scope->Var("x")->GetMutable<lite::Tensor>();
  x->Resize(DDim({2, 3}));
  auto* fc = scope->Var("fc_out")->GetMutable<lite::Tensor>();
  fc->Resize(DDim({1}));
  fc->mutable_data<float>()[0] = 1e-5f;

  std::vector<TestOpDesc> ops = MakePNormChain();
  ops[0].float_attrs = {{"porder", 1.0f}, {"epsilon", 0.0f}};  // L1 norm
  auto graph = BuildGraph(ops, {}, scope.get());
  fusion::PNormFillConstantMaxDivFuser fuser;
  ASSERT_EQ(fuser(graph.get()), 0u);
  ASSERT_EQ(CountOp(*graph, "p_norm"), 1);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
