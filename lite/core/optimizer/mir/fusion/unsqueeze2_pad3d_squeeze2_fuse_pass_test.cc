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
#include "lite/core/optimizer/mir/fusion/unsqueeze2_pad3d_squeeze2_fuse.h"

namespace paddle {
namespace lite {
namespace mir {

// unsqueeze2 → pad3d(paddings[4]=paddings[5]=0) → squeeze2 folds into pad2d.
std::vector<TestOpDesc> MakeUnsqueezePad3dSqueeze() {
  std::vector<TestOpDesc> ops;
  TestOpDesc u;
  u.type = "unsqueeze2";
  u.inputs = {{"X", {"x"}}};
  u.outputs = {{"Out", {"u_out"}}, {"XShape", {"u_xs"}}};
  u.int_vector_attrs = {{"axes", {1}}};
  ops.push_back(u);

  TestOpDesc p;
  p.type = "pad3d";
  p.inputs = {{"X", {"u_out"}}};
  p.outputs = {{"Out", {"p_out"}}};
  p.float_attrs = {{"value", 0.0f}};
  p.str_attrs = {{"mode", "constant"}, {"data_format", "NCDHW"}};
  p.int_vector_attrs = {{"paddings", {1, 1, 1, 1, 0, 0}}};
  ops.push_back(p);

  TestOpDesc s;
  s.type = "squeeze2";
  s.inputs = {{"X", {"p_out"}}};
  s.outputs = {{"Out", {"out"}}, {"XShape", {"s_xs"}}};
  s.int_vector_attrs = {{"axes", {1}}};
  ops.push_back(s);
  return ops;
}

TEST(Unsqueeze2Pad3dSqueeze2Fuser, fuse_to_pad2d) {
  auto scope = std::make_shared<Scope>();
  auto* x = scope->Var("x")->GetMutable<lite::Tensor>();
  x->Resize(DDim({1, 2, 2, 2}));
  auto* u_out = scope->Var("u_out")->GetMutable<lite::Tensor>();
  u_out->Resize(DDim({1, 1, 2, 2, 2}));
  auto* p_out = scope->Var("p_out")->GetMutable<lite::Tensor>();
  p_out->Resize(DDim({1, 1, 4, 4, 4}));
  auto* out = scope->Var("out")->GetMutable<lite::Tensor>();
  out->Resize(DDim({1, 4, 4, 4}));
  for (const auto& n : {"u_xs", "s_xs"}) {
    auto* xs = scope->Var(n)->GetMutable<lite::Tensor>();
    xs->Resize(DDim({5}));
  }

  auto graph = BuildGraph(MakeUnsqueezePad3dSqueeze(), {}, scope.get());
  ASSERT_EQ(CountOp(*graph, "unsqueeze2"), 1);
  ASSERT_EQ(CountOp(*graph, "pad3d"), 1);
  ASSERT_EQ(CountOp(*graph, "squeeze2"), 1);

  fusion::Unsqueeze2Pad3dSqueeze2Fuser fuser("unsqueeze2", "pad3d",
                                             "squeeze2");
  ASSERT_EQ(fuser(graph.get()), 1u);

  ASSERT_EQ(CountOp(*graph, "pad3d"), 0);
  ASSERT_EQ(CountOp(*graph, "pad2d"), 1);
}

TEST(Unsqueeze2Pad3dSqueeze2Fuser, skip_bad_paddings) {
  auto scope = std::make_shared<Scope>();
  auto* x = scope->Var("x")->GetMutable<lite::Tensor>();
  x->Resize(DDim({1, 2, 2, 2}));
  auto* u_out = scope->Var("u_out")->GetMutable<lite::Tensor>();
  u_out->Resize(DDim({1, 1, 2, 2, 2}));
  auto* p_out = scope->Var("p_out")->GetMutable<lite::Tensor>();
  p_out->Resize(DDim({1, 1, 4, 4, 4}));
  auto* out = scope->Var("out")->GetMutable<lite::Tensor>();
  out->Resize(DDim({1, 4, 4, 4}));
  for (const auto& n : {"u_xs", "s_xs"}) {
    auto* xs = scope->Var(n)->GetMutable<lite::Tensor>();
    xs->Resize(DDim({5}));
  }

  std::vector<TestOpDesc> ops = MakeUnsqueezePad3dSqueeze();
  ops[1].int_vector_attrs = {{"paddings", {1, 1, 1, 1, 1, 1}}};  // depth pad
  auto graph = BuildGraph(ops, {}, scope.get());

  fusion::Unsqueeze2Pad3dSqueeze2Fuser fuser("unsqueeze2", "pad3d",
                                             "squeeze2");
  ASSERT_EQ(fuser(graph.get()), 0u);
  ASSERT_EQ(CountOp(*graph, "pad3d"), 1);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
