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
#include "lite/core/optimizer/mir/fusion/shuffle_channel_fuser.h"

namespace paddle {
namespace lite {
namespace mir {

// reshape2(N,C*G,H,W → N,G,C,H,W) → transpose2(0,2,1,3) →
// reshape2(N,C,H,W) folds into shuffle_channel(group=C of reshape1).
std::vector<TestOpDesc> MakeShuffleChannel(const std::string& reshape_type,
                                           const std::string& transpose_type) {
  std::vector<TestOpDesc> ops;
  TestOpDesc r1;
  r1.type = reshape_type;
  r1.inputs = {{"X", {"x"}}};
  r1.outputs = {{"Out", {"y1"}}};
  r1.int_vector_attrs = {{"shape", {1, 2, 3, 2, 2}}};  // group=2
  if (reshape_type == "reshape2") r1.outputs["XShape"] = {"xs1"};
  ops.push_back(r1);

  TestOpDesc t;
  t.type = transpose_type;
  t.inputs = {{"X", {"y1"}}};
  t.outputs = {{"Out", {"y2"}}};
  t.int_vector_attrs = {{"axis", {0, 2, 1, 3, 4}}};
  if (transpose_type == "transpose2") t.outputs["XShape"] = {"xs2"};
  ops.push_back(t);

  TestOpDesc r2;
  r2.type = reshape_type;
  r2.inputs = {{"X", {"y2"}}};
  r2.outputs = {{"Out", {"out"}}};
  r2.int_vector_attrs = {{"shape", {1, 6, 2, 2}}};
  if (reshape_type == "reshape2") r2.outputs["XShape"] = {"xs3"};
  ops.push_back(r2);
  return ops;
}

TEST(ShuffleChannelFuser, fuse_reshape2_transpose2) {
  auto scope = std::make_shared<Scope>();
  auto* x = scope->Var("x")->GetMutable<lite::Tensor>();
  x->Resize(DDim({1, 6, 2, 2}));
  auto* y1 = scope->Var("y1")->GetMutable<lite::Tensor>();
  y1->Resize(DDim({1, 2, 3, 2, 2}));
  auto* y2 = scope->Var("y2")->GetMutable<lite::Tensor>();
  y2->Resize(DDim({1, 3, 2, 2, 2}));
  auto* out = scope->Var("out")->GetMutable<lite::Tensor>();
  out->Resize(DDim({1, 6, 2, 2}));
  for (const auto& n : {"xs1", "xs2", "xs3"}) {
    auto* xs = scope->Var(n)->GetMutable<lite::Tensor>();
    xs->Resize(DDim({6}));
  }

  auto graph = BuildGraph(MakeShuffleChannel("reshape2", "transpose2"), {},
                          scope.get());
  ASSERT_EQ(CountOp(*graph, "reshape2"), 2);
  ASSERT_EQ(CountOp(*graph, "transpose2"), 1);

  fusion::ShuffleChannelFuser fuser("reshape2", "transpose2");
  ASSERT_EQ(fuser(graph.get()), 1u);

  ASSERT_EQ(CountOp(*graph, "reshape2"), 0);
  ASSERT_EQ(CountOp(*graph, "transpose2"), 0);
  ASSERT_EQ(CountOp(*graph, "shuffle_channel"), 1);
}

TEST(ShuffleChannelFuser, fuse_reshape_transpose) {
  auto scope = std::make_shared<Scope>();
  auto* x = scope->Var("x")->GetMutable<lite::Tensor>();
  x->Resize(DDim({1, 6, 2, 2}));
  auto* y1 = scope->Var("y1")->GetMutable<lite::Tensor>();
  y1->Resize(DDim({1, 2, 3, 2, 2}));
  auto* y2 = scope->Var("y2")->GetMutable<lite::Tensor>();
  y2->Resize(DDim({1, 3, 2, 2, 2}));
  auto* out = scope->Var("out")->GetMutable<lite::Tensor>();
  out->Resize(DDim({1, 6, 2, 2}));

  auto graph = BuildGraph(MakeShuffleChannel("reshape", "transpose"), {},
                          scope.get());
  fusion::ShuffleChannelFuser fuser("reshape", "transpose");
  ASSERT_EQ(fuser(graph.get()), 1u);
  ASSERT_EQ(CountOp(*graph, "shuffle_channel"), 1);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
