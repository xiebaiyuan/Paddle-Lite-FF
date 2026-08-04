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
#include "lite/core/optimizer/mir/fusion/fill_range_fuser.h"
#include "lite/core/optimizer/mir/fusion/fusion_pass_test_util.h"

namespace paddle {
namespace lite {
namespace mir {

// 3 × fill_constant → range: the fill_constant ops are eliminated and the
// range's Start/End/Step tensors get the constant values written in place.
std::vector<TestOpDesc> MakeFillRange() {
  std::vector<TestOpDesc> ops;
  for (const auto& name : {"start", "end", "step"}) {
    ops.push_back({"fill_constant",
                   {},
                   {{"Out", {name}}},
                   {{"dtype", 5}},  // FP32
                   {{"value", 0.0f}},
                   {{"force_cpu", false}},
                   {},
                   {{"shape", {1}}}});
  }
  ops.push_back({"range",
                 {{"Start", {"start"}}, {"End", {"end"}}, {"Step", {"step"}}},
                 {{"Out", {"range_out"}}}});
  return ops;
}

TEST(FillRangeFuser, fuse_fill_range) {
  auto scope = std::make_shared<Scope>();
  // materialize Start/End/Step tensors so AttachImpl succeeds.
  for (const auto& name : {"start", "end", "step"}) {
    auto* t = scope->Var(name)->GetMutable<lite::Tensor>();
    t->Resize(DDim({1}));
    t->mutable_data<float>()[0] = 0.0f;
  }
  auto* out = scope->Var("range_out")->GetMutable<lite::Tensor>();
  out->Resize(DDim({1}));

  auto graph = BuildGraph(MakeFillRange(), {}, scope.get());
  ASSERT_EQ(CountOp(*graph, "fill_constant"), 3);
  ASSERT_EQ(CountOp(*graph, "range"), 1);

  fusion::FillRangeFuser fuser;
  // GenOpDesc reads fill_constant 'value' attrs and writes them into the
  // range's input tensors, then deletes the fill_constants.
  ASSERT_EQ(fuser(graph.get()), 1u);
  ASSERT_EQ(CountOp(*graph, "fill_constant"), 0);
  ASSERT_EQ(CountOp(*graph, "range"), 1);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
