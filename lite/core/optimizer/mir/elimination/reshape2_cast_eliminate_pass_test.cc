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
#include <set>
#include <string>
#include <vector>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/optimizer/mir/elimination/reshape2_cast_eliminate_pass.h"
#include "lite/core/optimizer/mir/fusion/fusion_pass_test_util.h"
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

namespace {

// int32 <-> int64 dtype codes used in the pass source.
constexpr int kInt32 = 2;
constexpr int kInt64 = 3;

// dtype attribute block for a cast op.
std::map<std::string, int> CastDtype(int in_dtype, int out_dtype) {
  return {{"in_dtype", in_dtype}, {"out_dtype", out_dtype}};
}

}  // namespace

// Phase 1: drop an identity reshape2(shape=[N]) whose X is a shape-vector
// tensor produced by `shape` (N elements, N<=4, no dynamic dims). The
// consumer (a cast below) is rewired to read the shape tensor directly.
TEST(Reshape2CastEliminatePass, drop_identity_reshape2_on_shape_tensor) {
  std::vector<TestOpDesc> ops;
  ops.push_back({"shape", {{"Input", {"feat"}}}, {{"Out", {"shape_v"}}}});
  ops.push_back({"reshape2",
                 {{"X", {"shape_v"}}},
                 {{"Out", {"reshaped"}}, {"XShape", {"reshaped_xshape"}}},
                 {},
                 {},
                 {{"inplace", false}},
                 {},
                 {{"shape", std::vector<int>{4}}}});
  ops.push_back({"cast",
                 {{"X", {"reshaped"}}},
                 {{"Out", {"casted"}}},
                 CastDtype(kInt32, kInt64),
                 {},
                 {},
                 {},
                 {}});
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(ops, {}, scope.get());

  ASSERT_EQ(CountOp(*graph, "reshape2"), 1);
  Reshape2CastEliminatePass pass;
  pass.Apply(graph);

  // identity reshape2 dropped; the cast now consumes the shape tensor.
  ASSERT_EQ(CountOp(*graph, "reshape2"), 0);
  ASSERT_EQ(CountOp(*graph, "cast"), 1);
}

// Phase 1 negative: a reshape2 whose X is a real feature tensor (produced by
// conv2d) must NOT be eliminated even when shape=[4].
TEST(Reshape2CastEliminatePass, keep_reshape2_on_feature_tensor) {
  std::vector<TestOpDesc> ops;
  ops.push_back({"conv2d",
                 {{"Input", {"feat"}}, {"Filter", {"filter"}}},
                 {{"Output", {"conv_out"}}},
                 {{"groups", 1}},
                 {},
                 {},
                 {},
                 {{"paddings", std::vector<int>{0, 0}},
                  {"dilations", std::vector<int>{1, 1}},
                  {"strides", std::vector<int>{1, 1}}}});
  ops.push_back({"reshape2",
                 {{"X", {"conv_out"}}},
                 {{"Out", {"reshaped"}}, {"XShape", {"reshaped_xshape"}}},
                 {},
                 {},
                 {{"inplace", false}},
                 {},
                 {{"shape", std::vector<int>{4}}}});
  ops.push_back({"cast",
                 {{"X", {"reshaped"}}},
                 {{"Out", {"casted"}}},
                 CastDtype(kInt32, kInt64),
                 {},
                 {},
                 {},
                 {}});
  // conv2d requires a Filter tensor in scope for AttachImpl.
  auto scope = std::make_shared<Scope>();
  auto* f = scope->Var("filter")->GetMutable<lite::Tensor>();
  f->Resize(DDim({4, 4, 3, 3}));
  auto graph = BuildGraph(ops, {}, scope.get());

  ASSERT_EQ(CountOp(*graph, "reshape2"), 1);
  Reshape2CastEliminatePass pass;
  pass.Apply(graph);

  ASSERT_EQ(CountOp(*graph, "reshape2"), 1);
  ASSERT_EQ(CountOp(*graph, "cast"), 1);
}

// Phase 2: drop the cast(A->B) -> cast(B->A) dtype round-trip. The widening
// cast's output feeds exactly one consumer (the narrowing cast), whose output
// is consumed by the reshape2 below.
TEST(Reshape2CastEliminatePass, drop_dtype_roundtrip) {
  std::vector<TestOpDesc> ops;
  ops.push_back({"shape", {{"Input", {"feat"}}}, {{"Out", {"shape_v"}}}});
  ops.push_back({"cast",
                 {{"X", {"shape_v"}}},
                 {{"Out", {"widened"}}},
                 CastDtype(kInt32, kInt64),
                 {},
                 {},
                 {},
                 {}});
  ops.push_back({"cast",
                 {{"X", {"widened"}}},
                 {{"Out", {"narrowed"}}},
                 CastDtype(kInt64, kInt32),
                 {},
                 {},
                 {},
                 {}});
  ops.push_back({"reshape2",
                 {{"X", {"narrowed"}}},
                 {{"Out", {"reshaped"}}, {"XShape", {"reshaped_xshape"}}},
                 {},
                 {},
                 {{"inplace", false}},
                 {},
                 {{"shape", std::vector<int>{4}}}});
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(ops, {}, scope.get());

  ASSERT_EQ(CountOp(*graph, "cast"), 2);
  Reshape2CastEliminatePass pass;
  pass.Apply(graph);

  // Both casts removed; the reshape2 consumes the original int32 shape tensor.
  ASSERT_EQ(CountOp(*graph, "cast"), 0);
  ASSERT_EQ(CountOp(*graph, "reshape2"), 1);
}

// Phase 2 negative: a one-way cast (A->B, no reverse cast) is never removed.
TEST(Reshape2CastEliminatePass, keep_one_way_cast) {
  std::vector<TestOpDesc> ops;
  ops.push_back({"shape", {{"Input", {"feat"}}}, {{"Out", {"shape_v"}}}});
  ops.push_back({"cast",
                 {{"X", {"shape_v"}}},
                 {{"Out", {"widened"}}},
                 CastDtype(kInt32, kInt64),
                 {},
                 {},
                 {},
                 {}});
  ops.push_back({"reshape2",
                 {{"X", {"widened"}}},
                 {{"Out", {"reshaped"}}, {"XShape", {"reshaped_xshape"}}},
                 {},
                 {},
                 {{"inplace", false}},
                 {},
                 {{"shape", std::vector<int>{4}}}});
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(ops, {}, scope.get());

  Reshape2CastEliminatePass pass;
  pass.Apply(graph);

  ASSERT_EQ(CountOp(*graph, "cast"), 1);
  ASSERT_EQ(CountOp(*graph, "reshape2"), 1);
}

// Phase 3: drop an identity cast (in_dtype == out_dtype).
TEST(Reshape2CastEliminatePass, drop_identity_cast) {
  std::vector<TestOpDesc> ops;
  ops.push_back({"shape", {{"Input", {"feat"}}}, {{"Out", {"shape_v"}}}});
  ops.push_back({"cast",
                 {{"X", {"shape_v"}}},
                 {{"Out", {"casted"}}},
                 CastDtype(kInt32, kInt32),
                 {},
                 {},
                 {},
                 {}});
  ops.push_back({"reshape2",
                 {{"X", {"casted"}}},
                 {{"Out", {"reshaped"}}, {"XShape", {"reshaped_xshape"}}},
                 {},
                 {},
                 {{"inplace", false}},
                 {},
                 {{"shape", std::vector<int>{4}}}});
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(ops, {}, scope.get());

  ASSERT_EQ(CountOp(*graph, "cast"), 1);
  Reshape2CastEliminatePass pass;
  pass.Apply(graph);

  ASSERT_EQ(CountOp(*graph, "cast"), 0);
  ASSERT_EQ(CountOp(*graph, "reshape2"), 1);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
