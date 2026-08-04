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
#include "lite/core/optimizer/mir/elimination/remove_scale1_pass.h"
#include "lite/core/optimizer/mir/fusion/fusion_pass_test_util.h"
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

namespace {

// relu(x) -> scale(scale, bias) -> sigmoid -> y
// The scale's input arg (relu_out) has exactly one producer (relu) and its
// output arg (scale_out) has exactly one consumer (sigmoid), which is the
// topology RemoveScale1Pass requires.
std::vector<TestOpDesc> MakeScaleBetweenOps(float scale, float bias) {
  std::vector<TestOpDesc> ops;
  ops.push_back({"relu", {{"X", {"x"}}}, {{"Out", {"relu_out"}}}});
  TestOpDesc s;
  s.type = "scale";
  s.inputs = {{"X", {"relu_out"}}};
  s.outputs = {{"Out", {"scale_out"}}};
  s.float_attrs = {{"scale", scale}, {"bias", bias}};
  s.bool_attrs = {{"bias_after_scale", true}};
  ops.push_back(s);
  ops.push_back({"sigmoid", {{"X", {"scale_out"}}}, {{"Out", {"y"}}}});
  return ops;
}

}  // namespace

// A plain scale(scale≈1, bias≈0) sitting between two ops is redundant and
// must be removed: sigmoid's input is rewired to relu_out.
TEST(RemoveScale1Pass, remove_identity_scale) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeScaleBetweenOps(1.0f, 0.0f), {}, scope.get());

  ASSERT_EQ(CountOp(*graph, "scale"), 1);

  RemoveScale1Pass pass;
  pass.Apply(graph);

  ASSERT_EQ(CountOp(*graph, "scale"), 0);
  ASSERT_EQ(CountOp(*graph, "relu"), 1);
  ASSERT_EQ(CountOp(*graph, "sigmoid"), 1);
}

// scale != 1 must be preserved.
TEST(RemoveScale1Pass, keep_scale_not_one) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeScaleBetweenOps(2.0f, 0.0f), {}, scope.get());

  RemoveScale1Pass pass;
  pass.Apply(graph);

  ASSERT_EQ(CountOp(*graph, "scale"), 1);
}

// A scale with fused relu (fuse_relu=true) must be preserved, even though
// scale==1 and bias==0. The pass reads the fuse_relu attr directly from the
// op desc.
TEST(RemoveScale1Pass, keep_scale_with_fused_relu) {
  auto scope = std::make_shared<Scope>();
  std::vector<TestOpDesc> ops = MakeScaleBetweenOps(1.0f, 0.0f);
  ops[1].bool_attrs = {{"bias_after_scale", true}, {"fuse_relu", true}};
  auto graph = BuildGraph(ops, {}, scope.get());

  RemoveScale1Pass pass;
  pass.Apply(graph);

  ASSERT_EQ(CountOp(*graph, "scale"), 1);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
