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
#include "lite/core/optimizer/mir/fusion/fusion_pass_test_util.h"
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/optimizer/mir/support_0_dim_tensor_pass.h"

namespace paddle {
namespace lite {
namespace mir {

TEST(Support0DimTensor, fix_empty_shape) {
  // fill_constant with an empty shape vector gets shape=[1] appended.
  std::vector<TestOpDesc> ops;
  ops.push_back({"fill_constant",
                 {},
                 {{"Out", {"out"}}},
                 {{"dtype", 5}},
                 {{"value", 1.0f}},
                 {{"force_cpu", false}},
                 {},
                 {{"shape", {}}}});
  auto scope = std::make_shared<Scope>();
  auto* out = scope->Var("out")->GetMutable<lite::Tensor>();
  out->Resize(DDim({1}));

  auto graph = BuildGraph(ops, {}, scope.get());
  ASSERT_EQ(CountOp(*graph, "fill_constant"), 1);

  Support0DimTensor pass;
  pass.Apply(graph);

  // the shape attr must now be non-empty (filled with 1).
  bool ok = false;
  for (auto& node : graph->nodes()) {
    if (node.IsStmt() && node.stmt()->op_info()->Type() == "fill_constant") {
      auto* info = node.stmt()->op_info();
      ok = info->HasAttr("shape") &&
           info->GetAttr<std::vector<int32_t>>("shape").size() >= 1;
    }
  }
  ASSERT_TRUE(ok);
}

TEST(Support0DimTensor, keep_non_empty_shape) {
  // fill_constant with shape=[1, 3] must be untouched.
  std::vector<TestOpDesc> ops;
  ops.push_back({"fill_constant",
                 {},
                 {{"Out", {"out"}}},
                 {{"dtype", 5}},
                 {{"value", 1.0f}},
                 {{"force_cpu", false}},
                 {},
                 {{"shape", {1, 3}}}});
  auto scope = std::make_shared<Scope>();
  auto* out = scope->Var("out")->GetMutable<lite::Tensor>();
  out->Resize(DDim({1, 3}));

  auto graph = BuildGraph(ops, {}, scope.get());
  Support0DimTensor pass;
  pass.Apply(graph);

  bool ok = false;
  for (auto& node : graph->nodes()) {
    if (node.IsStmt() && node.stmt()->op_info()->Type() == "fill_constant") {
      auto* info = node.stmt()->op_info();
      auto shape = info->GetAttr<std::vector<int32_t>>("shape");
      ok = shape.size() == 2 && shape[0] == 1 && shape[1] == 3;
    }
  }
  ASSERT_TRUE(ok);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
