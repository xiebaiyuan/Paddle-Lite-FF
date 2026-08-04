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
#include "lite/core/optimizer/mir/adaptive_1x1_pool2d_convert_global_pass.h"
#include "lite/core/optimizer/mir/fusion/fusion_pass_test_util.h"
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

// pool2d with adaptive=true + ksize=[1,1] + global_pooling=false is rewritten
// to global_pooling=true + adaptive=false.
std::vector<TestOpDesc> MakePool(bool adaptive, bool global_pooling) {
  TestOpDesc pool;
  pool.type = "pool2d";
  pool.inputs = {{"X", {"x"}}};
  pool.outputs = {{"Out", {"out"}}};
  pool.bool_attrs = {{"adaptive", adaptive},
                     {"global_pooling", global_pooling}};
  pool.str_attrs = {{"pooling_type", "max"}};
  pool.int_vector_attrs = {{"ksize", {1, 1}},
                           {"strides", {1, 1}},
                           {"paddings", {0, 0}}};
  std::vector<TestOpDesc> ops{pool};
  return ops;
}

TEST(Adaptive1x1Pool2dConvertGlobalPass, convert_adaptive_1x1) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakePool(true, false), {}, scope.get());
  ASSERT_EQ(CountOp(*graph, "pool2d"), 1);

  Adaptive1x1Pool2dConvertGlobalPass pass;
  pass.Apply(graph);

  // pool2d now carries global_pooling=true, adaptive=false.
  bool ok = false;
  for (auto& node : graph->nodes()) {
    if (node.IsStmt() && node.stmt()->op_info()->Type() == "pool2d") {
      auto* info = node.stmt()->op_info();
      ok = info->GetAttr<bool>("global_pooling") == true &&
           info->GetAttr<bool>("adaptive") == false;
    }
  }
  ASSERT_TRUE(ok);
}

TEST(Adaptive1x1Pool2dConvertGlobalPass, skip_adaptive_false) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakePool(false, false), {}, scope.get());

  Adaptive1x1Pool2dConvertGlobalPass pass;
  pass.Apply(graph);

  // adaptive=false → no rewrite; global_pooling stays false.
  bool ok = false;
  for (auto& node : graph->nodes()) {
    if (node.IsStmt() && node.stmt()->op_info()->Type() == "pool2d") {
      auto* info = node.stmt()->op_info();
      ok = info->GetAttr<bool>("global_pooling") == false &&
           info->GetAttr<bool>("adaptive") == false;
    }
  }
  ASSERT_TRUE(ok);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
