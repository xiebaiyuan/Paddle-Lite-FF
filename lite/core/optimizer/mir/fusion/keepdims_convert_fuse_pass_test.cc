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
#include "lite/core/optimizer/mir/fusion/keepdims_convert_fuser.h"

namespace paddle {
namespace lite {
namespace mir {

// reduce_sum(keep_dim=false) → out is replaced by a keepdims path + reshape.
std::vector<TestOpDesc> MakeKeepdimsOp(const std::string& op_type) {
  TestOpDesc op;
  op.type = op_type;
  op.inputs = {{"X", {"x"}}};
  op.outputs = {{"Out", {"out"}}};
  if (op_type == "reduce_sum") {
    op.int_vector_attrs = {{"dim", {1}}};
    op.bool_attrs = {{"keep_dim", false}, {"reduce_all", false}};
  } else if (op_type == "arg_max") {
    op.bool_attrs = {{"keepdims", false}};
  }
  return {op};
}

TEST(KeepdimsConvertFuser, convert_reduce_sum) {
  auto scope = std::make_shared<Scope>();
  auto* x = scope->Var("x")->GetMutable<lite::Tensor>();
  x->Resize(DDim({1, 2, 2}));
  x->mutable_data<float>()[0] = 1.0f;
  auto* out = scope->Var("out")->GetMutable<lite::Tensor>();
  out->Resize(DDim({1, 2}));

  auto graph = BuildGraph(MakeKeepdimsOp("reduce_sum"), {}, scope.get());
  fusion::KeepdimsConvertFuser fuser("reduce_sum");
  // InsertNewNode requires exactly one outlink from the op.
  ASSERT_EQ(fuser(graph.get()), 1u);

  // keep_dim set to true, reshape inserted after.
  ASSERT_EQ(CountOp(*graph, "reshape"), 1);
}

TEST(KeepdimsConvertFuser, skip_keepdim_true) {
  auto scope = std::make_shared<Scope>();
  auto* x = scope->Var("x")->GetMutable<lite::Tensor>();
  x->Resize(DDim({1, 2, 2}));
  auto* out = scope->Var("out")->GetMutable<lite::Tensor>();
  out->Resize(DDim({1, 2}));

  std::vector<TestOpDesc> ops = MakeKeepdimsOp("reduce_sum");
  ops[0].bool_attrs = {{"keep_dim", true}, {"reduce_all", false}};
  auto graph = BuildGraph(ops, {}, scope.get());

  fusion::KeepdimsConvertFuser fuser("reduce_sum");
  ASSERT_EQ(fuser(graph.get()), 0u);
  ASSERT_EQ(CountOp(*graph, "reshape"), 0);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
