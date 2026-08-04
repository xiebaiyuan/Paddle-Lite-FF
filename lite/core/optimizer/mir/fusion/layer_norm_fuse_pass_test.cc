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
#include "lite/core/optimizer/mir/fusion/layer_norm_fuser.h"

namespace paddle {
namespace lite {
namespace mir {

// LayerNorm expansion:
//   x → reduce_mean → mean
//   x → sub(x, mean) → centered
//   centered → pow(2) → centered² → reduce_mean → var
//   var → add(eps) → sqrt → std
//   centered → div(std) → normalized
//   normalized → mul(scale) → add(bias) → out
std::vector<TestOpDesc> MakeLayerNormChain() {
  std::vector<TestOpDesc> ops;
  const std::map<std::string, int> kAxisM1{{"axis", -1}};
  ops.push_back({"reduce_mean",
                 {{"X", {"x"}}},
                 {{"Out", {"mean"}}},
                 {},
                 {},
                 {{"keep_dim", true}, {"reduce_all", false}},
                 {},
                 {{"dim", {2}}}});
  ops.push_back({"elementwise_sub",
                 {{"X", {"x"}}, {"Y", {"mean"}}},
                 {{"Out", {"centered"}}},
                 kAxisM1,
                 {},
                 {},
                 {}});
  ops.push_back({"elementwise_pow",
                 {{"X", {"centered"}}, {"Y", {"pow_y"}}},
                 {{"Out", {"centered_sq"}}},
                 kAxisM1,
                 {},
                 {},
                 {}});
  ops.push_back({"reduce_mean",
                 {{"X", {"centered_sq"}}},
                 {{"Out", {"var"}}},
                 {},
                 {},
                 {{"keep_dim", true}, {"reduce_all", false}},
                 {},
                 {{"dim", {2}}}});
  ops.push_back({"elementwise_add",
                 {{"X", {"var"}}, {"Y", {"eps"}}},
                 {{"Out", {"var_eps"}}},
                 kAxisM1,
                 {},
                 {},
                 {}});
  ops.push_back({"sqrt", {{"X", {"var_eps"}}}, {{"Out", {"std"}}}});
  ops.push_back({"elementwise_div",
                 {{"X", {"centered"}}, {"Y", {"std"}}},
                 {{"Out", {"normalized"}}},
                 kAxisM1,
                 {},
                 {},
                 {}});
  ops.push_back({"elementwise_mul",
                 {{"X", {"normalized"}}, {"Y", {"scale"}}},
                 {{"Out", {"scaled"}}},
                 kAxisM1,
                 {},
                 {},
                 {}});
  ops.push_back({"elementwise_add",
                 {{"X", {"scaled"}}, {"Y", {"bias"}}},
                 {{"Out", {"out"}}},
                 kAxisM1,
                 {},
                 {},
                 {}});
  return ops;
}

TEST(LayerNormFuser, fuse_chain) {
  auto scope = std::make_shared<Scope>();
  // pow exponent must be scalar 2.0 — provide a fill_constant producer.
  SetScalar(scope.get(), "pow_y", 2.0f);
  SetScalar(scope.get(), "eps", 1e-5f);
  auto* scale = scope->Var("scale")->GetMutable<lite::Tensor>();
  scale->Resize(DDim({2}));
  scale->mutable_data<float>()[0] = 1.0f;
  scale->mutable_data<float>()[1] = 1.0f;
  auto* bias = scope->Var("bias")->GetMutable<lite::Tensor>();
  bias->Resize(DDim({2}));
  bias->mutable_data<float>()[0] = 0.0f;
  bias->mutable_data<float>()[1] = 0.0f;

  std::vector<TestOpDesc> ops = MakeLayerNormChain();
  // pow_y / eps need persistable producers for the validation path; add
  // fill_constant producers with the right values.
  TestOpDesc fc_pow;
  fc_pow.type = "fill_constant";
  fc_pow.outputs = {{"Out", {"pow_y"}}};
  fc_pow.int_attrs = {{"dtype", 5}};
  fc_pow.float_attrs = {{"value", 2.0f}};
  fc_pow.bool_attrs = {{"force_cpu", false}};
  fc_pow.int_vector_attrs = {{"shape", {1}}};
  TestOpDesc fc_eps;
  fc_eps.type = "fill_constant";
  fc_eps.outputs = {{"Out", {"eps"}}};
  fc_eps.int_attrs = {{"dtype", 5}};
  fc_eps.float_attrs = {{"value", 1e-5f}};
  fc_eps.bool_attrs = {{"force_cpu", false}};
  fc_eps.int_vector_attrs = {{"shape", {1}}};
  ops.insert(ops.begin() + 2, fc_pow);
  ops.insert(ops.begin() + 5, fc_eps);

  auto graph = BuildGraph(ops, {"scale", "bias"}, scope.get());
  ASSERT_EQ(CountOp(*graph, "reduce_mean"), 2);

  fusion::LayerNormFuser fuser;
  ASSERT_EQ(fuser(graph.get()), 1u);

  // whole chain → single layer_norm
  ASSERT_EQ(CountOp(*graph, "layer_norm"), 1);
  ASSERT_EQ(CountOp(*graph, "reduce_mean"), 0);
  ASSERT_EQ(CountOp(*graph, "elementwise_sub"), 0);
  ASSERT_EQ(CountOp(*graph, "elementwise_pow"), 0);
  ASSERT_EQ(CountOp(*graph, "sqrt"), 0);
  ASSERT_EQ(CountOp(*graph, "elementwise_div"), 0);
  ASSERT_EQ(CountOp(*graph, "elementwise_mul"), 0);
  ASSERT_EQ(CountOp(*graph, "elementwise_add"), 0);
}

TEST(LayerNormFuser, skip_wrong_pow_exponent) {
  auto scope = std::make_shared<Scope>();
  auto* scale = scope->Var("scale")->GetMutable<lite::Tensor>();
  scale->Resize(DDim({2}));
  scale->mutable_data<float>()[0] = 1.0f;
  scale->mutable_data<float>()[1] = 1.0f;
  auto* bias = scope->Var("bias")->GetMutable<lite::Tensor>();
  bias->Resize(DDim({2}));
  bias->mutable_data<float>()[0] = 0.0f;
  bias->mutable_data<float>()[1] = 0.0f;

  std::vector<TestOpDesc> ops = MakeLayerNormChain();
  TestOpDesc fc_pow;
  fc_pow.type = "fill_constant";
  fc_pow.outputs = {{"Out", {"pow_y"}}};
  fc_pow.int_attrs = {{"dtype", 5}};
  fc_pow.float_attrs = {{"value", 3.0f}};  // wrong exponent
  fc_pow.bool_attrs = {{"force_cpu", false}};
  fc_pow.int_vector_attrs = {{"shape", {1}}};
  TestOpDesc fc_eps;
  fc_eps.type = "fill_constant";
  fc_eps.outputs = {{"Out", {"eps"}}};
  fc_eps.int_attrs = {{"dtype", 5}};
  fc_eps.float_attrs = {{"value", 1e-5f}};
  fc_eps.bool_attrs = {{"force_cpu", false}};
  fc_eps.int_vector_attrs = {{"shape", {1}}};
  ops.insert(ops.begin() + 2, fc_pow);
  ops.insert(ops.begin() + 5, fc_eps);

  auto graph = BuildGraph(ops, {"scale", "bias"}, scope.get());
  fusion::LayerNormFuser fuser;
  ASSERT_EQ(fuser(graph.get()), 0u);
  ASSERT_EQ(CountOp(*graph, "layer_norm"), 0);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
