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
#include "lite/core/optimizer/mir/fusion/interpolate_fuser.h"

namespace paddle {
namespace lite {
namespace mir {

// type1: x → shape → slice(axes=[0], starts=[2], ends=[4]) → cast →
//        elementwise_mul(fill_constant) → interpolate, with x → interpolate.
std::vector<TestOpDesc> MakeInterpolateType1(const std::string& interp_type) {
  std::vector<TestOpDesc> ops;
  ops.push_back({"shape", {{"Input", {"x"}}}, {{"Out", {"shape_out"}}}});
  ops.push_back({"slice",
                 {{"Input", {"shape_out"}}},
                 {{"Out", {"slice_out"}}},
                 {},
                 {},
                 {},
                 {},
                 {{"axes", {0}}, {"starts", {2}}, {"ends", {4}}}});
  ops.push_back({"cast",
                 {{"X", {"slice_out"}}},
                 {{"Out", {"cast_out"}}},
                 {{"in_dtype", 2}, {"out_dtype", 5}},
                 {},
                 {},
                 {},
                 {}});
  ops.push_back({"fill_constant",
                 {},
                 {{"Out", {"fc_out"}}},
                 {{"dtype", 5}},
                 {{"value", 2.0f}},
                 {{"force_cpu", false}},
                 {},
                 {{"shape", {1}}}});
  ops.push_back({"elementwise_mul",
                 {{"X", {"cast_out"}}, {"Y", {"fc_out"}}},
                 {{"Out", {"mul_out"}}},
                 {{"axis", -1}},
                 {},
                 {},
                 {}});
  TestOpDesc interp;
  interp.type = interp_type;
  interp.inputs = {{"X", {"x"}}, {"OutSize", {"mul_out"}}};
  interp.outputs = {{"Out", {"out"}}};
  interp.float_attrs = {{"scale", 0.0f}};
  interp.bool_attrs = {{"align_corners", false}};
  interp.str_attrs = {{"interp_method", "bilinear"}};
  interp.int_attrs = {{"align_mode", 1}};
  ops.push_back(interp);
  return ops;
}

// type2: x → shape → slice → cast → scale → interpolate, with x → interpolate.
std::vector<TestOpDesc> MakeInterpolateType2(const std::string& interp_type) {
  std::vector<TestOpDesc> ops;
  ops.push_back({"shape", {{"Input", {"x"}}}, {{"Out", {"shape_out"}}}});
  ops.push_back({"slice",
                 {{"Input", {"shape_out"}}},
                 {{"Out", {"slice_out"}}},
                 {},
                 {},
                 {},
                 {},
                 {{"axes", {0}}, {"starts", {2}}, {"ends", {4}}}});
  ops.push_back({"cast",
                 {{"X", {"slice_out"}}},
                 {{"Out", {"cast_out"}}},
                 {{"in_dtype", 2}, {"out_dtype", 5}},
                 {},
                 {},
                 {},
                 {}});
  ops.push_back({"scale",
                 {{"X", {"cast_out"}}},
                 {{"Out", {"scale_out"}}},
                 {},
                 {{"scale", 2.0f}, {"bias", 0.0f}},
                 {{"bias_after_scale", true}},
                 {},
                 {}});
  TestOpDesc interp;
  interp.type = interp_type;
  interp.inputs = {{"X", {"x"}}, {"OutSize", {"scale_out"}}};
  interp.outputs = {{"Out", {"out"}}};
  interp.float_attrs = {{"scale", 0.0f}};
  interp.bool_attrs = {{"align_corners", false}};
  interp.str_attrs = {{"interp_method", "bilinear"}};
  interp.int_attrs = {{"align_mode", 1}};
  ops.push_back(interp);
  return ops;
}

TEST(InterpolateFuser, fuse_type1_bilinear) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeInterpolateType1("bilinear_interp"), {},
                          scope.get());
  ASSERT_EQ(CountOp(*graph, "shape"), 1);
  ASSERT_EQ(CountOp(*graph, "slice"), 1);
  ASSERT_EQ(CountOp(*graph, "bilinear_interp"), 1);

  fusion::InterpolateFuser fuser("bilinear_interp");
  ASSERT_EQ(fuser(graph.get()), 1u);

  // chain collapsed; interpolate keeps the scale from fill_constant.
  ASSERT_EQ(CountOp(*graph, "shape"), 0);
  ASSERT_EQ(CountOp(*graph, "slice"), 0);
  ASSERT_EQ(CountOp(*graph, "cast"), 0);
  ASSERT_EQ(CountOp(*graph, "elementwise_mul"), 0);
  ASSERT_EQ(CountOp(*graph, "bilinear_interp"), 1);
}

TEST(InterpolateFuser, fuse_type1_nearest) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeInterpolateType1("nearest_interp"), {},
                          scope.get());
  fusion::InterpolateFuser fuser("nearest_interp");
  ASSERT_EQ(fuser(graph.get()), 1u);
  ASSERT_EQ(CountOp(*graph, "shape"), 0);
  ASSERT_EQ(CountOp(*graph, "nearest_interp"), 1);
}

TEST(InterpolateFuser, fuse_type2_bilinear) {
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(MakeInterpolateType2("bilinear_interp"), {},
                          scope.get());
  fusion::InterpolateFuser2 fuser2("bilinear_interp");
  ASSERT_EQ(fuser2(graph.get()), 1u);
  ASSERT_EQ(CountOp(*graph, "shape"), 0);
  ASSERT_EQ(CountOp(*graph, "scale"), 0);
  ASSERT_EQ(CountOp(*graph, "bilinear_interp"), 1);
}

TEST(InterpolateFuser, skip_wrong_slice_attr) {
  auto scope = std::make_shared<Scope>();
  std::vector<TestOpDesc> ops = MakeInterpolateType1("bilinear_interp");
  ops[1].int_vector_attrs = {{"axes", {1}}, {"starts", {2}}, {"ends", {4}}};
  auto graph = BuildGraph(ops, {}, scope.get());

  fusion::InterpolateFuser fuser("bilinear_interp");
  ASSERT_EQ(fuser(graph.get()), 0u);
  ASSERT_EQ(CountOp(*graph, "shape"), 1);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
