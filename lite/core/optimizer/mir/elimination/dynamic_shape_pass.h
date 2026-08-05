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

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "lite/core/optimizer/mir/pass_v2.h"

namespace paddle {
namespace lite {
namespace mir {

/*
 * DynamicShapePass — restore dynamic input shape from x2paddle residue.
 *
 * Problem: x2paddle converts ONNX models with a fixed input shape dict
 * (e.g. --input_shape_dict '{"x":[1,3,48,320]}'). While the ONNX input is
 * truly dynamic (dim_param), the exported Paddle model hardcodes the input
 * dims AND the width-derived values inside reshape2 shape attrs (e.g.
 * reshape2 shape=[1,120,40] where 40 = 320/8, 120 = 320*3/8, 15 = 320/64).
 * At inference with a different width W, the reshape2's ValidateShape fails
 * (capacity != input_size) → error=-23.
 *
 * The width-derived values are NOT arbitrary constants: they are fixed
 * fractions of the training width (W/8, W*3/8, W/64), all related by
 * factors of 8. This pass detects them by asking: "if I replace this value
 * with -1, does the reshape become valid (single unknown dim, element count
 * conserved)?" If yes, the value is width-derived — make it dynamic (-1).
 *
 * Safety: a value is only replaced when BOTH hold:
 *  1. It is a positive multiple of 8 (the downsampling stride family), AND
 *  2. Replacing it with -1 yields a shape whose known dims' product divides
 *     the input's total element count (checked via the same arithmetic as
 *     reshape_op.cc ValidateShape).
 * A legitimately fixed dimension (e.g. a hardcoded channel count 192 that
 * does NOT divide the input size when replaced by -1) is left untouched.
 */
class DynamicShapePass : public PassV2 {
 public:
  DynamicShapePass() : PassV2(Pass::Kind::kProgramWise) {}
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;
  bool ShouldOnlyApplyOnce() const override { return true; }
  int OptimizationLevel() const override { return PassV2::kLevelBasic; }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
