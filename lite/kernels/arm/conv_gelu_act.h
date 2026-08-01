// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/backends/arm/math/activation.h"
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

// Helpers for conv kernels to fuse a trailing gelu activation.
//
// The ARM conv math kernels dispatch activations through a flag_act bit
// pattern / asm switch (relu/relu6/leaky_relu/hard_swish only) and LOG(FATAL)
// on anything else. gelu needs erf, which none of those fast paths can
// express, so instead of extending every asm path we:
//
//   1. call ActivateGeluUnset() before the conv math runs — it clears
//      has_active so the math kernel only adds bias (safe for every path),
//   2. call ApplyGeluIfFused() on the output tensor afterwards, applying the
//      exact same scalar approximation as the standalone gelu kernel.
//
// The fused result is bit-identical to conv + standalone gelu.
inline bool IsGeluFused(const operators::ActivationParam& act_param) {
  return act_param.has_active &&
         act_param.active_type == lite_api::ActivationType::kGelu;
}

// Clear has_active so the underlying conv math (asm flag_act dispatch) treats
// the conv as activation-free. gelu is re-applied afterwards.
inline void UnsetGeluForConvMath(operators::ActivationParam* act_param) {
  if (act_param != nullptr && IsGeluFused(*act_param)) {
    act_param->has_active = false;
  }
}

// Apply gelu to the (already bias-added) conv output. `count` is the total
// number of float elements. Checked against active_type rather than
// has_active because UnsetGeluForConvMath() clears has_active before the
// conv math runs.
inline void ApplyGeluIfFused(const operators::ActivationParam& act_param,
                             float* data,
                             int64_t count) {
  if (act_param.active_type != lite_api::ActivationType::kGelu) {
    return;
  }
  for (int64_t i = 0; i < count; ++i) {
    data[i] = paddle::lite::arm::math::act_gelu_scalar(
        data[i], act_param.gelu_approximate);
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
