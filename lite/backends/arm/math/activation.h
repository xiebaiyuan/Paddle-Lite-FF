// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <arm_neon.h>
#include <cmath>
#include <string>

namespace paddle {
namespace lite {
namespace arm {
namespace math {

// Forward declaration: exp_ps is defined in funcs.h (included by
// conv_block_utils.h etc.). We cannot include funcs.h here — it includes
// activation.h, creating a cycle — but erff_approx_v4 below needs it.
inline float32x4_t exp_ps(float32x4_t x);

// ============================================================
// Scalar gelu helpers, shared between the standalone gelu kernel
// (activation_extra_compute.cc -> act_gelu<float>) and the fused
// conv+gelu path (conv_block_utils.h write_to_output_c4_fp32).
//
// Keeping them inline in this header lets the conv kernels reuse the
// exact same erf/tanh approximation without re-instantiating the
// templated act_gelu<float> (which would otherwise be "explicit
// specialization after instantiation" once conv_block_utils.h is
// included from multiple translation units).
// ============================================================

static const float tansig_table[201] = {
    0.000000f, 0.039979f, 0.079830f, 0.119427f, 0.158649f, 0.197375f, 0.235496f,
    0.272905f, 0.309507f, 0.345214f, 0.379949f, 0.413644f, 0.446244f, 0.477700f,
    0.507977f, 0.537050f, 0.564900f, 0.591519f, 0.616909f, 0.641077f, 0.664037f,
    0.685809f, 0.706419f, 0.725897f, 0.744277f, 0.761594f, 0.777888f, 0.793199f,
    0.807569f, 0.821040f, 0.833655f, 0.845456f, 0.856485f, 0.866784f, 0.876393f,
    0.885352f, 0.893698f, 0.901468f, 0.908698f, 0.915420f, 0.921669f, 0.927473f,
    0.932862f, 0.937863f, 0.942503f, 0.946806f, 0.950795f, 0.954492f, 0.957917f,
    0.961090f, 0.964028f, 0.966747f, 0.969265f, 0.971594f, 0.973749f, 0.975743f,
    0.977587f, 0.979293f, 0.980869f, 0.982327f, 0.983675f, 0.984921f, 0.986072f,
    0.987136f, 0.988119f, 0.989027f, 0.989867f, 0.990642f, 0.991359f, 0.992020f,
    0.992631f, 0.993196f, 0.993718f, 0.994199f, 0.994644f, 0.995055f, 0.995434f,
    0.995784f, 0.996108f, 0.996407f, 0.996682f, 0.996937f, 0.997172f, 0.997389f,
    0.997590f, 0.997775f, 0.997946f, 0.998104f, 0.998249f, 0.998384f, 0.998508f,
    0.998623f, 0.998728f, 0.998826f, 0.998916f, 0.999000f, 0.999076f, 0.999147f,
    0.999213f, 0.999273f, 0.999329f, 0.999381f, 0.999428f, 0.999472f, 0.999513f,
    0.999550f, 0.999585f, 0.999617f, 0.999646f, 0.999673f, 0.999699f, 0.999722f,
    0.999743f, 0.999763f, 0.999781f, 0.999798f, 0.999813f, 0.999828f, 0.999841f,
    0.999853f, 0.999865f, 0.999875f, 0.999885f, 0.999893f, 0.999902f, 0.999909f,
    0.999916f, 0.999923f, 0.999929f, 0.999934f, 0.999939f, 0.999944f, 0.999948f,
    0.999952f, 0.999956f, 0.999959f, 0.999962f, 0.999965f, 0.999968f, 0.999970f,
    0.999973f, 0.999975f, 0.999977f, 0.999978f, 0.999980f, 0.999982f, 0.999983f,
    0.999984f, 0.999986f, 0.999987f, 0.999988f, 0.999989f, 0.999990f, 0.999990f,
    0.999991f, 0.999992f, 0.999992f, 0.999993f, 0.999994f, 0.999994f, 0.999994f,
    0.999995f, 0.999995f, 0.999996f, 0.999996f, 0.999996f, 0.999997f, 0.999997f,
    0.999997f, 0.999997f, 0.999997f, 0.999998f, 0.999998f, 0.999998f, 0.999998f,
    0.999998f, 0.999998f, 0.999999f, 0.999999f, 0.999999f, 0.999999f, 0.999999f,
    0.999999f, 0.999999f, 0.999999f, 0.999999f, 0.999999f, 0.999999f, 0.999999f,
    0.999999f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
    1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
};

inline float tansig_approx(float x) {
  if (x >= 8) return 1;
  if (x <= -8) return -1;

  float sign = x < 0 ? -1 : 1;
  x = x * sign;
  int i = static_cast<int>(floor(0.5f + 25 * x));
  x -= 0.04f * i;
  float y = tansig_table[i];
  float dy = 1 - y * y;
  y = y + x * dy * (1 - y * x);
  return sign * y;
}

// Abramowitz-Stegun erf approximation used by act_gelu<float> (max error
// < 1 ulp). Same coefficients as the NEON erff_approx_v4.
inline float erff_approx(float a) {
  float r, s, t, u;

  t = fabsf(a);
  s = a * a;
  if (t > 0.927734375f) {  // 475/512
    // maximum error 0.99527 ulp
    r = fmaf(
        -1.72853470e-5f, t, 3.83197126e-4f);  // -0x1.220000p-16,0x1.91cfb2p-12
    u = fmaf(
        -3.88396438e-3f, t, 2.42546219e-2f);  // -0x1.fd1438p-9, 0x1.8d6342p-6
    r = fmaf(r, s, u);
    r = fmaf(r, t, -1.06777877e-1f);  // -0x1.b55cb8p-4
    r = fmaf(r, t, -6.34846687e-1f);  // -0x1.450aa0p-1
    r = fmaf(r, t, -1.28717512e-1f);  // -0x1.079d0cp-3
    r = fmaf(r, t, -t);
    r = 1.0f - expf(r);
    r = copysignf(r, a);
  } else {
    // maximum error 0.98929 ulp
    r = -5.96761703e-4f;              // -0x1.38e000p-11
    r = fmaf(r, s, 4.99119423e-3f);   //  0x1.471a58p-8
    r = fmaf(r, s, -2.67681349e-2f);  // -0x1.b691b2p-6
    r = fmaf(r, s, 1.12819925e-1f);   //  0x1.ce1c44p-4
    r = fmaf(r, s, -3.76125336e-1f);  // -0x1.812700p-2
    r = fmaf(r, s, 1.28379166e-1f);   //  0x1.06eba8p-3
    r = fmaf(r, a, a);
  }
  return r;
}

// Elementwise gelu matching act_gelu<float>'s scalar (remain) path:
//   approximate=true  -> 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
//   approximate=false -> 0.5*x*(1+erf(x/sqrt(2)))
inline float act_gelu_scalar(float x, bool approximate) {
  if (approximate) {
    const float pi = std::atan(1) * 4;
    const float sqrt_2_div_pi = std::sqrt(2 / pi);
    return 0.5f * x *
           (1 +
            tansig_approx(sqrt_2_div_pi * (x + 0.044715f * std::pow(x, 3))));
  }
  const float sqrt_2_rec = 1.0f / std::sqrt(2.0f);
  return 0.5f * x * (1 + erff_approx(x * sqrt_2_rec));
}

// ============================================================
// NEON (float32x4_t) gelu helpers — same approximation as the scalar
// versions above, but processing 4 lanes at once. The standalone gelu
// kernel (act_gelu<float>) uses these on the SIMD main path; exposing
// them here lets the fused conv+gelu path (write_to_output_c4_fp32's
// kGelu branch) reuse the exact same NEON math instead of falling back
// to per-element scalar calls (which measurably regressed conv latency).
// ============================================================

inline float32x4_t tansig_approx_v4(float32x4_t x) {
  float32x4_t v_8 = vdupq_n_f32(8.f);
  float32x4_t v_8_ = vdupq_n_f32(-8.f);
  float32x4_t v_1_ = vdupq_n_f32(-1.f);
  float32x4_t vzero = vdupq_n_f32(0.f);
  float32x4_t vones = vdupq_n_f32(1.f);

  uint32x4_t v_comge8 = vcgeq_f32(x, v_8);
  uint32x4_t v_comle8_ = vcleq_f32(x, v_8_);
  uint32x4_t v_comlt0 = vcltq_f32(x, vzero);

  float32x4_t vsign = vbslq_f32(v_comlt0, v_1_, vones);
  x = vmulq_f32(x, vsign);
  float32x4_t v_x_25 = vmlaq_f32(vdupq_n_f32(0.5f), x, vdupq_n_f32(25.f));

  int32x4_t tab_i_v = vcvtq_s32_f32(v_x_25);
  int tab_i_0 = vgetq_lane_s32(tab_i_v, 0);
  int tab_i_1 = vgetq_lane_s32(tab_i_v, 1);
  int tab_i_2 = vgetq_lane_s32(tab_i_v, 2);
  int tab_i_3 = vgetq_lane_s32(tab_i_v, 3);

  float32x4_t tab_data_v;
  tab_data_v = vsetq_lane_f32(tansig_table[tab_i_0], tab_data_v, 0);
  tab_data_v = vsetq_lane_f32(tansig_table[tab_i_1], tab_data_v, 1);
  tab_data_v = vsetq_lane_f32(tansig_table[tab_i_2], tab_data_v, 2);
  tab_data_v = vsetq_lane_f32(tansig_table[tab_i_3], tab_data_v, 3);

  float32x4_t tab_i_f = vcvtq_f32_s32(tab_i_v);
  x = vmlsq_f32(x, vdupq_n_f32(0.04), tab_i_f);

  float32x4_t v_dy = vmlsq_f32(vones, tab_data_v, tab_data_v);  // dy
  float32x4_t v_res = vmlsq_f32(vones, tab_data_v, x);
  v_res = vmulq_f32(v_dy, v_res);
  v_res = vmlaq_f32(tab_data_v, x, v_res);
  v_res = vmulq_f32(vsign, v_res);

  v_res = vbslq_f32(v_comge8, vones, v_res);
  v_res = vbslq_f32(v_comle8_, v_1_, v_res);
  return v_res;
}

#define c_erff_r0_p0 -1.72853470e-5f
#define c_erff_r0_p1 3.83197126e-4f
#define c_erff_r0_p2 -3.88396438e-3f
#define c_erff_r0_p3 2.42546219e-2f
#define c_erff_r0_p4 -1.06777877e-1f
#define c_erff_r0_p5 -6.34846687e-1f
#define c_erff_r0_p6 -1.28717512e-1f
#define c_erff_r1_p0 -5.96761703e-4f
#define c_erff_r1_p1 4.99119423e-3f
#define c_erff_r1_p2 -2.67681349e-2f
#define c_erff_r1_p3 1.12819925e-1f
#define c_erff_r1_p4 -3.76125336e-1f
#define c_erff_r1_p5 1.28379166e-1f
#define c_erff_threshold 0.927734375f

inline float32x4_t erff_approx_v4(float32x4_t a) {
  float32x4_t coef0 = vdupq_n_f32(c_erff_r0_p0);
  float32x4_t coef1 = vdupq_n_f32(c_erff_r0_p1);
  float32x4_t coef2 = vdupq_n_f32(c_erff_r0_p2);
  float32x4_t coef3 = vdupq_n_f32(c_erff_r0_p3);
  float32x4_t coef4 = vdupq_n_f32(c_erff_r0_p4);
  float32x4_t coef5 = vdupq_n_f32(c_erff_r0_p5);
  float32x4_t coef6 = vdupq_n_f32(c_erff_r0_p6);

  float32x4_t vzero_4 = vdupq_n_f32(0.f);
  float32x4_t vones_4 = vdupq_n_f32(1.f);
  float32x4_t vmask_4 = vdupq_n_f32(c_erff_threshold);
  float32x4_t r0, r1, s, t, u, t_;

  // r0 (t > 0.927734375f)
  t = vabsq_f32(a);
  s = vmulq_f32(a, a);
  t_ = vsubq_f32(vzero_4, t);
  r0 = vmlaq_f32(coef1, coef0, t);
  u = vmlaq_f32(coef3, coef2, t);
  r0 = vmlaq_f32(u, r0, s);
  r0 = vmlaq_f32(coef4, r0, t);
  r0 = vmlaq_f32(coef5, r0, t);
  r0 = vmlaq_f32(coef6, r0, t);
  r0 = vmlaq_f32(t_, r0, t);
  r0 = vsubq_f32(vones_4, exp_ps(r0));

  uint32x4_t vm_gt0 = vcgtq_f32(a, vzero_4);
  r0 = vbslq_f32(vm_gt0, r0, vsubq_f32(vzero_4, r0));

  // r1 (t <= 0.927734375f)
  coef0 = vdupq_n_f32(c_erff_r1_p0);
  coef1 = vdupq_n_f32(c_erff_r1_p1);
  coef2 = vdupq_n_f32(c_erff_r1_p2);
  coef3 = vdupq_n_f32(c_erff_r1_p3);
  coef4 = vdupq_n_f32(c_erff_r1_p4);
  coef5 = vdupq_n_f32(c_erff_r1_p5);

  r1 = coef0;
  r1 = vmlaq_f32(coef1, r1, s);
  r1 = vmlaq_f32(coef2, r1, s);
  r1 = vmlaq_f32(coef3, r1, s);
  r1 = vmlaq_f32(coef4, r1, s);
  r1 = vmlaq_f32(coef5, r1, s);
  r1 = vmlaq_f32(a, r1, a);

  // choose r0 or r1
  uint32x4_t v_mask_re = vcltq_f32(t, vmask_4);
  r0 = vbslq_f32(v_mask_re, r1, r0);
  return r0;
}

// Elementwise gelu on 4 lanes, mirroring act_gelu<float>'s SIMD main
// path exactly (approximate -> tanh-based, exact -> erf-based) so the
// fused conv result is bit-identical to the standalone kernel.
inline float32x4_t act_gelu_v4(float32x4_t x, bool approximate) {
  if (approximate) {
    const float pi = std::atan(1) * 4;
    const float sqrt_2_div_pi = std::sqrt(2 / pi);
    float32x4_t sqrt_2_div_pi_v4 = vdupq_n_f32(sqrt_2_div_pi);
    float32x4_t coeff_v4 = vdupq_n_f32(0.044715f);
    float32x4_t vones_4 = vdupq_n_f32(1.f);
    float32x4_t v05_4 = vdupq_n_f32(0.5f);

    float32x4_t vx_pow = vmulq_f32(x, x);
    vx_pow = vmulq_f32(vx_pow, x);
    vx_pow = vmlaq_f32(x, vx_pow, coeff_v4);
    vx_pow = vmulq_f32(vx_pow, sqrt_2_div_pi_v4);

    float32x4_t v_res = tansig_approx_v4(vx_pow);
    v_res = vaddq_f32(v_res, vones_4);
    v_res = vmulq_f32(v_res, x);
    v_res = vmulq_f32(v_res, v05_4);
    return v_res;
  }
  const float sqrt_2_rec = 1.0f / std::sqrt(2.0f);
  float32x4_t v_sqrt2_rec = vdupq_n_f32(sqrt_2_rec);
  float32x4_t vones_4 = vdupq_n_f32(1.f);
  float32x4_t vdata_0_5 = vdupq_n_f32(0.5f);

  float32x4_t v_tmp = vmulq_f32(v_sqrt2_rec, x);
  float32x4_t v_erf = erff_approx_v4(v_tmp);
  float32x4_t res = vmulq_f32(vdata_0_5, x);
  v_erf = vaddq_f32(vones_4, v_erf);
  res = vmulq_f32(res, v_erf);
  return res;
}

template <typename T>
void act_relu(const T* din, T* dout, int size, int threads);

template <typename T>
void act_relu_neg(
    const T* din, T* dout, int size, float negative_slope, int threads);

template <typename T>
void act_clipped_relu(const T* din, T* dout, int size, float coef, int threads);

template <typename T>
void act_prelu(const T* din,
               T* dout,
               int outer_size,
               int channel_size,
               int inner_size,
               std::string mode,
               const float* alpha_data,
               int threads);

template <typename T>
void act_sigmoid(const T* din, T* dout, int size, int threads);

template <typename T>
void act_tanh(const T* din, T* dout, int size, int threads);

template <typename T>
void act_swish(const T* din, T* dout, int size, float coef, int threads);

template <typename T>
void act_log(const T* din, T* dout, int size, int threads);

template <typename T>
void act_exp(const T* din, T* dout, int size, int threads);

template <typename T>
void act_floor(const T* din, T* dout, int size, int threads);

template <typename T>
void act_hard_sigmoid(const T* din,
                      T* dout,
                      const int64_t size,
                      const float slope,
                      const float offset,
                      int threads);

template <typename T>
void act_rsqrt(const T* din, T* dout, int size, int threads);

template <typename T>
void act_sqrt(const T* din, T* dout, int size, int threads);

template <typename T>
void act_square(const T* din, T* dout, int size, int threads);

template <typename T>
void act_hard_swish(const T* din,
                    T* dout,
                    int size,
                    float threshold,
                    float scale,
                    float offset,
                    int threads);
template <typename T>
void act_reciprocal(const T* din, T* dout, int size, int threads);

template <typename T>
void act_abs(const T* din, T* dout, int size, int threads);

template <typename T>
void act_thresholded_relu(
    const T* din, T* dout, int size, float threshold, int threads);

template <typename T>
void act_elu(const T* din, T* dout, int size, float alpha, int threads);

template <typename T>
void act_gelu(const T* din, T* dout, int size, bool approximate, int threads);

template <typename T>
void erf(const T* din, T* dout, int size, int threads);

template <typename T>
void sign(const T* din, T* dout, int size, int threads);

template <typename T>
void softplus(const T* din, T* dout, int size, float beta, int threads);

template <typename T>
void mish(const T* din, T* dout, int size, float threshold);

template <typename T>
void act_silu(const T* din, T* dout, int size, int threads);

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
