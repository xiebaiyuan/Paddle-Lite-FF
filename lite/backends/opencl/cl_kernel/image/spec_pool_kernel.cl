/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <cl_common.h>

__kernel void pool(__read_only image2d_t input,
                   __write_only image2d_t output,
                   __private const int in_height,
                   __private const int in_width,
                   __private const int out_height,
                   __private const int out_width,
                   __private const int ksize_h,
                   __private const int ksize_w,
                   __private const int stride_h,
                   __private const int stride_w,
                   __private const int pad_top,
                   __private const int pad_left,
                   __private const int adaptive) { //cyh
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);
  const int out_n = out_nh / out_height;
  const int out_h = out_nh % out_height;

  int start_h = out_h * stride_h - pad_top;
  int end_h = min(start_h + ksize_h, in_height);
  start_h = max(start_h, 0);

  int start_w = out_w * stride_w - pad_left;
  int end_w = min(start_w + ksize_w, in_width);
  start_w = max(start_w, 0);
  if(adaptive==1){
     start_h = out_h * in_height/out_height;
     end_h  = (out_h+1) * in_height/out_height;
     start_w = out_w * in_width /out_width;
     end_w = (out_w+1) * in_width /out_width;
  }

  const int pos_in_x = out_c * in_width;
  const int pos_in_y = out_n * in_height;
  const int pos_out_x = mad24(out_c, out_width, out_w);

#ifdef POOL_AVG

  CL_DTYPE4 res = (CL_DTYPE4)(0.0f);
  int div;
#ifdef EXCLUSIVE
  div = (end_h - start_h) * (end_w - start_w);
#else
  div = ksize_w * ksize_h;
#endif // EXCLUSIVE

#ifdef GLOBAL
  // pool_avg_global: force to use fp32 to avoid the loss of accuracy
  float4 res_f32 = 0.f;
  for (int y = start_h; y < end_h; ++y) {
    for (int x = start_w; x < end_w; ++x) {
      res_f32 += read_imagef(input, SAMPLER, (int2)(pos_in_x + x, pos_in_y + y));
    }
  }
  res_f32 /= (float)div;
#ifdef CL_DTYPE_half
  res = convert_half4(res_f32);
#else
  res = res_f32;
#endif

#else
  // pool_avg: use default precision
  for (int y = start_h; y < end_h; ++y) {
    for (int x = start_w; x < end_w; ++x) {
      res += READ_IMG_TYPE(
          CL_DTYPE_CHAR, input, SAMPLER, (int2)(pos_in_x + x, pos_in_y + y));
    }
  }
  res /= (CL_DTYPE)div;
#endif // GLOBAL

#else

  // POOL_MAX
  CL_DTYPE4 res = (CL_DTYPE4)(-FLT_MAX);
  for (int y = start_h; y < end_h; ++y) {
    for (int x = start_w; x < end_w; ++x) {
      CL_DTYPE4 tmp = READ_IMG_TYPE(
          CL_DTYPE_CHAR, input, SAMPLER, (int2)(pos_in_x + x, pos_in_y + y));
      res = max(res, tmp);
    }
  }

#endif // POOL_AVG

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(pos_out_x, out_nh), res);
}