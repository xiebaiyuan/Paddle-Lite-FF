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
__kernel void pixel_unshuffle(__read_only image2d_t input_image,
                              __write_only image2d_t output_image,
                              __private const int in_N,
                              __private const int in_C,
                              __private const int in_H,
                              __private const int in_W,
                              __private const int out_N,
                              __private const int out_C,
                              __private const int out_H,
                              __private const int out_W,
                              __private const int downscale_factor) {
  const int in_c4 = get_global_id(0);
  const int in_w = get_global_id(1);
  const int in_nh = get_global_id(2);

  int in_h = in_nh % in_H;
  int in_n = in_nh / in_H;

  CL_DTYPE4 res = (CL_DTYPE4)(0, 0, 0, 0);
  CL_DTYPE4 in;
  int in_c;
  int out_c;
  int offset;
  int offset_h;
  int offset_w;
  int out_w;
  int out_h;
  int out_nh;
  int2 out_pos;
  int2 in_pos;

  in_c = in_c4 * 4 + 0;
  out_c = in_c / (downscale_factor * downscale_factor);
  offset = in_c % (downscale_factor * downscale_factor);
  offset_h = offset / downscale_factor;
  offset_w = offset % downscale_factor;

  out_w = in_w * downscale_factor + offset_w;
  out_h = in_h * downscale_factor + offset_h;
  out_nh = in_n * out_H + out_h;

  out_pos.x = out_w + (out_c / 4) * in_W;
  out_pos.y = out_nh;

  in = READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, SAMPLER, out_pos);
  if (out_c % 4 == 0) {
    res.x = in.x;
  } else if (out_c % 4 == 1) {
    res.x = in.y;
  } else if (out_c % 4 == 2) {
    res.x = in.z;
  } else if (out_c % 4 == 3) {
    res.x = in.w;
  }

  in_c = in_c4 * 4 + 1;
  out_c = in_c / (downscale_factor * downscale_factor);
  offset = in_c % (downscale_factor * downscale_factor);
  offset_h = offset / downscale_factor;
  offset_w = offset % downscale_factor;

  out_w = in_w * downscale_factor + offset_w;
  out_h = in_h * downscale_factor + offset_h;
  out_nh = in_n * out_H + out_h;

  out_pos.x = out_w + (out_c / 4) * in_W;
  out_pos.y = out_nh;

  in = READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, SAMPLER, out_pos);
  if (out_c % 4 == 0) {
    res.y = in.x;
  } else if (out_c % 4 == 1) {
    res.y = in.y;
  } else if (out_c % 4 == 2) {
    res.y = in.z;
  } else if (out_c % 4 == 3) {
    res.y = in.w;
  }

  in_c = in_c4 * 4 + 2;
  out_c = in_c / (downscale_factor * downscale_factor);
  offset = in_c % (downscale_factor * downscale_factor);
  offset_h = offset / downscale_factor;
  offset_w = offset % downscale_factor;

  out_w = in_w * downscale_factor + offset_w;
  out_h = in_h * downscale_factor + offset_h;
  out_nh = in_n * out_H + out_h;

  out_pos.x = out_w + (out_c / 4) * in_W;
  out_pos.y = out_nh;

  in = READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, SAMPLER, out_pos);
  if (out_c % 4 == 0) {
    res.z = in.x;
  } else if (out_c % 4 == 1) {
    res.z = in.y;
  } else if (out_c % 4 == 2) {
    res.z = in.z;
  } else if (out_c % 4 == 3) {
    res.z = in.w;
  }

  in_c = in_c4 * 4 + 3;
  out_c = in_c / (downscale_factor * downscale_factor);
  offset = in_c % (downscale_factor * downscale_factor);
  offset_h = offset / downscale_factor;
  offset_w = offset % downscale_factor;

  out_w = in_w * downscale_factor + offset_w;
  out_h = in_h * downscale_factor + offset_h;
  out_nh = in_n * out_H + out_h;

  out_pos.x = out_w + (out_c / 4) * in_W;
  out_pos.y = out_nh;

  in = READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, SAMPLER, out_pos);
  if (out_c % 4 == 0) {
    res.w = in.x;
  } else if (out_c % 4 == 1) {
    res.w = in.y;
  } else if (out_c % 4 == 2) {
    res.w = in.z;
  } else if (out_c % 4 == 3) {
    res.w = in.w;
  }

  in_pos.x = in_c4 * (in_W / downscale_factor) + in_w;
  in_pos.y = in_nh;
  if (in_pos.x < out_W * ((out_C + 3) / 4) && in_pos.y < out_H * out_N) {
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, in_pos, res);
  }
}
