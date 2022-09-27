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

__kernel void sigmoid(__read_only image2d_t input,
                      __write_only image2d_t output,
                      __private const float threshold,
                      __private const float scale) {
  const int x = get_global_id(0);  // image_width
  const int y = get_global_id(1);  // image_height

  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x, y));
  CL_DTYPE4 out;

  out.x = (CL_DTYPE)(1.0f / (1.0f + pow(2.71828182f, -1.0f * (float)(in.x))));
  out.y = (CL_DTYPE)(1.0f / (1.0f + pow(2.71828182f, -1.0f * (float)(in.y))));
  out.z = (CL_DTYPE)(1.0f / (1.0f + pow(2.71828182f, -1.0f * (float)(in.z))));
  out.w = (CL_DTYPE)(1.0f / (1.0f + pow(2.71828182f, -1.0f * (float)(in.w))));

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(x, y), out);
}

/***************************************************************************
 * For case: Axis N/H/W or Axis C that all input channels is aligned: Start
 ***************************************************************************/
#define CHECK_IDX                                  \
  int c_blk_idx = get_global_id(0);                \
  int w_idx = get_global_id(1);                    \
  int nh_idx = get_global_id(2);                   \
  if (c_blk_idx >= output_shape.y ||               \
      w_idx >= output_shape.w ||                   \
      nh_idx >= output_shape.x * output_shape.z) { \
    return;                                        \
  }                                                \
  CL_DTYPE4 result;

// axis = 1
#define DOConcat2InputAxis1                                            \
  int boundary0 = input_shape0.y; /* C_blk0 */                         \
  int boundary1 = boundary0 + input_shape1.y; /* C_blk0 + C_blk1 */    \
  int2 input_pos;                                                      \
  input_pos.y = nh_idx;                                                \
  if (c_blk_idx < boundary0) {                                         \
    input_pos.x = c_blk_idx * input_shape0.w + w_idx;                  \
    result = READ_IMG_TYPE(CL_DTYPE_CHAR, input0, SAMPLER, input_pos); \
  } else if (c_blk_idx < boundary1) {                                  \
    input_pos.x = (c_blk_idx - boundary0) * input_shape1.w + w_idx;    \
    result = READ_IMG_TYPE(CL_DTYPE_CHAR, input1, SAMPLER, input_pos); \
  }

#define DOConcat3InputAxis1                                            \
  DOConcat2InputAxis1;                                                 \
  int boundary2 = boundary1 + input_shape2.y;                          \
  if (c_blk_idx >= boundary1 && c_blk_idx < boundary2) {               \
    input_pos.x = (c_blk_idx - boundary1) * input_shape2.w + w_idx;    \
    result = READ_IMG_TYPE(CL_DTYPE_CHAR, input2, SAMPLER, input_pos); \
  }

#define DOConcat4InputAxis1                                            \
  DOConcat3InputAxis1;                                                 \
  int boundary3 = boundary2 + input_shape3.y;                          \
  if (c_blk_idx >= boundary2 && c_blk_idx < boundary3) {               \
    input_pos.x = (c_blk_idx - boundary2) * input_shape3.w + w_idx;    \
    result = READ_IMG_TYPE(CL_DTYPE_CHAR, input3, SAMPLER, input_pos); \
  }


// axis = 3
#define DOConcat2InputAxis3                                            \
  int boundary0 = input_shape0.w; /* W0 */                             \
  int boundary1 = boundary0 + input_shape1.w; /* W0 + W1 */            \
  int2 input_pos;                                                      \
  input_pos.y = nh_idx;                                                \
  if (w_idx < boundary0) {                                             \
    input_pos.x = c_blk_idx * input_shape0.w + w_idx;                  \
    result = READ_IMG_TYPE(CL_DTYPE_CHAR, input0, SAMPLER, input_pos); \
  } else if (w_idx < boundary1) {                                      \
    input_pos.x = c_blk_idx * input_shape1.w + w_idx - boundary0;      \
    result = READ_IMG_TYPE(CL_DTYPE_CHAR, input1, SAMPLER, input_pos); \
  }

#define DOConcat3InputAxis3                                            \
  DOConcat2InputAxis3;                                                 \
  int boundary2 = boundary1 + input_shape2.w;                          \
  if (w_idx >= boundary1 && w_idx < boundary2) {                       \
    input_pos.x = c_blk_idx * input_shape2.w + w_idx - boundary1;      \
    result = READ_IMG_TYPE(CL_DTYPE_CHAR, input2, SAMPLER, input_pos); \
  }



#define WRITE_IMG_DATA \
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(c_blk_idx * output_shape.w + w_idx, nh_idx), result);

#define CONCAT2(Inputnum, Axis)                                              \
  __kernel void Concat##Inputnum##Axis(__read_only image2d_t input0,         \
                                       __read_only image2d_t input1,         \
                                       __write_only image2d_t output,        \
                                       int4 input_shape0, int4 input_shape1, \
                                       int4 output_shape) {                  \
    CHECK_IDX                                                                \
    DOConcat##Inputnum##Axis                                                 \
    WRITE_IMG_DATA                                                           \
  }

#define CONCAT3(Inputnum, Axis)                                              \
  __kernel void Concat##Inputnum##Axis(__read_only image2d_t input0,         \
                                       __read_only image2d_t input1,         \
                                       __read_only image2d_t input2,         \
                                       __write_only image2d_t output,        \
                                       int4 input_shape0, int4 input_shape1, \
                                       int4 input_shape2,                    \
                                       int4 output_shape) {                  \
    CHECK_IDX                                                                \
    DOConcat##Inputnum##Axis                                                 \
    WRITE_IMG_DATA                                                           \
  }

#define CONCAT4(Inputnum, Axis)                                              \
  __kernel void Concat##Inputnum##Axis(__read_only image2d_t input0,         \
                                       __read_only image2d_t input1,         \
                                       __read_only image2d_t input2,         \
                                       __read_only image2d_t input3,         \
                                       __write_only image2d_t output,        \
                                       int4 input_shape0, int4 input_shape1, \
                                       int4 input_shape2, int4 input_shape3, \
                                       int4 output_shape) {                  \
    CHECK_IDX                                                                \
    DOConcat##Inputnum##Axis                                                 \
    WRITE_IMG_DATA                                                           \
  }

/*************************************************************************
 * For case: Axis N/H/W or Axis C that all input channels is aligned: End
 *************************************************************************/
// axis = 1
CONCAT3(3Input, Axis1)
CONCAT4(4Input, Axis1)
// axis = 3
CONCAT2(2Input, Axis3)
CONCAT3(3Input, Axis3)

__kernel void concatByCWith3Inputs(
                    __write_only image2d_t output_image,
                    __private const int output_tensor_c,
                    __private const int output_tensor_w,
                    __read_only image2d_t input0_image,
                    __private const int input0_tensor_c,
                    __read_only image2d_t input1_image,
                    __private const int input1_tensor_c,
                    __read_only image2d_t input2_image,
                    __private const int input2_tensor_c) {
  const int out_c = get_global_id(0);   // [0, (output_tensor_c + 3) / 4)
  const int out_w = get_global_id(1);   // [0, output_tensor_w)
  const int out_nh = get_global_id(2);  // [0, output_tensor_n * output_tensor_h)

  int2 output_pos;
  output_pos.x = out_c * output_tensor_w + out_w;
  output_pos.y = out_nh;
  CL_DTYPE4 output_data;

  for (int i = 0; i < 4; i++) {
    int c = out_c * 4 + i;
    if (c >= output_tensor_c) {
        break;
    }
    int c_in;
    CL_DTYPE4 input_data;
    if (c < input0_tensor_c) {
      c_in = c;
      int2 input_pos;
      input_pos.x = (c_in / 4) * output_tensor_w + out_w;
      input_pos.y = out_nh;
      input_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input0_image, SAMPLER, input_pos);
    } else if (c < input0_tensor_c + input1_tensor_c) {
      c_in = c - input0_tensor_c;
      int2 input_pos;
      input_pos.x = (c_in / 4) * output_tensor_w + out_w;
      input_pos.y = out_nh;
      input_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input1_image, SAMPLER, input_pos);
    } else {
      c_in = c - input0_tensor_c - input1_tensor_c;
      int2 input_pos;
      input_pos.x = (c_in / 4) * output_tensor_w + out_w;
      input_pos.y = out_nh;
      input_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input2_image, SAMPLER, input_pos);
    }
    int value_offset = c_in % 4;
    float value;
    if (value_offset == 0) {
      value = input_data.x;
    } else if (value_offset == 1) {
      value = input_data.y;
    } else if (value_offset == 2) {
      value = input_data.z;
    } else if (value_offset == 3) {
      value = input_data.w;
    }
    if (i == 0) {
      output_data.x = value;
    } else if (i == 1) {
      output_data.y = value;
    } else if (i == 2) {
      output_data.z = value;
    } else if (i == 3) {
      output_data.w = value;
    }
  }
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos, output_data);
}



inline void elt_fuse_func_wrapper(__read_only image2d_t second_input_image,
                                  const int2 pos,
                                  CL_DTYPE4 *value_p) {
  CL_DTYPE4 second_val =
      READ_IMG_TYPE(CL_DTYPE_CHAR, second_input_image, SAMPLER, pos);
  *value_p += second_val;
#ifdef ELT_ACT_FUSE
  *value_p = fmax(*value_p, (CL_DTYPE4)0);
#endif
}

__kernel void conv2d_1x1_opt(
    __private const int global_size_dim0,
    __private const int global_size_dim1,
    __private const int global_size_dim2,
    __read_only image2d_t input_image,
    __read_only image2d_t filter,
    __read_only image2d_t bias,
    __write_only image2d_t output_image,
    __private const int stride,
    __private const int offset,
    __private const int input_c_block,
    __private const int input_c_origin,
    __private const int dilation,
    __private const int input_width,  /* of one block */
    __private const int input_height, /* of one block */
    __private const int output_width,
    __private const int output_height,
    __private const int old_w,
    __read_only image2d_t prelu_alpha
#ifdef ELT_FUSE
    ,
    __read_only image2d_t second_input_image
#endif
    ) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);
  if (out_c >= global_size_dim0 || out_w >= global_size_dim1 ||
      out_nh >= global_size_dim2) {
    return;
  }

  int out_w0 = out_w;
  int out_w1 = out_w + global_size_dim1;
  int out_w2 = out_w + global_size_dim1 * 2;
  int out_w3 = out_w + global_size_dim1 * 3;

  int outpos_main = mul24(out_c, old_w);
  int2 output_pos0 = (int2)(outpos_main + out_w0, out_nh);
  int2 output_pos1 = (int2)(outpos_main + out_w1, out_nh);
  int2 output_pos2 = (int2)(outpos_main + out_w2, out_nh);
  int2 output_pos3 = (int2)(outpos_main + out_w3, out_nh);

  int2 stride_xy = (int2)(stride, stride);

  int2 ouput_pos_in_one_block0 = (int2)(out_w0, out_nh);
  int2 in_pos_in_one_block0 =
      ouput_pos_in_one_block0 * stride_xy + (int2)(offset, offset);

  int2 ouput_pos_in_one_block1 = (int2)(out_w1, out_nh);
  int2 in_pos_in_one_block1 =
      ouput_pos_in_one_block1 * stride_xy + (int2)(offset, offset);

  int2 ouput_pos_in_one_block2 = (int2)(out_w2, out_nh);
  int2 in_pos_in_one_block2 =
      ouput_pos_in_one_block2 * stride_xy + (int2)(offset, offset);

  int2 ouput_pos_in_one_block3 = (int2)(out_w3, out_nh);
  int2 in_pos_in_one_block3 =
      ouput_pos_in_one_block3 * stride_xy + (int2)(offset, offset);

#ifdef BIASE_CH
  CL_DTYPE4 output0 =
      READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, (int2)(out_c, 0));
  CL_DTYPE4 output1 = output0;
  CL_DTYPE4 output2 = output0;
  CL_DTYPE4 output3 = output0;
#else
  CL_DTYPE4 output0 = 0.0f;
  CL_DTYPE4 output1 = 0.0f;
  CL_DTYPE4 output2 = 0.0f;
  CL_DTYPE4 output3 = 0.0f;
#endif

  int max_w_bound = input_c_block * input_width;
  int burndary_index = input_c_block * 4 - input_c_origin;
  for (int i = 0; i < input_c_block; ++i) {
    // ------------0---------------
    int2 pos_in = (int2)(i * input_width + in_pos_in_one_block0.x,
                         in_pos_in_one_block0.y);
    CL_DTYPE4 input0 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, SAMPLER, pos_in);

    CL_DTYPE4 weight0 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, filter, SAMPLER, (int2)(out_c, i * 4 + 0));
    CL_DTYPE4 weight1 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, filter, SAMPLER, (int2)(out_c, i * 4 + 1));
    CL_DTYPE4 weight2 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, filter, SAMPLER, (int2)(out_c, i * 4 + 2));
    CL_DTYPE4 weight3 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, filter, SAMPLER, (int2)(out_c, i * 4 + 3));

    if ((max_w_bound - pos_in.x - 1) < input_width &&
        (max_w_bound - pos_in.x - 1) >= 0) {
      if (burndary_index == 0) {
        output0 = mad(input0.x, weight0, output0);
        output0 = mad(input0.y, weight1, output0);
        output0 = mad(input0.z, weight2, output0);
        output0 = mad(input0.w, weight3, output0);
      } else if (burndary_index == 1) {
        output0 = mad(input0.x, weight0, output0);
        output0 = mad(input0.y, weight1, output0);
        output0 = mad(input0.z, weight2, output0);
        output0 = mad(0.0f, weight3, output0);

      } else if (burndary_index == 2) {
        output0 = mad(input0.x, weight0, output0);
        output0 = mad(input0.y, weight1, output0);
        output0 = mad(0.0f, weight2, output0);
        output0 = mad(0.0f, weight3, output0);
      } else if (burndary_index == 3) {
        output0 = mad(input0.x, weight0, output0);
        output0 = mad(0.0f, weight1, output0);
        output0 = mad(0.0f, weight2, output0);
        output0 = mad(0.0f, weight3, output0);
      }
    } else {
      output0 = mad(input0.x, weight0, output0);
      output0 = mad(input0.y, weight1, output0);
      output0 = mad(input0.z, weight2, output0);
      output0 = mad(input0.w, weight3, output0);
    }

    // -------------1--------------
    pos_in = (int2)(i * input_width + in_pos_in_one_block1.x,
                    in_pos_in_one_block1.y);
    CL_DTYPE4 input1 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, SAMPLER, pos_in);

    if (abs(max_w_bound - pos_in.x) < input_width) {
      if (burndary_index == 0) {
        output1 = mad(input1.x, weight0, output1);
        output1 = mad(input1.y, weight1, output1);
        output1 = mad(input1.z, weight2, output1);
        output1 = mad(input1.w, weight3, output1);
      } else if (burndary_index == 1) {
        output1 = mad(input1.x, weight0, output1);
        output1 = mad(input1.y, weight1, output1);
        output1 = mad(input1.z, weight2, output1);
        output1 = mad(0.0f, weight3, output1);

      } else if (burndary_index == 2) {
        output1 = mad(input1.x, weight0, output1);
        output1 = mad(input1.y, weight1, output1);
        output1 = mad(0.0f, weight2, output1);
        output1 = mad(0.0f, weight3, output1);
      } else if (burndary_index == 3) {
        output1 = mad(input1.x, weight0, output1);
        output1 = mad(0.0f, weight1, output1);
        output1 = mad(0.0f, weight2, output1);
        output1 = mad(0.0f, weight3, output1);
      }
    } else {
      output1 = mad(input1.x, weight0, output1);
      output1 = mad(input1.y, weight1, output1);
      output1 = mad(input1.z, weight2, output1);
      output1 = mad(input1.w, weight3, output1);
    }

    // -------------2--------------
    pos_in = (int2)(i * input_width + in_pos_in_one_block2.x,
                    in_pos_in_one_block2.y);
    CL_DTYPE4 input2 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, SAMPLER, pos_in);

    if (abs(max_w_bound - pos_in.x) < input_width) {
      if (burndary_index == 0) {
        output2 = mad(input2.x, weight0, output2);
        output2 = mad(input2.y, weight1, output2);
        output2 = mad(input2.z, weight2, output2);
        output2 = mad(input2.w, weight3, output2);
      } else if (burndary_index == 1) {
        output2 = mad(input2.x, weight0, output2);
        output2 = mad(input2.y, weight1, output2);
        output2 = mad(input2.z, weight2, output2);
        output2 = mad(0.0f, weight3, output2);

      } else if (burndary_index == 2) {
        output2 = mad(input2.x, weight0, output2);
        output2 = mad(input2.y, weight1, output2);
        output2 = mad(0.0f, weight2, output2);
        output2 = mad(0.0f, weight3, output2);
      } else if (burndary_index == 3) {
        output2 = mad(input2.x, weight0, output2);
        output2 = mad(0.0f, weight1, output2);
        output2 = mad(0.0f, weight2, output2);
        output2 = mad(0.0f, weight3, output2);
      }
    } else {
      output2 = mad(input2.x, weight0, output2);
      output2 = mad(input2.y, weight1, output2);
      output2 = mad(input2.z, weight2, output2);
      output2 = mad(input2.w, weight3, output2);
    }

    // -------------3--------------
    pos_in = (int2)(i * input_width + in_pos_in_one_block3.x,
                    in_pos_in_one_block3.y);
    CL_DTYPE4 input3 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, SAMPLER, pos_in);

    if (abs(max_w_bound - pos_in.x) < input_width) {
      if (burndary_index == 0) {
        output3 = mad(input3.x, weight0, output3);
        output3 = mad(input3.y, weight1, output3);
        output3 = mad(input3.z, weight2, output3);
        output3 = mad(input3.w, weight3, output3);
      } else if (burndary_index == 1) {
        output3 = mad(input3.x, weight0, output3);
        output3 = mad(input3.y, weight1, output3);
        output3 = mad(input3.z, weight2, output3);
        output3 = mad(0.0f, weight3, output3);

      } else if (burndary_index == 2) {
        output3 = mad(input3.x, weight0, output3);
        output3 = mad(input3.y, weight1, output3);
        output3 = mad(0.0f, weight2, output3);
        output3 = mad(0.0f, weight3, output3);
      } else if (burndary_index == 3) {
        output3 = mad(input3.x, weight0, output3);
        output3 = mad(0.0f, weight1, output3);
        output3 = mad(0.0f, weight2, output3);
        output3 = mad(0.0f, weight3, output3);
      }
    } else {
      output3 = mad(input3.x, weight0, output3);
      output3 = mad(input3.y, weight1, output3);
      output3 = mad(input3.z, weight2, output3);
      output3 = mad(input3.w, weight3, output3);
    }
  }

  CL_DTYPE4 alpha0, alpha1, alpha2, alpha3;
#ifdef PRELU_CH  //{
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(out_c, 0));
  alpha1 = alpha0;
  alpha2 = alpha0;
  alpha3 = alpha0;
//}
#elif defined(PRELU_ELE)  //{
  if (out_w0 < old_w) {
    alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, output_pos0);
  }
  if (out_w1 < old_w) {
    alpha1 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, output_pos1);
  }
  if (out_w2 < old_w) {
    alpha2 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, output_pos2);
  }
  if (out_w3 < old_w) {
    alpha3 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, output_pos3);
  }
//}
#elif defined(PRELU_ALL)  //{
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(0, 0));
  alpha0.y = alpha0.x;
  alpha0.z = alpha0.x;
  alpha0.w = alpha0.x;
  alpha1 = alpha0;
  alpha2 = alpha0;
  alpha3 = alpha0;
//}
#endif
  output0 = activation_type4(output0, alpha0);
  output1 = activation_type4(output1, alpha1);
  output2 = activation_type4(output2, alpha2);
  output3 = activation_type4(output3, alpha3);

#ifdef SCALE_ACTIVATION
  output0 = fuse_scale(output0, 1.f, 0.f, 0.f);
  output1 = fuse_scale(output1, 1.f, 0.f, 0.f);
  output2 = fuse_scale(output2, 1.f, 0.f, 0.f);
  output3 = fuse_scale(output3, 1.f, 0.f, 0.f);
#endif

  if (out_w0 < old_w) {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image, output_pos0, &output0);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos0, output0);
  }

  if (out_w1 < old_w) {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image, output_pos1, &output1);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos1, output1);
  }

  if (out_w2 < old_w) {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image, output_pos2, &output2);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos2, output2);
  }

  if (out_w3 < old_w) {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image, output_pos3, &output3);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos3, output3);
  }
}

__kernel void conv2d_1x1_h1w4c1(
    __private const int global_size_dim0,
    __private const int global_size_dim1,
    __private const int global_size_dim2,
    __read_only image2d_t input_image,
    __read_only image2d_t filter,
    __read_only image2d_t bias,
    __write_only image2d_t output_image,
    __private const int stride,
    __private const int offset,
    __private const int input_c,
    __private const int input_c_origin,
    __private const int dilation,
    __private const int input_width,  /* of one block */
    __private const int input_height, /* of one block */
    __private const int output_width,
    __private const int output_height,
    __private const int old_w,
    __read_only image2d_t prelu_alpha
#ifdef ELT_FUSE
    ,
    __read_only image2d_t second_input_image
#endif
    ) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);
  if (out_c >= global_size_dim0 || out_w >= global_size_dim1 ||
      out_nh >= global_size_dim2) {
    return;
  }

  int out_w0 = out_w;
  int out_w1 = out_w + global_size_dim1;
  int out_w2 = out_w + global_size_dim1 * 2;
  int out_w3 = out_w + global_size_dim1 * 3;

  int outpos_main = mul24(out_c, old_w);
  int2 output_pos0 = (int2)(outpos_main + out_w0, out_nh);
  int2 output_pos1 = (int2)(outpos_main + out_w1, out_nh);
  int2 output_pos2 = (int2)(outpos_main + out_w2, out_nh);
  int2 output_pos3 = (int2)(outpos_main + out_w3, out_nh);

  int2 stride_xy = (int2)(stride, stride);

  int2 ouput_pos_in_one_block0 = (int2)(out_w0, out_nh);
  int2 in_pos_in_one_block0 =
      ouput_pos_in_one_block0 * stride_xy + (int2)(offset, offset);

  int2 ouput_pos_in_one_block1 = (int2)(out_w1, out_nh);
  int2 in_pos_in_one_block1 =
      ouput_pos_in_one_block1 * stride_xy + (int2)(offset, offset);

  int2 ouput_pos_in_one_block2 = (int2)(out_w2, out_nh);
  int2 in_pos_in_one_block2 =
      ouput_pos_in_one_block2 * stride_xy + (int2)(offset, offset);

  int2 ouput_pos_in_one_block3 = (int2)(out_w3, out_nh);
  int2 in_pos_in_one_block3 =
      ouput_pos_in_one_block3 * stride_xy + (int2)(offset, offset);

#ifdef BIASE_CH
  CL_DTYPE4 output0 =
      READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, (int2)(out_c, 0));
  CL_DTYPE4 output1 = output0;
  CL_DTYPE4 output2 = output0;
  CL_DTYPE4 output3 = output0;
#else
  CL_DTYPE4 output0 = 0.0f;
  CL_DTYPE4 output1 = 0.0f;
  CL_DTYPE4 output2 = 0.0f;
  CL_DTYPE4 output3 = 0.0f;
#endif

  for (int i = 0; i < input_c; ++i) {
    // ------------0---------------
    int2 pos_in = (int2)(i * input_width + in_pos_in_one_block0.x,
                         in_pos_in_one_block0.y);
    CL_DTYPE4 input0 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, SAMPLER, pos_in);

    CL_DTYPE4 weight0 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, filter, SAMPLER, (int2)(out_c, i * 4 + 0));
    CL_DTYPE4 weight1 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, filter, SAMPLER, (int2)(out_c, i * 4 + 1));
    CL_DTYPE4 weight2 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, filter, SAMPLER, (int2)(out_c, i * 4 + 2));
    CL_DTYPE4 weight3 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, filter, SAMPLER, (int2)(out_c, i * 4 + 3));

    output0 = mad(input0.x, weight0, output0);
    output0 = mad(input0.y, weight1, output0);
    output0 = mad(input0.z, weight2, output0);
    output0 = mad(input0.w, weight3, output0);

    pos_in = (int2)(i * input_width + in_pos_in_one_block1.x,
                    in_pos_in_one_block1.y);
    CL_DTYPE4 input1 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, SAMPLER, pos_in);
    output1 = mad(input1.x, weight0, output1);
    output1 = mad(input1.y, weight1, output1);
    output1 = mad(input1.z, weight2, output1);
    output1 = mad(input1.w, weight3, output1);

    pos_in = (int2)(i * input_width + in_pos_in_one_block2.x,
                    in_pos_in_one_block2.y);
    CL_DTYPE4 input2 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, SAMPLER, pos_in);
    output2 = mad(input2.x, weight0, output2);
    output2 = mad(input2.y, weight1, output2);
    output2 = mad(input2.z, weight2, output2);
    output2 = mad(input2.w, weight3, output2);

    pos_in = (int2)(i * input_width + in_pos_in_one_block3.x,
                    in_pos_in_one_block3.y);
    CL_DTYPE4 input3 =
        READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, SAMPLER, pos_in);
    output3 = mad(input3.x, weight0, output3);
    output3 = mad(input3.y, weight1, output3);
    output3 = mad(input3.z, weight2, output3);
    output3 = mad(input3.w, weight3, output3);
  }

  CL_DTYPE4 alpha0, alpha1, alpha2, alpha3;
#ifdef PRELU_CH  //{
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(out_c, 0));
  alpha1 = alpha0;
  alpha2 = alpha0;
  alpha3 = alpha0;
//}
#elif defined(PRELU_ELE)  //{
  if (out_w0 < old_w) {
    alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, output_pos0);
  }
  if (out_w1 < old_w) {
    alpha1 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, output_pos1);
  }
  if (out_w2 < old_w) {
    alpha2 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, output_pos2);
  }
  if (out_w3 < old_w) {
    alpha3 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, output_pos3);
  }
//}
#elif defined(PRELU_ALL)  //{
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(0, 0));
  alpha0.y = alpha0.x;
  alpha0.z = alpha0.x;
  alpha0.w = alpha0.x;
  alpha1 = alpha0;
  alpha2 = alpha0;
  alpha3 = alpha0;
//}
#endif
  output0 = activation_type4(output0, alpha0);
  output1 = activation_type4(output1, alpha1);
  output2 = activation_type4(output2, alpha2);
  output3 = activation_type4(output3, alpha3);

#ifdef SCALE_ACTIVATION
  output0 = fuse_scale(output0, 1.f, 0.f, 0.f);
  output1 = fuse_scale(output1, 1.f, 0.f, 0.f);
  output2 = fuse_scale(output2, 1.f, 0.f, 0.f);
  output3 = fuse_scale(output3, 1.f, 0.f, 0.f);
#endif

  if (out_w0 < old_w) {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image, output_pos0, &output0);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos0, output0);
  }

  if (out_w1 < old_w) {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image, output_pos1, &output1);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos1, output1);
  }

  if (out_w2 < old_w) {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image, output_pos2, &output2);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos2, output2);
  }

  if (out_w3 < old_w) {
#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image, output_pos3, &output3);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos3, output3);
  }
}

/*
inline void elt_fuse_func_wrapper(__read_only image2d_t second_input_image,
                                  const int2 pos,
                                  CL_DTYPE4 *value_p) {
  CL_DTYPE4 second_val =
      READ_IMG_TYPE(CL_DTYPE_CHAR, second_input_image, SAMPLER, pos);
  *value_p += second_val;
#ifdef ELT_ACT_FUSE
  *value_p = fmax(*value_p, (CL_DTYPE4)0);
#endif
}*/

__kernel void conv2d_1x1_fc(__read_only image2d_t input,
                            __write_only image2d_t output,
                            __global CL_DTYPE16 *weights,
#ifdef BIASE_CH
                            __read_only image2d_t biases,
#endif  // BIASE_CH
#ifdef PRELU
                            __read_only image2d_t prelu_alpha,
#endif  // PRELU
#ifdef ELT_FUSE
                            __read_only image2d_t second_input_image,
#endif  // ELT_FUSE
                            int in_c_blks,
                            int out_c_blks) {
  int out_c = get_global_id(0);
  int2 tid = (int2)(get_local_id(0), get_local_id(1));
  CL_DTYPE4 s = (CL_DTYPE4)(0.0f);

  if (out_c < out_c_blks) {
    for (int c = tid.y; c < in_c_blks; c += 4) {
      CL_DTYPE4 v = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(c, 0));
      CL_DTYPE16 w = weights[c * out_c_blks + out_c];
      CL_DTYPE4 partial = v.x * w.s0123;
      partial += v.y * w.s4567;
      partial += v.z * w.s89ab;
      partial += v.w * w.scdef;
      s += partial;
    }
  }
  __local CL_DTYPE4 temp[32][4];
  temp[tid.x][tid.y] = s;
  barrier(CLK_LOCAL_MEM_FENCE);

  if (out_c >= out_c_blks) {
    return;
  }
  if (tid.y == 0) {
    s += temp[tid.x][1];
    s += temp[tid.x][2];
    s += temp[tid.x][3];
    int2 output_pos0 = (int2)(out_c, 0);

#ifdef BIASE_CH
    CL_DTYPE4 output0 =
        s + READ_IMG_TYPE(CL_DTYPE_CHAR, biases, SAMPLER, output_pos0);
#else
    CL_DTYPE4 output0 = s;
#endif

    CL_DTYPE4 alpha0;
#ifdef PRELU_CH
    alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, output_pos0);
#elif defined(PRELU_ELE)
    alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, output_pos0);
#elif defined(PRELU_ALL)
    alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(0, 0));
    alpha0.y = alpha0.x;
    alpha0.z = alpha0.x;
    alpha0.w = alpha0.x;
#endif
    output0 = activation_type4(output0, alpha0);
#ifdef SCALE_ACTIVATION
    output0 = fuse_scale(output0, 1.f, 0.f, 0.f);
#endif

#ifdef ELT_FUSE
    elt_fuse_func_wrapper(second_input_image, output_pos0, &output0);
#endif
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, output_pos0, output0);
  }
}

__kernel void elementwise_add(__read_only image2d_t input,
                              __read_only image2d_t bias,
                              __write_only image2d_t outputImage,
                              int h, int w) {
     int x = get_global_id(0);
     int y = get_global_id(1);

     int2 coords;
     coords.x = x;
     coords.y = y;

#ifdef BROADCAST
     int c_blk = x / w;
     int n_blk = y / h;
     CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(c_blk, n_blk));
#else
     CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, coords);
#endif
     CL_DTYPE4 biase = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, coords);
     CL_DTYPE4 alpha;
     CL_DTYPE4 output = activation_type4(in + biase, alpha);

     WRITE_IMG_TYPE(CL_DTYPE_CHAR, outputImage,coords,output);
 }

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

 __kernel void SplitChannel(__read_only image2d_t input,
                            __write_only image2d_t output0,
                            __write_only image2d_t output1,
                            __private const int axis,
                            __private const int out0_dims_axis,
                            __private const int in_dims_second,
                            __private const int in_dims_last,
                            __private const int width) {
   const int channel_blk_idx = get_global_id(0);
   const int width_idx = get_global_id(1);
   const int hb_idx = get_global_id(2);

   const int2 in_pos = (int2)(channel_blk_idx * in_dims_last + width_idx, hb_idx);
   const CL_DTYPE4 in_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, in_pos);
   const int c = channel_blk_idx * 4;

   // write all data to output0 directly
   if (c < out0_dims_axis) {
     WRITE_IMG_TYPE(CL_DTYPE_CHAR, output0, in_pos, in_data);
   }

   // deal with the last channel of output0
   int channel_offset = out0_dims_axis % 4;
   if (channel_blk_idx == out0_dims_axis / 4) { // only the last channel_blk of output0 hits this
     if (channel_offset != 0) {
       CL_DTYPE4 out0_last_val;
       if (channel_offset == 1) {
         out0_last_val = (CL_DTYPE4)(in_data.x, 0, 0, 0);
       } else if (channel_offset == 2) {
         out0_last_val = (CL_DTYPE4)(in_data.x, in_data.y, 0, 0);
       } else if (channel_offset == 3) {
         out0_last_val = (CL_DTYPE4)(in_data.x, in_data.y, in_data.z, 0);
       }
       WRITE_IMG_TYPE(CL_DTYPE_CHAR, output0, in_pos, out0_last_val);
     }
   }

   // deal with output1
   if (c + 4 >= out0_dims_axis) { // only theads for output1 hit this
     const int2 out_pos = (int2)((channel_blk_idx - out0_dims_axis / 4) * in_dims_last + width_idx, hb_idx);
     if (channel_offset == 0) { // write all data to output1 directly
       WRITE_IMG_TYPE(CL_DTYPE_CHAR, output1, out_pos, in_data);
     } else {
       CL_DTYPE4 combined_val;
       CL_DTYPE4 latter = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(in_pos.x + in_dims_last, in_pos.y));
       if (channel_offset == 1) {
         combined_val = (CL_DTYPE4)(in_data.y, in_data.z, in_data.w, latter.x);
       } else if (channel_offset == 2) {
         combined_val = (CL_DTYPE4)(in_data.z, in_data.w, latter.x, latter.y);
       } else if (channel_offset == 3) {
         combined_val = (CL_DTYPE4)(in_data.w, latter.x, latter.y, latter.z);
       }
       WRITE_IMG_TYPE(CL_DTYPE_CHAR, output1, out_pos, combined_val);
     }
   }
 }

 __kernel void transpose_general_buffer(__global const CL_DTYPE* src,
                                        __global CL_DTYPE* dst,
                                        __global const int* out_idxs,
                                        __private const int out_tensor_c,
                                        __private const int out_tensor_h,
                                        __private const int out_tensor_w,
                                        __private const int out_tensor_hw) {
   int hidx = get_global_id(0); // [0, h) columns of dst
   int widx = get_global_id(1); // [0, w) rows of dst
   int chidx = get_global_id(2); // [0, ch) channels of dst

   // idx = chidx * out_tensor_hw + hidx * out_tensor_w + widx
   const int idx = mad((CL_DTYPE)chidx,
                       (CL_DTYPE)out_tensor_hw,
                       (CL_DTYPE)(mul24(hidx, out_tensor_w) + widx));

   dst[out_idxs[idx]] = src[idx];
 }