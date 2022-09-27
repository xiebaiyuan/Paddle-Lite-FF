#include <cl_common.h>

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
