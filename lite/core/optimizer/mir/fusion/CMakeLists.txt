# Fusion-pass unit tests. These exercise the FuseBase pattern matchers on a
# hand-built SSAGraph (see fusion_pass_test_util), so they need the full
# bundled API (passes + ops). They run on both host (LITE_WITH_ARM=OFF, x86
# kernels) and ARM builds (LITE_WITH_ARM=ON, ARM kernels) — the graph only
# needs one registered kernel per op to build.
lite_cc_test(test_gelu_fuse_pass
             SRCS gelu_fuse_pass_test.cc fusion_pass_test_util.cc DEPS core)
lite_cc_test(test_div_mul_fuse_pass
             SRCS div_mul_fuse_pass_test.cc fusion_pass_test_util.cc DEPS core)
lite_cc_test(test_scale_clip_mul_fuse_pass
             SRCS scale_clip_mul_fuse_pass_test.cc fusion_pass_test_util.cc DEPS core)
lite_cc_test(test_conv_activation_gelu_fuse
             SRCS conv_activation_gelu_fuse_test.cc fusion_pass_test_util.cc
             DEPS core)
lite_cc_test(test_conv_activation_fuse_pass
             SRCS conv_activation_fuse_pass_test.cc fusion_pass_test_util.cc
             DEPS core)
lite_cc_test(test_conv_scale_fuse_pass
             SRCS conv_scale_fuse_pass_test.cc fusion_pass_test_util.cc
             DEPS core)
lite_cc_test(test_conv_bn_fuse_pass
             SRCS conv_bn_fuse_pass_test.cc fusion_pass_test_util.cc
             DEPS core)
lite_cc_test(test_conv_elementwise_fuse_pass
             SRCS conv_elementwise_fuse_pass_test.cc fusion_pass_test_util.cc
             DEPS core)
lite_cc_test(test_conv_elementwise_tree_fuse_pass
             SRCS conv_elementwise_tree_fuse_pass_test.cc
             fusion_pass_test_util.cc DEPS core)
lite_cc_test(test_conv_hardswish_fuse_pass
             SRCS conv_hardswish_fuse_pass_test.cc fusion_pass_test_util.cc
             DEPS core)
lite_cc_test(test_conv_conv_fuse_pass
             SRCS conv_conv_fuse_pass_test.cc fusion_pass_test_util.cc
             DEPS core)
lite_cc_test(test_elementwise_add_activation_fuse_pass
             SRCS elementwise_add_activation_fuse_pass_test.cc
             fusion_pass_test_util.cc DEPS core)
lite_cc_test(test_elementwise_add_reshape_fuse_pass
             SRCS elementwise_add_reshape_fuse_pass_test.cc
             fusion_pass_test_util.cc DEPS core)
lite_cc_test(test_elementwise_add_scale_fuse_pass
             SRCS elementwise_add_scale_fuse_pass_test.cc
             fusion_pass_test_util.cc DEPS core)
lite_cc_test(test_fc_prelu_fuse_pass
             SRCS fc_prelu_fuse_pass_test.cc fusion_pass_test_util.cc
             DEPS core)
lite_cc_test(test_fc_fuser_fuse_pass
             SRCS fc_fuser_fuse_pass_test.cc fusion_pass_test_util.cc
             DEPS core)
lite_cc_test(test_fill_range_fuse_pass
             SRCS fill_range_fuse_pass_test.cc fusion_pass_test_util.cc
             DEPS core)
lite_cc_test(test_flatten_fc_fuse_pass
             SRCS flatten_fc_fuse_pass_test.cc fusion_pass_test_util.cc
             DEPS core)
lite_cc_test(test_greater_than_cast_fuse_pass
             SRCS greater_than_cast_fuse_pass_test.cc
             fusion_pass_test_util.cc DEPS core)
lite_cc_test(test_inplace_fuse_pass
             SRCS inplace_fuse_pass_test.cc fusion_pass_test_util.cc
             DEPS core)
lite_cc_test(test_instance_norm_activation_fuse_pass
             SRCS instance_norm_activation_fuse_pass_test.cc
             fusion_pass_test_util.cc DEPS core)
lite_cc_test(test_interpolate_fuse_pass
             SRCS interpolate_fuse_pass_test.cc fusion_pass_test_util.cc
             DEPS core)
lite_cc_test(test_keepdims_convert_fuse_pass
             SRCS keepdims_convert_fuse_pass_test.cc fusion_pass_test_util.cc
             DEPS core)
lite_cc_test(test_layer_norm_fuse_pass
             SRCS layer_norm_fuse_pass_test.cc fusion_pass_test_util.cc
             DEPS core)
lite_cc_test(test_matmul_fuse_pass
             SRCS matmul_fuse_pass_test.cc fusion_pass_test_util.cc
             DEPS core)
lite_cc_test(test_matmul_elementwise_add_fuse_pass
             SRCS matmul_elementwise_add_fuse_pass_test.cc
             fusion_pass_test_util.cc DEPS core)
lite_cc_test(test_p_norm_fill_constant_max_div_fuse_pass
             SRCS p_norm_fill_constant_max_div_fuse_pass_test.cc
             fusion_pass_test_util.cc DEPS core)
lite_cc_test(test_reshape2_matmul_fuse_pass
             SRCS reshape2_matmul_fuse_pass_test.cc fusion_pass_test_util.cc
             DEPS core)
lite_cc_test(test_scale_activation_fuse_pass
             SRCS scale_activation_fuse_pass_test.cc fusion_pass_test_util.cc
             DEPS core)
lite_cc_test(test_scaleacts_fuse_pass
             SRCS scaleacts_fuse_pass_test.cc fusion_pass_test_util.cc
             DEPS core)
lite_cc_test(test_scales_fuse_pass
             SRCS scales_fuse_pass_test.cc fusion_pass_test_util.cc
             DEPS core)
lite_cc_test(test_shuffle_channel_fuse_pass
             SRCS shuffle_channel_fuse_pass_test.cc fusion_pass_test_util.cc
             DEPS core)
lite_cc_test(test_sigmoid_elementmul_fuse_pass
             SRCS sigmoid_elementmul_fuse_pass_test.cc
             fusion_pass_test_util.cc DEPS core)
lite_cc_test(test_squeeze2_matmul_fuse_pass
             SRCS squeeze2_matmul_fuse_pass_test.cc fusion_pass_test_util.cc
             DEPS core)
lite_cc_test(test_transpose_softmax_transpose_fuse_pass
             SRCS transpose_softmax_transpose_fuse_pass_test.cc
             fusion_pass_test_util.cc DEPS core)
lite_cc_test(test_unsqueeze2_pad3d_squeeze2_fuse_pass
             SRCS unsqueeze2_pad3d_squeeze2_fuse_pass_test.cc
             fusion_pass_test_util.cc DEPS core)
lite_cc_test(test_transformer_attention_fuse_pass
             SRCS transformer_attention_fuse_pass_test.cc
             fusion_pass_test_util.cc DEPS core)
lite_cc_test(test_quant_dequant_fuse_pass
             SRCS quant_dequant_fuse_pass_test.cc fusion_pass_test_util.cc
             DEPS core)
