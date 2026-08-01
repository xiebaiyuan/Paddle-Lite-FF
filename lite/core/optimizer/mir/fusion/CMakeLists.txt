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
