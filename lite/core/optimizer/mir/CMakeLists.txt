add_subdirectory(fusion)
add_subdirectory(elimination)
add_subdirectory(subgraph)
lite_cc_test(test_pattern_matcher SRCS pattern_matcher_test.cc DEPS core)
# Top-level pass unit tests, built on the hand-written SSAGraph harness
# (fusion/fusion_pass_test_util). Run on both host and ARM builds.
lite_cc_test(test_adaptive_1x1_pool2d_convert_global_pass
             SRCS adaptive_1x1_pool2d_convert_global_pass_test.cc
             fusion/fusion_pass_test_util.cc DEPS core)
lite_cc_test(test_silu_to_swish_pass
             SRCS silu_to_swish_pass_test.cc fusion/fusion_pass_test_util.cc
             DEPS core)
lite_cc_test(test_swish_to_silu_fuse_pass
             SRCS swish_to_silu_fuse_pass_test.cc
             fusion/fusion_pass_test_util.cc DEPS core)
lite_cc_test(test_support_0_dim_tensor_pass
             SRCS support_0_dim_tensor_pass_test.cc
             fusion/fusion_pass_test_util.cc DEPS core)
# for mobile, unnecessary to compile the following testings.
if(LITE_WITH_ARM)
    return()
endif()
lite_cc_test(test_mir_pass_manager SRCS pass_manager_test.cc DEPS core)
