if (NOT LITE_WITH_ARM)
  # NOTE disabled for the proto_desc is not valid yet.
  # TODO(Superjomn) enable them if valid latter.
  # lite_cc_test(test_identity_scale_eliminate_pass
  #   SRCS identity_scale_eliminate_pass_test.cc
  #   DEPS core proto_desc cpp_op_desc
  #   ops
  #   )
endif()

# assign_eliminate_pass test. Reuses fusion_pass_test_util to build a
# hand-written SSAGraph. Runs on both host and ARM builds — the graph only
# needs one registered kernel per op.
lite_cc_test(test_assign_eliminate_pass
             SRCS assign_eliminate_pass_test.cc
             ${CMAKE_CURRENT_SOURCE_DIR}/../fusion/fusion_pass_test_util.cc
             DEPS core)
lite_cc_test(test_identity_scale_eliminate_pass
             SRCS identity_scale_eliminate_pass_test.cc
             ${CMAKE_CURRENT_SOURCE_DIR}/../fusion/fusion_pass_test_util.cc
             DEPS core)
lite_cc_test(test_identity_dropout_eliminate_pass
             SRCS identity_dropout_eliminate_pass_test.cc
             ${CMAKE_CURRENT_SOURCE_DIR}/../fusion/fusion_pass_test_util.cc
             DEPS core)
lite_cc_test(test_identity_transpose_eliminate_pass
             SRCS identity_transpose_eliminate_pass_test.cc
             ${CMAKE_CURRENT_SOURCE_DIR}/../fusion/fusion_pass_test_util.cc
             DEPS core)
lite_cc_test(test_elementwise_mul_constant_eliminate_pass
             SRCS elementwise_mul_constant_eliminate_pass_test.cc
             ${CMAKE_CURRENT_SOURCE_DIR}/../fusion/fusion_pass_test_util.cc
             DEPS core)
lite_cc_test(test_remove_scale1_pass
             SRCS remove_scale1_pass_test.cc
             ${CMAKE_CURRENT_SOURCE_DIR}/../fusion/fusion_pass_test_util.cc
             DEPS core)
lite_cc_test(test_reshape_calc_offline_pass
             SRCS reshape_calc_offline_pass_test.cc
             ${CMAKE_CURRENT_SOURCE_DIR}/../fusion/fusion_pass_test_util.cc
             DEPS core)
lite_cc_test(test_remove_tf_redundant_ops_pass
             SRCS remove_tf_redundant_ops_pass_test.cc
             ${CMAKE_CURRENT_SOURCE_DIR}/../fusion/fusion_pass_test_util.cc
             DEPS core)
lite_cc_test(test_reshape2_cast_eliminate_pass
             SRCS reshape2_cast_eliminate_pass_test.cc
             ${CMAKE_CURRENT_SOURCE_DIR}/../fusion/fusion_pass_test_util.cc
             DEPS core)
lite_cc_test(test_dynamic_shape_pass
             SRCS dynamic_shape_pass_test.cc
             ${CMAKE_CURRENT_SOURCE_DIR}/../fusion/fusion_pass_test_util.cc
             DEPS core)
lite_cc_test(test_ssd_boxes_calc_offline_pass
             SRCS ssd_boxes_calc_offline_pass_test.cc
             ${CMAKE_CURRENT_SOURCE_DIR}/../fusion/fusion_pass_test_util.cc
             DEPS core)
lite_cc_test(test_scale_calc_offline_pass
             SRCS scale_calc_offline_pass_test.cc
             ${CMAKE_CURRENT_SOURCE_DIR}/../fusion/fusion_pass_test_util.cc
             DEPS core)
lite_cc_test(test_fill_constant_calc_offline_pass
             SRCS fill_constant_calc_offline_pass_test.cc
             ${CMAKE_CURRENT_SOURCE_DIR}/../fusion/fusion_pass_test_util.cc
             DEPS core)
lite_cc_test(test_range_calc_offline_pass
             SRCS range_calc_offline_pass_test.cc
             ${CMAKE_CURRENT_SOURCE_DIR}/../fusion/fusion_pass_test_util.cc
             DEPS core)
lite_cc_test(test_unsqueeze_calc_offline_pass
             SRCS unsqueeze_calc_offline_pass_test.cc
             ${CMAKE_CURRENT_SOURCE_DIR}/../fusion/fusion_pass_test_util.cc
             DEPS core)
lite_cc_test(test_assign_value_calc_offline_pass
             SRCS assign_value_calc_offline_pass_test.cc
             ${CMAKE_CURRENT_SOURCE_DIR}/../fusion/fusion_pass_test_util.cc
             DEPS core)
