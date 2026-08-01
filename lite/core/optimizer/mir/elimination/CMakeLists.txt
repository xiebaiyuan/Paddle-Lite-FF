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
