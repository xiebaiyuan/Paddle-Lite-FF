# Use ccache if found ccache program

find_program(CCACHE_PATH ccache)

if(CCACHE_PATH)
    message(STATUS "Ccache is founded, use ccache to speed up compile.")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_PATH})
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ${CCACHE_PATH})
    # Propagate ccache to third-party ExternalProject builds (gtest/gflags/
    # glog/flatbuffers/...). Their cmake runs in a separate build dir so they
    # do not inherit RULE_LAUNCH_COMPILE; pass the compiler launcher via the
    # shared EXTERNAL_OPTIONAL_ARGS that the external/*.cmake files append.
    set(CCACHE_LAUNCHER_ARG "-DCMAKE_CXX_COMPILER_LAUNCHER=${CCACHE_PATH}"
                            "-DCMAKE_C_COMPILER_LAUNCHER=${CCACHE_PATH}")
    set(EXTERNAL_OPTIONAL_ARGS ${CCACHE_LAUNCHER_ARG})
else()
    set(CCACHE_LAUNCHER_ARG "")
    set(EXTERNAL_OPTIONAL_ARGS "")
endif(CCACHE_PATH)
