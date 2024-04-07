# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include(${OHOS_SDK}/native/build/cmake/ohos.toolchain.cmake)
#if(LITE_WITH_OPENMP)
#    set(OpenMP_C_FLAGS "-fopenmp")
#    set(OpenMP_C_LIB_NAMES "omp")
#    set(OpenMP_CXX_FLAGS "-fopenmp")
#    set(OpenMP_CXX_LIB_NAMES "omp")
#    set(OpenMP_omp_LIBRARY omp)
#    set(OpenMP_C_FLAGS_WORK "-fopenmp")
#    set(OpenMP_C_LIB_NAMES_WORK "omp")
#    set(OpenMP_CXX_FLAGS_WORK "-fopenmp")
#    set(OpenMP_CXX_LIB_NAMES_WORK "omp")
#endif()

# Definitions
add_definitions(-DLITE_WITH_LINUX)
add_definitions(-DLITE_WITH_OHOS)

if(ARM_TARGET_ARCH_ABI STREQUAL "armv7")
    set(OHOS_ARCH "armeabi-v7a")
endif()

if(ARM_TARGET_ARCH_ABI STREQUAL "armv8")
    set(OHOS_ARCH "arm64-v8a")
endif()

#
#if(ARM_TARGET_LANG STREQUAL "clang")
#    set(CMAKE_C_COMPILER clang)
#    set(CMAKE_CXX_COMPILER clang++)
#    message(STATUS "CMAKE_CXX_COMPILER_TARGET: ${CMAKE_CXX_COMPILER_TARGET}")
#endif()

if(OHOS)
    message(STATUS "Build with OHOS")
    set(CROSS_COMPILE_CMAKE_ARGS ${CROSS_COMPILE_CMAKE_ARGS}
        "-DCMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}"
        "-DCMAKE_CXX_STANDARD=11"
        "-DCMAKE_TOOLCHAIN_FILE=${OHOS_SDK}/native/build/cmake/ohos.toolchain.cmake"
        "-DARM_TARGET_ARCH_ABI=${ARM_TARGET_ARCH_ABI}"
        "-DOHOS_ARCH=${OHOS_ARCH}"
        "-DOHOS_TOOLCHAIN=${ARM_TARGET_LANG}"
        "-DCMAKE_EXE_LINKER_FLAGS=${CMAKE_EXE_LINKER_FLAGS}"
        "-DCMAKE_SHARED_LINKER_FLAGS=${CMAKE_SHARED_LINKER_FLAGS}"
        "-DARM_NEON=ON"
        )
endif()

