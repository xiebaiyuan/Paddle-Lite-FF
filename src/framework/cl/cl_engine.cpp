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

#include "framework/cl/cl_engine.h"
#include "CL/cl.h"
#include "framework/cl/cl_tool.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace paddle_mobile {
namespace framework {
constexpr const char *kQualcommAdrenoGPUStr = "QUALCOMM Adreno(TM)";
constexpr const char *kMaliGPUStr = "Mali";

/*
GPUType ParseGPUType(const std::string &device_name) {
  constexpr const char *kQualcommAdrenoGPUStr = "QUALCOMM Adreno(TM)";
  constexpr const char *kMaliGPUStr = "Mali";
  constexpr const char *kPowerVRGPUStr = "PowerVR";

  if (device_name == kQualcommAdrenoGPUStr) {
    return GPUType::QUALCOMM_ADRENO;
  } else if (device_name.find(kMaliGPUStr) != std::string::npos) {
    return GPUType::MALI;
  } else if (device_name.find(kPowerVRGPUStr) != std::string::npos) {
    return GPUType::PowerVR;
  } else {
    return GPUType::UNKNOWN;
  }
}
*/

bool CLEngine::Init() {
  if (initialized_) {
    return true;
  }
  cl_int status;
  SetPlatform();
  SetClDeviceId();
  SetClInfos();
  initialized_ = true;
  return initialized_;
  //  setClCommandQueue();
  //  std::string filename = "./HelloWorld_Kernel.cl";
  //  loadKernelFromFile(filename.c_str();
  //  buildProgram();
}

CLEngine *CLEngine::Instance() {
  static CLEngine cl_engine_;
  cl_engine_.Init();
  return &cl_engine_;
}

bool CLEngine::SetPlatform() {
  platform_ = NULL;      // the chosen platform
  cl_uint numPlatforms;  // the NO. of platforms
  cl_int status = clGetPlatformIDs(0, NULL, &numPlatforms);

  /**For clarity, choose the first available platform. */
  if (numPlatforms > 0) {
    cl_platform_id *platforms = reinterpret_cast<cl_platform_id *>(
        malloc(numPlatforms * sizeof(cl_platform_id)));
    status = clGetPlatformIDs(numPlatforms, platforms, NULL);
    platform_ = platforms[0];
    free(platforms);
    return true;
  } else {
    return false;
  }
}

bool CLEngine::SetClDeviceId() {
  cl_uint numDevices = 0;
  devices_ = NULL;
  cl_int status =
      clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);

  if (numDevices > 0) {
    devices_ = reinterpret_cast<cl_device_id *>(
        malloc(numDevices * sizeof(cl_device_id)));
    status = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, numDevices, devices_,
                            NULL);
    return true;
  }
  return false;
}
void CLEngine::getPlatformInfo(cl_platform_id id, cl_platform_info name,
                               std::string str) {
  cl_int errNum;
  std::size_t paramValueSize;

  errNum = clGetPlatformInfo(id, name, 0, NULL, &paramValueSize);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Failed to find OpenCL platform " << str << "." << std::endl;
    return;
  }

  char *info = (char *)alloca(sizeof(char) * paramValueSize);
  errNum = clGetPlatformInfo(id, name, paramValueSize, info, NULL);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Failed to find OpenCL platform " << str << "." << std::endl;
    return;
  }

  std::cout << "\t" << str << ":\t" << info << std::endl;
}

bool CLEngine::GetWorkItemSize() {
  PADDLE_MOBILE_ENFORCE(devices_ != nullptr, "devices_ can not be null ");
  if (devices_ == nullptr) {
    return false;
  }
  /* cl_int clGetDeviceInfo(	cl_device_id device,
       cl_device_info param_name,
       size_t param_value_size,
       void *param_value,
       size_t *param_value_size_ret)*/
  //  size_t [] max_work_item_size = getDeviceInfo(*devices_,
  //   CL_DEVICE_MAX_WORK_ITEM_SIZES,std::string("GetWorkItemSize")
  //  DLOG << "max_work_item_size" << max_work_item_size;

  cl_int errNum;
  std::size_t paramValueSize;

  errNum = clGetDeviceInfo(*devices_, CL_DEVICE_MAX_WORK_ITEM_SIZES, 0, NULL,
                           &paramValueSize);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Failed to find OpenCL device info "
              << "GetWorkItemSize"
              << "." << std::endl;
    return false;
  }

  //  size_t[] *info = (size_t[] *)alloca(sizeof(size_t[]) * paramValueSize);
  //  errNum = clGetDeviceInfo(*devices_, CL_DEVICE_MAX_WORK_ITEM_SIZES,
  //  paramValueSize, work_size, NULL);

  cl_uint max_work_item_dimensions;
  clGetDeviceInfo(*devices_, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                  sizeof(max_work_item_dimensions), &max_work_item_dimensions,
                  nullptr);
  printf("哈哈 : CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS %u\n",
         max_work_item_dimensions);
  size_t *max_work_item_sizes =
      (size_t *)malloc(sizeof(size_t) * max_work_item_dimensions);
  clGetDeviceInfo(*devices_, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                  sizeof(size_t) * max_work_item_dimensions,
                  max_work_item_sizes, nullptr);
  printf("哈哈 CL_DEVICE_MAX_WORK_ITEM_SIZES: ");
  for (size_t i = 0; i < max_work_item_dimensions; ++i)
    printf("%lu\t", max_work_item_sizes[i]);
  printf("\n");
  free(max_work_item_sizes);
//  cl_ulong global_mem_cache_size;
  clGetDeviceInfo(*devices_, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
                  sizeof(global_mem_cache_size_), &global_mem_cache_size_,
                  nullptr);
  printf("哈哈 CL_DEVICE_GLOBAL_MEM_CACHE_SIZE: %lu B = %lu KB\n",
         static_cast<unsigned long>(global_mem_cache_size_),
         static_cast<unsigned long>(global_mem_cache_size_ / 1024));
  cl_uint max_compute_units_;
  clGetDeviceInfo(*devices_, CL_DEVICE_MAX_COMPUTE_UNITS,
                  sizeof(max_compute_units_), &max_compute_units_, NULL);
  printf("哈哈 CL_DEVICE_MAX_COMPUTE_UNITS: %u\n", max_compute_units_);
  if (errNum != CL_SUCCESS) {
    std::cerr << "Failed to find OpenCL device info "
              << "GetWorkItemSize"
              << "." << std::endl;
    return false;
  }
  return true;
}
size_t CLEngine::getKernelWorkGroupInfo(cl_kernel kernel) {
  size_t global;  // global domain size
  size_t local;   // local domain size
  clGetKernelWorkGroupInfo(kernel, *devices_, CL_KERNEL_WORK_GROUP_SIZE,
                           sizeof(local), &local, NULL);
  printf("getKernelWorkGroupInfo %d \n", local);
  return local;
}

const uint32_t kernel_cache_size = (4 + 4 + 4) * 4 * 4;
// TODO(liuqi): Fix the specific value.
const uint32_t lws_limit = 128;
std::vector<uint32_t> CLEngine::LocalWorkSize1x1(size_t work_group_size,
                                                 const uint32_t *gws) {
  std::vector<uint32_t> lws(4, 0);
  if (work_group_size == 0) {
    lws[0] = lws[1] = lws[2] = 1;
  } else {

//    cl_uint max_work_item_dimensions;
//    clGetDeviceInfo(*devices_, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
//                    sizeof(max_work_item_dimensions), &max_work_item_dimensions,
//                    nullptr);
//    printf(" LocalWorkSize1x1: CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS %u\n",
//           max_work_item_dimensions);
//    size_t *max_work_item_sizes =
//        (size_t *)malloc(sizeof(size_t) * max_work_item_dimensions);
//    clGetDeviceInfo(*devices_, CL_DEVICE_MAX_WORK_ITEM_SIZES,
//                    sizeof(size_t) * max_work_item_dimensions,
//                    max_work_item_sizes, nullptr);
//    printf("LocalWorkSize1x1: CL_DEVICE_MAX_WORK_ITEM_SIZES: ");
//    for (size_t i = 0; i < max_work_item_dimensions; ++i)
//      printf("%lu\t", max_work_item_sizes[i]);
//    printf("\n");
//    free(max_work_item_sizes);
//    clGetDeviceInfo(*devices_, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
//                    sizeof(global_mem_cache_size_), &global_mem_cache_size_,
//                    nullptr);
//    printf("LocalWorkSize1x1: CL_DEVICE_GLOBAL_MEM_CACHE_SIZE: %lu B = %lu KB\n",
//           static_cast<unsigned long>(global_mem_cache_size_),
//           static_cast<unsigned long>(global_mem_cache_size_ / 1024));
//    clGetDeviceInfo(*devices_, CL_DEVICE_MAX_COMPUTE_UNITS,
//                    sizeof(max_compute_units_), &max_compute_units_, NULL);
//    printf("LocalWorkSize1x1: CL_DEVICE_MAX_COMPUTE_UNITS: %u\n", max_compute_units_);

//    uint32_t compute_units = max_compute_units_;

    // Base GPU cache size used for computing local work group size.
    const int32_t kBaseGPUMemCacheSize = 16384/2;
//    const int32_t kBaseGPUMemCacheSize = 65536;

    const uint32_t lw_base = std::max<uint32_t>(
        static_cast<const uint32_t &>(global_mem_cache_size_ / kBaseGPUMemCacheSize), 1);
    // work_group_size 是getKernelGroupInfo
    lws[1] = std::min<uint32_t>(gws[1],static_cast<const uint32_t &>(work_group_size));
    if (lws[1] >= lw_base) {
      lws[0] = std::min<uint32_t>(gws[0], lw_base);
    } else if ((1 < lws[1] && lws[1] < lw_base) && gws[0] >= lws_limit) {
      lws[0] = std::min<uint32_t>(gws[0], lw_base);
    } else {
      lws[0] = gws[0] / 8;
      if (lws[0] < lw_base) {
        lws[0] = std::max<uint32_t>(gws[0] / 4, lw_base);
      }
    }
    lws[0] = std::min<uint32_t>(
        lws[0], static_cast<const uint32_t &>(work_group_size / lws[1]));
    const uint32_t lws_size = lws[0] * lws[1];
    lws[2] = std::min<uint32_t>(
        static_cast<const uint32_t &>(
            (global_mem_cache_size_ / kernel_cache_size / lws_size / max_compute_units_) * 8),
        gws[2]);
    if (lws[2] == 0) {
      lws[2] = std::min<uint32_t>(gws[2], lw_base);
    }

    lws[2] = std::max<uint32_t>(
        std::min<uint32_t>(
            lws[2], static_cast<const uint32_t &>(work_group_size / lws_size)),
        1);

    /*for (int j = 0; j < 3; ++j) {
      while (lws[j]>1&&!isMultipul(gws[j], lws[j]) ) {
        lws[j]--;
      }
    }*/
  }
  return lws;
}
void CLEngine::SetClInfos() {
  // todo init properties_
  char buffer[1024];

  cl_uint num_platforms;
  clGetPlatformIDs(0, NULL, &num_platforms);
  //	printf("%d PLATFORMS FOUND\n", num_platforms);
  cl_platform_id *platforms =
      (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
  clGetPlatformIDs(num_platforms, platforms, NULL);
  for (cl_uint k = 0; k < num_platforms; ++k) {
    cl_platform_id platform = platforms[k];
    printf("CL_PLATFORM: %d\n", k);
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(buffer), buffer, NULL);
    printf("CL_PLATFORM_NAME: %s\n", buffer);
    clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, sizeof(buffer), buffer,
                      NULL);
    printf("CL_PLATFORM_VENDOR: %s\n", buffer);
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(buffer), buffer,
                      NULL);
    printf("CL_PLATFORM_VERSION: %s\n", buffer);
    clGetPlatformInfo(platform, CL_PLATFORM_PROFILE, sizeof(buffer), buffer,
                      NULL);
    printf("CL_PLATFORM_PROFILE: %s\n", buffer);
    clGetPlatformInfo(platform, CL_PLATFORM_EXTENSIONS, sizeof(buffer), buffer,
                      NULL);
    printf("CL_PLATFORM_EXTENSIONS: %s\n", buffer);
    printf("\n");

    cl_uint num_devices;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    //		printf("%d DEVICES FOUND\n", num_devices);
    cl_device_id *devices =
        (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
    for (cl_uint j = 0; j < num_devices; ++j) {
      cl_device_id device = devices[j];
      printf("CL_DEVICE: %d\n", j);
      cl_device_type type;
      clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
      if (type & CL_DEVICE_TYPE_DEFAULT)
        printf("CL_DEVICE_TYPE: %s\n", "CL_DEVICE_TYPE_DEFAULT");
      if (type & CL_DEVICE_TYPE_CPU)
        printf("CL_DEVICE_TYPE: %s\n", "CL_DEVICE_TYPE_CPU");
      if (type & CL_DEVICE_TYPE_GPU)
        printf("CL_DEVICE_TYPE: %s\n", "CL_DEVICE_TYPE_GPU");
      if (type & CL_DEVICE_TYPE_ACCELERATOR)
        printf("CL_DEVICE_TYPE: %s\n", "CL_DEVICE_TYPE_ACCELERATOR");
      //      if (type & CL_DEVICE_TYPE_CUSTOM)
      //        printf("CL_DEVICE_TYPE: %s\n", "CL_DEVICE_TYPE_CUSTOM");
      clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(buffer), buffer, NULL);
      printf("CL_DEVICE_NAME: %s\n", buffer);

      parseDeviceName(buffer);
//      std::vector<cl_context_properties> properties;

      if (gpu_type_ == GpuType ::ADRENO){
        printf("use anreno properties \n");
        properties_.push_back(0x40C2);
        properties_.push_back(0x40C3);
        properties_.push_back(0x40C9);
        properties_.push_back(0x40CA);
        properties_.push_back(0);
        properties_.reserve(5);
      }



      clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL);
      printf("CL_DEVICE_VENDOR: %s\n", buffer);
      cl_uint vendor_id;
      clGetDeviceInfo(device, CL_DEVICE_VENDOR_ID, sizeof(vendor_id),
                      &vendor_id, NULL);
      printf("CL_DEVICE_VENDOR_ID: %d\n", vendor_id);
      clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL);
      printf("CL_DEVICE_VERSION: %s\n", buffer);
      clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(buffer), buffer, NULL);
      printf("CL_DRIVER_VERSION: %s\n", buffer);
      //      clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION,
      //      sizeof(buffer),
      //                      buffer, NULL);
      //      printf("CL_DEVICE_OPENCL_C_VERSION: %s\n", buffer);
      clGetDeviceInfo(device, CL_DEVICE_PROFILE, sizeof(buffer), buffer, NULL);
      printf("CL_DEVICE_PROFILE: %s\n", buffer);
      clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(buffer), buffer,
                      NULL);
      printf("CL_DEVICE_EXTENSIONS: %s\n", buffer);
      //      printf("CL_DEVICE_BUILT_IN_KERNELS: %s\n",
      //             clGetDeviceInfo(device, CL_DEVICE_BUILT_IN_KERNELS,
      //             sizeof(buffer),
      //                             buffer, NULL) == CL_SUCCESS
      //                 ? buffer
      //                 : "UNSUPPORTED");
      clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
                      sizeof(max_compute_units_), &max_compute_units_, NULL);
      printf("CL_DEVICE_MAX_COMPUTE_UNITS: %u\n", max_compute_units_);
      cl_uint max_work_item_dimensions;
      clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                      sizeof(max_work_item_dimensions),
                      &max_work_item_dimensions, NULL);
      printf("CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: %u\n",
             max_work_item_dimensions);
      size_t *max_work_item_sizes =
          (size_t *)malloc(sizeof(size_t) * max_work_item_dimensions);
      clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                      sizeof(size_t) * max_work_item_dimensions,
                      max_work_item_sizes, NULL);
      printf("CL_DEVICE_MAX_WORK_ITEM_SIZES: ");
      for (size_t i = 0; i < max_work_item_dimensions; ++i)
        printf("%lu\t", max_work_item_sizes[i]);
      printf("\n");
      free(max_work_item_sizes);
      size_t max_work_group_size;
      clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                      sizeof(max_work_group_size), &max_work_group_size, NULL);
      printf("CL_DEVICE_MAX_WORK_GROUP_SIZE: %lu\n", max_work_group_size);
      cl_uint preferred_vector_width_char;
      clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR,
                      sizeof(preferred_vector_width_char),
                      &preferred_vector_width_char, NULL);
      printf("CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR: %u\n",
             preferred_vector_width_char);
      cl_uint preferred_vector_width_short;
      clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT,
                      sizeof(preferred_vector_width_short),
                      &preferred_vector_width_short, NULL);
      printf("CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT: %u\n",
             preferred_vector_width_short);
      cl_uint preferred_vector_width_int;
      clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT,
                      sizeof(preferred_vector_width_int),
                      &preferred_vector_width_int, NULL);
      printf("CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT: %u\n",
             preferred_vector_width_int);
      cl_uint preferred_vector_width_long;
      clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG,
                      sizeof(preferred_vector_width_long),
                      &preferred_vector_width_long, NULL);
      printf("CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG: %u\n",
             preferred_vector_width_long);
      cl_uint preferred_vector_width_float;
      clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,
                      sizeof(preferred_vector_width_float),
                      &preferred_vector_width_float, NULL);
      printf("CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT: %u\n",
             preferred_vector_width_float);
      cl_uint preferred_vector_width_double;
      clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
                      sizeof(preferred_vector_width_double),
                      &preferred_vector_width_double, NULL);
      printf("CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE: %u\n",
             preferred_vector_width_double);
      //      cl_uint preferred_vector_width_half;
      //      clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF,
      //                      sizeof(preferred_vector_width_half),
      //                      &preferred_vector_width_half, NULL);
      //      printf("CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF: %u\n",
      //             preferred_vector_width_half);
      //      cl_uint native_vector_width_char;
      //      clGetDeviceInfo(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR,
      //                      sizeof(native_vector_width_char),
      //                      &native_vector_width_char, NULL);
      //      printf("CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR: %u\n",
      //             native_vector_width_char);
      //      cl_uint native_vector_width_short;
      //      clGetDeviceInfo(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT,
      //                      sizeof(native_vector_width_short),
      //                      &native_vector_width_short, NULL);
      //      printf("CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT: %u\n",
      //             native_vector_width_short);
      //      cl_uint native_vector_width_int;
      //      clGetDeviceInfo(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_INT,
      //                      sizeof(native_vector_width_int),
      //                      &native_vector_width_int, NULL);
      //      printf("CL_DEVICE_NATIVE_VECTOR_WIDTH_INT: %u\n",
      //             native_vector_width_int);
      //      cl_uint native_vector_width_long;
      //      clGetDeviceInfo(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG,
      //                      sizeof(native_vector_width_long),
      //                      &native_vector_width_long, NULL);
      //      printf("CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG: %u\n",
      //             native_vector_width_long);
      //      cl_uint native_vector_width_float;
      //      clGetDeviceInfo(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT,
      //                      sizeof(native_vector_width_float),
      //                      &native_vector_width_float, NULL);
      //      printf("CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT: %u\n",
      //             native_vector_width_float);
      //      cl_uint native_vector_width_double;
      //      clGetDeviceInfo(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE,
      //                      sizeof(native_vector_width_double),
      //                      &native_vector_width_double, NULL);
      //      printf("CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE: %u\n",
      //             native_vector_width_double);
      //      cl_uint native_vector_width_half;
      //      clGetDeviceInfo(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF,
      //                      sizeof(native_vector_width_half),
      //                      &native_vector_width_half, NULL);
      //      printf("CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF: %u\n",
      //             native_vector_width_half);
      cl_uint max_clock_frequency;
      clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY,
                      sizeof(max_clock_frequency), &max_clock_frequency, NULL);
      printf("CL_DEVICE_MAX_CLOCK_FREQUENCY: %u MHz\n", max_clock_frequency);
      cl_uint address_bits;
      clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS, sizeof(address_bits),
                      &address_bits, NULL);
      printf("CL_DEVICE_ADDRESS_BITS: %u\n", address_bits);
      cl_ulong max_mem_alloc_size;
      clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                      sizeof(max_mem_alloc_size), &max_mem_alloc_size, NULL);
      printf("CL_DEVICE_MAX_MEM_ALLOC_SIZE: %lu B = %lu MB\n",
             static_cast<unsigned long>(max_mem_alloc_size),
             static_cast<unsigned long>(max_mem_alloc_size / 1048576));
      cl_bool image_support;
      clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof(image_support),
                      &image_support, NULL);
      printf("CL_DEVICE_IMAGE_SUPPORT: %u\n", image_support);
      size_t max_parameter_size;
      clGetDeviceInfo(device, CL_DEVICE_MAX_PARAMETER_SIZE,
                      sizeof(max_parameter_size), &max_parameter_size, NULL);
      printf("CL_DEVICE_MAX_PARAMETER_SIZE: %lu B\n", max_parameter_size);
      cl_device_mem_cache_type global_mem_cache_type;
      clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE,
                      sizeof(global_mem_cache_type), &global_mem_cache_type,
                      NULL);
      if (global_mem_cache_type == CL_NONE)
        printf("CL_DEVICE_GLOBAL_MEM_CACHE_TYPE: %s\n", "CL_NONE");
      if (global_mem_cache_type == CL_READ_ONLY_CACHE)
        printf("CL_DEVICE_GLOBAL_MEM_CACHE_TYPE: %s\n", "CL_READ_ONLY_CACHE");
      if (global_mem_cache_type == CL_READ_WRITE_CACHE)
        printf("CL_DEVICE_GLOBAL_MEM_CACHE_TYPE: %s\n", "CL_READ_WRITE_CACHE");
      cl_uint global_mem_cacheline_size;
      clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,
                      sizeof(global_mem_cacheline_size),
                      &global_mem_cacheline_size, NULL);
      printf("CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE: %u B\n",
             global_mem_cacheline_size);
      clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
                      sizeof(global_mem_cache_size_), &global_mem_cache_size_,
                      NULL);
      printf("CL_DEVICE_GLOBAL_MEM_CACHE_SIZE: %lu B = %lu KB\n",
             static_cast<unsigned long>(global_mem_cache_size_),
             static_cast<unsigned long>(global_mem_cache_size_ / 1024));
      cl_ulong global_mem_size;
      clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE,
                      sizeof(global_mem_size), &global_mem_size, NULL);
      printf("CL_DEVICE_GLOBAL_MEM_SIZE: %lu B = %lu MB\n",
             static_cast<unsigned long>(global_mem_size),
             static_cast<unsigned long>(global_mem_size / 1048576));
      cl_ulong max_constant_buffer_size;
      clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
                      sizeof(max_constant_buffer_size),
                      &max_constant_buffer_size, NULL);
      printf("CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: %lu B = %lu KB\n",
             static_cast<unsigned long>(max_constant_buffer_size),
             static_cast<unsigned long>(max_constant_buffer_size / 1024));
      cl_uint max_constant_args;
      clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_ARGS,
                      sizeof(max_constant_args), &max_constant_args, NULL);
      printf("CL_DEVICE_MAX_CONSTANT_ARGS: %u\n", max_constant_args);
      cl_device_local_mem_type local_mem_type;
      clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(local_mem_type),
                      &local_mem_type, NULL);
      if (local_mem_type == CL_NONE)
        printf("CL_DEVICE_LOCAL_MEM_TYPE: %s\n", "CL_NONE");
      if (local_mem_type == CL_LOCAL)
        printf("CL_DEVICE_LOCAL_MEM_TYPE: %s\n", "CL_LOCAL");
      if (local_mem_type == CL_GLOBAL)
        printf("CL_DEVICE_LOCAL_MEM_TYPE: %s\n", "CL_GLOBAL");
      cl_ulong local_mem_size;
      clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem_size),
                      &local_mem_size, NULL);
      printf("CL_DEVICE_LOCAL_MEM_SIZE: %lu B = %lu KB\n",
             static_cast<unsigned long>(local_mem_size),
             static_cast<unsigned long>(local_mem_size / 1024));
      cl_bool error_correction_support;
      clGetDeviceInfo(device, CL_DEVICE_ERROR_CORRECTION_SUPPORT,
                      sizeof(error_correction_support),
                      &error_correction_support, NULL);
      printf("CL_DEVICE_ERROR_CORRECTION_SUPPORT: %u\n",
             error_correction_support);
      cl_bool host_unified_memory;
      //      clGetDeviceInfo(device, CL_DEVICE_HOST_UNIFIED_MEMORY,
      //                      sizeof(host_unified_memory), &host_unified_memory,
      //                      NULL);
      //      printf("CL_DEVICE_HOST_UNIFIED_MEMORY: %u\n",
      //      host_unified_memory);
      size_t profiling_timer_resolution;
      clGetDeviceInfo(device, CL_DEVICE_PROFILING_TIMER_RESOLUTION,
                      sizeof(profiling_timer_resolution),
                      &profiling_timer_resolution, NULL);
      printf("CL_DEVICE_PROFILING_TIMER_RESOLUTION: %lu ns\n",
             profiling_timer_resolution);
      cl_bool endian_little;
      clGetDeviceInfo(device, CL_DEVICE_ENDIAN_LITTLE, sizeof(endian_little),
                      &endian_little, NULL);
      printf("CL_DEVICE_ENDIAN_LITTLE: %u\n", endian_little);
      cl_bool available;
      clGetDeviceInfo(device, CL_DEVICE_AVAILABLE, sizeof(available),
                      &available, NULL);
      printf("CL_DEVICE_AVAILABLE: %u\n", available);
      cl_bool compier_available;
      clGetDeviceInfo(device, CL_DEVICE_COMPILER_AVAILABLE,
                      sizeof(compier_available), &compier_available, NULL);
      printf("CL_DEVICE_COMPILER_AVAILABLE: %u\n", compier_available);
      //      cl_bool linker_available;
      //      clGetDeviceInfo(device, CL_DEVICE_LINKER_AVAILABLE,
      //                      sizeof(linker_available), &linker_available,
      //                      NULL);
      //      printf("CL_DEVICE_LINKER_AVAILABLE: %u\n", linker_available);
      cl_device_exec_capabilities exec_capabilities;
      clGetDeviceInfo(device, CL_DEVICE_EXECUTION_CAPABILITIES,
                      sizeof(exec_capabilities), &exec_capabilities, NULL);
      if (exec_capabilities & CL_EXEC_KERNEL)
        printf("CL_DEVICE_EXECUTION_CAPABILITIES: %s\n", "CL_EXEC_KERNEL");
      if (exec_capabilities & CL_EXEC_NATIVE_KERNEL)
        printf("CL_DEVICE_EXECUTION_CAPABILITIES: %s\n",
               "CL_EXEC_NATIVE_KERNEL");
      cl_command_queue_properties queue_properties;
      clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES,
                      sizeof(queue_properties), &queue_properties, NULL);
      if (queue_properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
        printf("CL_DEVICE_QUEUE_PROPERTIES: %s\n",
               "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE");
      if (queue_properties & CL_QUEUE_PROFILING_ENABLE)
        printf("CL_DEVICE_QUEUE_PROPERTIES: %s\n", "CL_QUEUE_PROFILING_ENABLE");
      //      size_t printf_buffer_size;
      //      clGetDeviceInfo(device, CL_DEVICE_PRINTF_BUFFER_SIZE,
      //                      sizeof(printf_buffer_size), &printf_buffer_size,
      //                      NULL);
      //      printf("CL_DEVICE_PRINTF_BUFFER_SIZE: %lu B = %lu KB\n",
      //             printf_buffer_size, printf_buffer_size / 1024);
      //      cl_bool preferred_interop_user_sync;
      //      clGetDeviceInfo(device, CL_DEVICE_PREFERRED_INTEROP_USER_SYNC,
      //                      sizeof(preferred_interop_user_sync),
      //                      &preferred_interop_user_sync, NULL);
      //      printf("CL_DEVICE_PREFERRED_INTEROP_USER_SYNC: %u\n",
      //             preferred_interop_user_sync);
      //			cl_device_id parent_device;
      //			 clGetDeviceInfo(device,
      // CL_DEVICE_PARENT_DEVICE, sizeof(parent_device), &parent_device, NULL);
      //			printf("CL_DEVICE_PARENT_DEVICE: %u\n",
      // parent_device);
      //      cl_uint reference_count;
      //      clGetDeviceInfo(device, CL_DEVICE_REFERENCE_COUNT,
      //                      sizeof(reference_count), &reference_count, NULL);
      //      printf("CL_DEVICE_REFERENCE_COUNT: %u\n", reference_count);
      //      printf("\n");
    }
    free(devices);
  }
  free(platforms);
}
void CLEngine::parseDeviceName(char *buffer) {
  constexpr const char *kQualcommAdrenoGPUStr = "QUALCOMM Adreno(TM)";
  constexpr const char *kMaliGPUStr = "Mali";
  std::string name(buffer);
  if (name == kQualcommAdrenoGPUStr) {
    gpu_type_ = GpuType::ADRENO;
  } else if (name.find(kMaliGPUStr) != std::string::npos) {
    gpu_type_ = GpuType::MALI;
  } else {
    gpu_type_ = GpuType::OTHERS;
  }
}

// std::unique_ptr<_cl_kernel, clKernel_deleter> CLEngine::GSetKernel(
//    const std::string &kernel_name) {
//  std::unique_ptr<_cl_kernel, clKernel_deleter> kernel(
//      clCreateKernel(program_.get(), kernel_name.c_str(), NULL);
//  return std::move(kernel);
//}
//
// bool CLEngine::SetClCommandQueue() {
//  cl_int status;
//  command_queue_.reset(
//          clCreateCommandQueue(context_.get(), devices_[0], 0, &status);
//  return true;
//}

// bool CLEngine::SetClContext() {
//  context_.reset(clCreateContext(NULL, 1, devices_, NULL, NULL, NULL);
//  return true;
//}

// bool CLEngine::LoadKernelFromFile(const char *kernel_file) {
//  size_t size;
//  char *str;
//  std::fstream f(kernel_file, (std::fstream::in | std::fstream::binary);
//
//  if (!f.is_open() {
//    return false;
//  }
//
//  size_t fileSize;
//  f.seekg(0, std::fstream::end);
//  size = fileSize = (size_t)f.tellg();
//  f.seekg(0, std::fstream::beg);
//  str = new char[size + 1];
//  if (!str) {
//    f.close();
//    return 0;
//  }
//
//  f.read(str, fileSize);
//  f.close();
//  str[size] = '\0';
//  const char *source = str;
//  size_t sourceSize[] = {strlen(source)};
//  program_.reset(
//      clCreateProgramWithSource(context_.get(), 1, &source, sourceSize,
//      NULL);
//  return true;
//}

}  // namespace framework
}  // namespace paddle_mobile
