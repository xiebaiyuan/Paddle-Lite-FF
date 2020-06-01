// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>
#include <memory>
#include <random>
#include "lite/backends/opencl/target_wrapper.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/opencl/test_helper.h"

#define FP16_MAX_DIFF (5e-1)

namespace paddle {
namespace lite {

void pool_avg(const int padding_height,
              const int padding_width,
              const int stride_height,
              const int stride_width,
              const int ksize_height,
              const int ksize_width,
              const float* input_data,
              const DDim& in_dim,
              float* output_data,
              const DDim& out_dim) {
  const int batch_size = in_dim[0];
  const int input_height = in_dim[2];
  const int input_width = in_dim[3];
  const int output_channels = out_dim[1];
  const int output_height = out_dim[2];
  const int output_width = out_dim[3];

  const size_t input_spatial_size = input_height * input_width;
  const size_t output_spatial_size = output_height * output_width;

  for (int i = 0; i < batch_size; i++) {
    for (int c = 0; c < output_channels; ++c) {
      int channel = i * output_channels + c;
      const float* input_ptr = input_data + channel * input_spatial_size;
      float* output_ptr = output_data + channel * output_spatial_size;

      for (int ph = 0; ph < output_height; ++ph) {
        int hstart = ph * stride_height - padding_height;
        int hend = std::min(hstart + ksize_height, input_height);
        hstart = std::max(hstart, 0);
        for (int pw = 0; pw < output_width; ++pw) {
          int wstart = pw * stride_width - padding_width;
          int wend = std::min(wstart + ksize_width, input_width);
          wstart = std::max(wstart, 0);

          float val = 0.f;
          int count = 0;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              val += input_ptr[h * input_width + w];
              ++count;
            }
          }
          output_ptr[ph * output_width + pw] =
              (count > 0) ? val * (1.f / count) : 0.f;
        }
      }
    }
  }
}

TEST(pixel_shuffle_image2d, compute) {
  LOG(INFO) << "create kernel ...";
  auto kernels = KernelRegistry::Global().Create("pixel_shuffle",
                                                 TARGET(kOpenCL),
                                                 PRECISION(kFP16),
                                                 DATALAYOUT(kImageDefault));
  ASSERT_FALSE(kernels.empty());

  const int INPUT_N = 1;
  const int INPUT_C = 4;
  const int INPUT_H = 2;
  const int INPUT_W = 2;
  const int UPSCALE_FACTOR = 2;

  auto kernel = std::move(kernels.front());

  LOG(INFO) << "prepare to test kernel ====> " << kernel->doc();

  lite::Tensor x, out;
  operators::PixelShuffleParam param;
  param.x = &x;
  param.output = &out;
  param.upscale_factor = UPSCALE_FACTOR;

  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();

  kernel->SetParam(param);
  std::unique_ptr<KernelContext> pixel_shuffle_context(new KernelContext);
  context->As<OpenCLContext>().CopySharedTo(
      &(pixel_shuffle_context->As<OpenCLContext>()));

  kernel->SetContext(std::move(pixel_shuffle_context));

  const DDim in_dim =
      DDim(std::vector<DDim::value_type>{INPUT_N, INPUT_C, INPUT_H, INPUT_W});
  const DDim out_dim = DDim(
      std::vector<DDim::value_type>{INPUT_N,
                                    INPUT_C / UPSCALE_FACTOR / UPSCALE_FACTOR,
                                    INPUT_H * UPSCALE_FACTOR,
                                    INPUT_W * UPSCALE_FACTOR});
  LOG(INFO) << "in_dim: " << in_dim;
  LOG(INFO) << "UPSCALE_FACTOR: " << UPSCALE_FACTOR;
  LOG(INFO) << "out_dim: " << out_dim;

  x.Resize(in_dim);
  out.Resize(out_dim);

  std::default_random_engine engine;
  std::uniform_real_distribution<float> dist(-2, 2);
  std::vector<float> input_v(INPUT_N * INPUT_C * INPUT_H * INPUT_W);

  int index = 0;
  for (auto& i : input_v) {
    i = index++;
  }
  VLOG(1) << "input_v ..... ";
  for (size_t i = 0; i < input_v.size(); i++) {
    VLOG(10) << input_v[i];
  }

  LOG(INFO) << "prepare input";
  CLImageConverterDefault* default_converter = new CLImageConverterDefault();
  DDim x_image_shape = default_converter->InitImageDimInfoWith(in_dim);
  LOG(INFO) << "x_image_shape = " << x_image_shape[0] << " "
            << x_image_shape[1];
  std::vector<half_t> x_image_data(x_image_shape.production() * 4);  // 4 : RGBA
  default_converter->NCHWToImage(input_v.data(), x_image_data.data(), in_dim);
  auto* x_image = x.mutable_data<half_t, cl::Image2D>(
      x_image_shape[0], x_image_shape[1], x_image_data.data());
  VLOG(1) << "x_image_data ..... ";
  for (size_t i = 0; i < x_image_data.size(); i++) {
    VLOG(10) << Half2Float(x_image_data[i]);
  }
  DDim out_image_shape = default_converter->InitImageDimInfoWith(out_dim);
  LOG(INFO) << "out_image_shape = " << out_image_shape[0] << " "
            << out_image_shape[1];
  auto* out_image = out.mutable_data<half_t, cl::Image2D>(out_image_shape[0],
                                                          out_image_shape[1]);
  kernel->Launch();
  CLRuntime::Global()->command_queue().finish();
#if 0
  auto GetCurrentUS = []() -> double {
    struct timeval time;
    gettimeofday(&time, NULL);
    return 1e+6 * time.tv_sec + time.tv_usec;
  };
  const int profile_times = 10000;
  long sum_time = 0;
  for (size_t i = 0; i < profile_times; i++) {
    auto begin_time_inner = GetCurrentUS();

    kernel->Launch();
    CLRuntime::Global()->command_queue().finish();
    auto end_time_inner = GetCurrentUS();
    int delta = end_time_inner - begin_time_inner;
    sum_time += delta;
    LOG(INFO) << "pool avg : " << i + 1 << "/" << profile_times << "  =  "
              << (delta);
  }

  LOG(INFO) << "pool avg avg time : " << sum_time / profile_times;
#endif
  std::unique_ptr<float[]> out_ref(new float[out_dim.production()]);
  // pool_avg(0, 0, 2, 2, 7, 7, input_v.data(), in_dim, out_ref.get(), out_dim);
  std::vector<float> out_data_v{
      0, 4, 1, 5, 8, 12, 9, 13, 2, 6, 3, 7, 10, 14, 11, 15};

  const size_t cl_image2d_row_pitch{0};
  const size_t cl_image2d_slice_pitch{0};
  half_t* out_image_data = new half_t[out_image_shape.production() * 4];
  TargetWrapperCL::ImgcpySync(out_image_data,
                              out_image,
                              out_image_shape[0],
                              out_image_shape[1],
                              cl_image2d_row_pitch,
                              cl_image2d_slice_pitch,
                              IoDirection::DtoH);
  VLOG(1) << "out_image_data ..... ";
  for (size_t i = 0; i < out_image_shape.production() * 4; i++) {
    VLOG(10) << Half2Float(out_image_data[i]);
  }
  float* out_data = new float[out_image_shape.production() * 4];
  default_converter->ImageToNCHW(
      out_image_data, out_data, out_image_shape, out_dim);

  VLOG(1) << "out_data ..... ";
  for (int i = 0; i < out_dim.production(); i++) {
    VLOG(10) << out_data[i];
  }

  for (int i = 0; i < out_dim.production(); i++) {
    auto abs_diff = abs(out_data[i] - out_ref[i]);
    auto relative_diff = COMPUTE_RELATIVE_DIFF(out_data[i], out_ref[i]);
    EXPECT_EQ((relative_diff <= FP16_MAX_DIFF) || (abs_diff <= FP16_MAX_DIFF),
              true);
    if ((relative_diff > FP16_MAX_DIFF) && (abs_diff > FP16_MAX_DIFF)) {
      LOG(ERROR) << "error idx:" << i << " out_data[" << i
                 << "]:" << out_data[i] << " "
                                           "out_ref["
                 << i << "]:" << out_ref[i] << " abs_diff:" << abs_diff
                 << " relative_diff:" << relative_diff
                 << " FP16_MAX_DIFF:" << FP16_MAX_DIFF;
    }
  }
}

}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(pixel_shuffle, kOpenCL, kFP16, kImageDefault, image2d);
