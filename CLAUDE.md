# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

### Android Build (Primary)

This project focuses on Android/iOS platforms. Use the `build_android.sh` script:

```bash
# Android ARM64 with OpenCL (推荐使用 clang)
./lite/tools/build_android.sh --arch=armv8 \
    --toolchain=clang \
    --android_stl=c++_shared \
    --with_java=ON \
    --with_cv=OFF \
    --with_log=OFF \
    --with_extra=ON \
    --with_exception=ON \
    --with_static_lib=ON \
    --with_opencl=ON
```

**Common options:**
- `--arch=armv8|armv7` - ARM architecture (default: armv8)
- `--toolchain=clang|gcc` - Compiler (推荐使用 clang)
- `--android_stl=c++_shared|c++_static` - Android STL type
- `--with_java=ON|OFF` - Build Java JNI (需要 Java)
- `--with_cv=ON|OFF` - Include OpenCV functions
- `--with_log=ON|OFF` - Enable logging
- `--with_opencl=ON|OFF` - Enable OpenCL GPU support

### macOS Build (ARM / x86)

使用 `build_macos.sh` 在 macOS 上编译 ARM 或 x86 库，可用于本地验证无需 Android 设备。

```bash
# 环境准备
switch_cmake 3.22.1
export PATH="/Users/baidu/miniforge3/bin:$PATH"

# ARM64 编译 (带 OpenCL)
./lite/tools/build_macos.sh --with_opencl=ON --with_extra=ON --with_exception=ON arm64

# ARM64 benchmark 编译 (自动开启 extra/exception，关闭 log)
./lite/tools/build_macos.sh --with_benchmark=ON --with_opencl=ON arm64

# x86 编译
./lite/tools/build_macos.sh x86
```

**常用选项 (`build_macos.sh`):**
- `arm64` / `x86` - 目标架构 (位置参数，放最后)
- `--with_opencl=ON|OFF` - OpenCL GPU 支持 (默认 OFF)
- `--with_benchmark=ON|OFF` - 编译 benchmark 二进制 (默认 OFF，开启后自动设置 extra=ON, exception=ON)
- `--with_log=ON|OFF` - 日志输出 (默认 ON，benchmark 模式下默认 OFF)
- `--with_cv=ON|OFF` - OpenCV 函数 (默认 OFF)
- `--with_extra=ON|OFF` - 额外算子 (序列模型如 OCR/NLP) (默认 OFF)
- `--with_exception=ON|OFF` - 异常支持 (默认 OFF)
- `--with_arm82_fp16=ON|OFF` - FP16 内核 (默认 OFF，开启后强制使用 clang)
- `--with_testing=ON|OFF` - 编译单元测试 (默认 OFF)

**编译输出目录:**
- ARM64: `build.macos.armmacos.armv8/` (带 OpenCL 时为 `build.macos.armmacos.armv8.opencl/`)
- x86: `build.lite.x86/`

**macOS build notes:**
- 需要 `ulimit -n 1024` 增加文件描述符限制 (脚本内已自动设置)
- cmake 3.22.1 可用 (`switch_cmake 3.22.1`)，cmake 3.10.3 不可用，cmake 4.x 不兼容

### Build Output

编译成功后，输出在 `build.lite.android.armv8.clang/inference_lite_lib.android.armv8.opencl/cxx/`:
- `libpaddle_api_light_bundled.a` - 静态库
- `libpaddle_light_api_shared.so` - 动态库

### Model Optimization Tool (opt)

```bash
# 在 macOS 上构建 opt 工具
switch_cmake 3.22.1
./lite/tools/build_macos.sh build_optimize_tool
```

**opt 构建注意事项:**
- 如果 third-party 子模块有问题，可尝试：删除 `third-party/` 目录，使用下载的 `third-party-651c7c4.tar.gz` 解压替代；同时在 git 上恢复 third-party 相关文件。但对比下载资源与 git submodule 内容，这两步可能非必须。

### 更新 FlatBuffers 生成头文件 (update_fbs)

当修改了 `.fbs` schema 文件后，需要重新生成对应的 `*_generated.h` 头文件并提交到 `third-party/flatbuffers/pre-build/`。

**FBS schema 文件列表:**
- `lite/model_parser/flatbuffers/framework.fbs` - 模型框架 schema (ProgramDesc, BlockDesc, OpDesc 等)
- `lite/model_parser/flatbuffers/param.fbs` - 模型参数 schema (CombinedParamsDesc, ParamDesc)
- `lite/backends/opencl/utils/cache.fbs` - OpenCL 二进制内核缓存 schema
- `lite/backends/opencl/utils/tune_cache.fbs` - OpenCL 调优缓存 schema

**更新步骤:**

```bash
# Step 1: 编辑 .fbs schema 文件

# Step 2: 运行更新脚本 (会自动下载 flatc 编译器并重新生成头文件)
switch_cmake 3.22.1
cd /Users/baidu/workspace/github/Paddle-Lite
./third-party/flatbuffers/update_fbs.sh

# Step 3: 提交更新后的预编译头文件
git add -f third-party/flatbuffers/pre-build
git commit -m "update flatbuffers pre-build headers"
```

**工作原理:**
- `update_fbs.sh` 使用 cmake 选项 `-DLITE_UPDATE_FBS_HEAD=ON` 触发从源码编译 flatc (v1.12.0)，然后调用 `make fbs_headers` 生成 `*_generated.h`
- 生成的头文件和 flatbuffers 运行时头文件一起拷贝到 `third-party/flatbuffers/pre-build/`
- 正常编译时 (`LITE_UPDATE_FBS_HEAD=OFF`，默认)，CMake 的 `fbs_headers` target 直接从 `pre-build/` 拷贝预编译头文件到源码目录，无需编译 flatc
- 构建输出目录: `build.lite.flatbuffer/`

## Environment Setup

```bash
# Activate Python (conda base)
conda activate base

# Switch to cmake 3.22.1 (3.10.3 不可用，4.x 不兼容)
switch_cmake 3.22.1

# Set Android NDK (如果未设置)
export ANDROID_NDK=/opt/android-ndk-r27d
```

## Architecture Overview

Paddle Lite 是轻量级深度学习推理框架，专为移动端优化。

### 核心抽象 (`lite/core/`)

**OpLite** (`op_lite.h`): 算子基类
- `AttachImpl()` - 绑定运行时环境
- `InferShape()` - 推断输出形状
- `AttachKernel()` - 绑定内核
- `Run()` - 执行算子

**KernelLite** (`kernel.h`): 内核基类，由 (Target, Precision, Layout) 三元组标识

**Place**: 组合 target/precision/layout

**算子/内核注册** (`op_registry.h`):
```cpp
REGISTER_LITE_KERNEL(
    conv2d, kARM, kFloat, kNCHW, Conv2DCompute, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
```

### 目录结构

- `lite/operators/` - 算子定义
- `lite/kernels/` - 内核实现
  - `arm/` - ARM CPU 内核 (NEON, dotprod, fp16)
  - `opencl/` - OpenCL GPU 内核
  - `host/` - Host CPU 后备内核
- `lite/backends/` - 后端工具和数学库
- `lite/core/optimizer/mir/` - 图优化 Pass
- `lite/api/` - C++, Java, Python API

### 后端支持

- **ARM CPU**: NEON, dotprod (v8.2), fp16, int8
- **OpenCL**: Buffer 和 Image 格式内核
- **Metal**: Apple Metal (iOS)
- **X86**: AVX/SSE, MKL (仅测试)

### MIR 优化 Pass

- `StaticKernelPickPass` - 选择最优内核
- `TypeTargetCastPass`, `TypePrecisionCastPass`, `TypeLayoutCastPass` - 类型转换
- `VariablePlaceInferencePass` - 变量布局推断
- `MemoryOptimizePass` - 内存优化

### 常见模式

**添加新算子:**
1. `operators/new_op.cc` 创建 `OpLite` 子类
2. `REGISTER_LITE_OP(new_op, NewOp)`

**添加新内核:**
1. `kernels/<backend>/new_op_compute.cc` 创建 `KernelLite` 子类
2. 实现 `Run()` 方法
3. `REGISTER_LITE_KERNEL(new_op, target, precision, layout, NewOpCompute, alias)`

**添加优化 Pass:**
1. `lite/core/optimizer/mir/new_pass.cc` 继承 `mir::StmtPass`
2. 实现 `Apply()` 方法

## Debugging Tips

- `--with_log=ON` 启用日志
- `--with_exception=ON` 启用异常（需要 clang for armv7）
- macOS 编译用 `--with_java=OFF`
