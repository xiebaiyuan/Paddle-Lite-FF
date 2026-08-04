# CLAUDE.md

> ## ⚠️ 本机是 ARM Mac（Apple Silicon）
>
> **本开发机是 ARM (arm64) Mac，禁止使用任何 x86 架构的工具、二进制、库或内容。**
>
> - 不要下载/调用 x86 版本的工具链、预编译库、wheel、conda 包。
> - 不要执行 x86-only 的编译/运行路径；本机验证一律走 `arm64` 目标。
> - 文档中出现的 `x86` 相关内容（如 `build_macos.sh x86`、`build.lite.x86/`）仅作仓库参考，**本机不适用**。
> - 可用 `uname -m` 确认架构（应返回 `arm64`）。

# 全局约定
- **Obsidian**：Vault `/Users/baidu/AI_DOC/`，决策/架构/bug 归档到 `obsidian-docs/`（软链）
- **Archive**：删除文件用 `trash`（不用 `rm`）；产物归档用 `swe-archive` skill
- **Python**：用 `uv` 管理环境（如 `PADDLEOCR/.../pp3_env`）
- @RTK.md

# 构建

> **本机禁止一切 x86 构建**：`build.lite.x86`/`build.macos.local_test` 等 x86 产物在本机不可用（`lite/backends/x86/math/sse/` 是 x86 专属 SSE 数学库，ARM Mac 编译不过）。测试一律走 ARM 测试构建（见下）。

## 环境准备（每次构建前）
```bash
switch_cmake 3.22.1    # 3.10.3 不可用，4.x 不兼容
# Bash 工具里每次是新 shell，switch_cmake 是 shell 函数不跨 shell 持久化，需显式：
export PATH="/opt/homebrew/Cellar/cmake/3.22.1/bin:$PATH"
ulimit -n 1024
```

## macOS ARM 库（本地验证推理）
```bash
./lite/tools/build_macos.sh --with_opencl=ON --with_extra=ON --with_exception=ON arm64
```
- 产物目录：`build.macos.armmacos.armv8/`（带 OpenCL 时 `.opencl/` 后缀）
- 推理测试：`build.macos.armmacos.armv8/lite/api/test_model_bin`
  ```
  test_model_bin --model_dir=<nb> --use_optimize_nb=true --backend=arm_cpu \
    --input_shape=1,3,640,640 --in_txt=<prefix_> --out_txt=<out_dir>/out_
  ```
  `--in_txt` 读 `prefix_1.txt`（原始 float 二进制）

## Android（主目标）
```bash
./lite/tools/build_android.sh --arch=armv8 --toolchain=clang --android_stl=c++_shared \
    --with_java=ON --with_cv=OFF --with_log=OFF --with_extra=ON --with_exception=ON \
    --with_static_lib=ON --with_opencl=ON
```
产物：`build.lite.android.armv8.clang/inference_lite_lib.android.armv8.opencl/cxx/`

## opt 工具（模型转换）
```bash
./lite/tools/build_macos.sh build_optimize_tool   # 产物 build.opt/lite/api/opt
# 转换: opt --model_dir=<pdmodel_dir> --optimize_out=<out> --valid_targets=arm --optimize_out_type=naive_buffer
```
- 部署模式：**opt → nb → 部署**（老模型用新 opt 重新转换）
- third-party 子模块异常时：删 `third-party/`，`build_macos.sh` 会自动从 `https://paddlelite-data.bj.bcebos.com/third_party_libs/` 下载 `third-party-651c7c4.tar.gz` 重建（脚本内置逻辑，无需手动找备份）

## 更新 FlatBuffers 头文件（update_fbs）
改 `.fbs` 后：`./third-party/flatbuffers/update_fbs.sh` 重新生成 `*_generated.h` 到 `third-party/flatbuffers/pre-build/`，`git add -f` 提交。

## 测试构建（跑 gtest）
- **本机一律走 ARM 测试构建**（x86 `build.macos.local_test/` 已废弃，SSE 数学库编译不过）：
  ```bash
  ./lite/tools/build_macos.sh --with_testing=ON arm64
  # 产物 build.macos.armmacos.armv8.test/，单测二进制在对应源码目录下，如：
  # build.macos.armmacos.armv8.test/lite/core/optimizer/mir/fusion/test_conv_activation_fuse_pass
  ```
- fusion pass 单测在 `lite/core/optimizer/mir/fusion/`（详见下）
- ccache 已全量启用（`brew install ccache`），清产物重建测试约 10s

# 架构概览

## 核心抽象（`lite/core/`）
- **OpLite** (`op_lite.h`)：`AttachImpl` 绑运行时、`InferShape` 推断形状、`Run` 执行
- **KernelLite** (`kernel.h`)：由 (Target, Precision, Layout) 三元组标识
- **注册** (`op_registry.h`)：
  ```cpp
  REGISTER_LITE_KERNEL(conv2d, kARM, kFloat, kNCHW, Conv2DCompute, def)
      .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
      .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kARM))})
      .Finalize();
  ```

## 目录
- `lite/operators/` 算子定义 ｜ `lite/kernels/{arm,opencl,host}/` 内核 ｜ `lite/backends/` 数学库
- `lite/core/optimizer/mir/` 图优化 Pass ｜ `lite/api/` API

## Conv kernel 选择与 activation dispatch（本次沉淀，改 conv 必读）
- **kernel 选择**（`lite/kernels/arm/conv_compute.cc` PrepareForRun）：
  - 3x3 s1 groups=1 no-dilation → **WinogradConv**
  - 3x3 s2 groups=1 小通道 → DirectConv；1x1 → GemmLikeConv；depthwise → DepthwiseConv
- **activation dispatch 两条路径**：
  1. `write_to_output_c4_fp32`（`lite/backends/arm/math/conv_block_utils.h`）— winograd/direct3x3s2/dw5x5 写回统一走这里，**kGelu 已加 C 分支**
  2. kernel Run 层后处理（`lite/kernels/arm/conv_gelu_act.h`）— gemmlike/conv_transpose/depthwise3x3，`UnsetGeluForConvMath` 清 has_active 避免 asm LOG(FATAL) + `ApplyGeluIfFused` 逐位一致补 gelu
- 新增 conv 激活融合：kernel 支持后，`conv_activation_fuse_pass` act_types 加字符串即可（图侧）＋ `conv_op.h`/`subgraph_matcher` 属性透传

## MIR fusion pass
- 图优化在 `lite/core/optimizer/mir/fusion/`，FuseBase pattern 模式（`pattern_matcher_high_api.h`）
- pass 注册：`optimizer.cc` 加 pass 名 + `lite/api/paddle_use_passes.h` 加 USE_MIR_PASS
- **x2paddle 产物陷阱**：rec 类模型（x2paddle 转）含大量 `assign` 共享变量"寄存器池"（node_0/node_1/node_5 被 100+ 算子共享），**破坏 fusion pattern 的 SSA 封闭性** → conv+act 类融合不可行（det 无 assign 正常）。验证：dot 图统计 assign 数 + 共享变量 producer/consumer
- 单测：`fusion_pass_test_util.h/.cc`（ProgramDesc → SSAGraph）+ `*_fuse_pass_test.cc`

## 常用调试
- 编译带 `--with_log=ON` / `--with_exception=ON`（armv7 需 clang）
- 转换产物分析：`opt --optimized_nb_model_path=<nb> --visualization_file_output_path=<dir>` 出 `Block_0.dot`，统计 op 节点
- 模型位置：`PADDLEOCR/deploy/ppocr-android/model-convert/paddle26_models/{det,medium_rec,tiny_rec}/inference_model/`
