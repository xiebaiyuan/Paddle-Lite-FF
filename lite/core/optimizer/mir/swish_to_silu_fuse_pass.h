//
// Created by baidu on 2025/4/10.
//
// lite/core/optimizer/mir/swish_to_silu_fuse_pass.h
#pragma once

#include <memory>
#include <string>
#include "lite/core/optimizer/mir/pass.h"

namespace paddle {
namespace lite {
namespace mir {

/*
 * 将 Swish 转换为 SiLU 的优化 Pass
 * Swish(x) = x * sigmoid(x) 和 SiLU(x) = x * sigmoid(x) 在数学上是等价的
 */
class SwishToSiluFusePass : public ProgramPass {
public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle