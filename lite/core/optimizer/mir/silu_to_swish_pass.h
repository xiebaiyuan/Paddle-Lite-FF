// lite/core/optimizer/mir/silu_to_swish_pass.h
#pragma once

#include <memory>
#include <string>
#include "lite/core/optimizer/mir/pass.h"

namespace paddle {
namespace lite {
namespace mir {

/*
 * 将 SiLU 转换为 Swish 的优化 Pass
 * SiLU(x) = x * sigmoid(x) 和 Swish(x) = x * sigmoid(x) 在数学上是等价的
 */
class SiluToSwishPass : public ProgramPass {
public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle