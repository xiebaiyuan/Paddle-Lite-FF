// lite/core/optimizer/mir/lite_silu_convert_pass.h
#pragma once

#include <memory>
#include <string>
#include "lite/core/optimizer/mir/pass.h"

namespace paddle {
namespace lite {
namespace mir {

/*
 * Convert the silu op to sigmoid and elementwise_mul ops.
 * For example:
 *   - before:
 *     silu(X) -> Out
 *   - after:
 *     sigmoid(X) -> sigmoid_out
 *     elementwise_mul(X, sigmoid_out) -> Out
 */
class LiteSiluConvertPass : public ProgramPass {
public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle