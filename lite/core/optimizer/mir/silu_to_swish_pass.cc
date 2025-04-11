// lite/core/optimizer/mir/silu_to_swish_pass.cc
#include "lite/core/optimizer/mir/silu_to_swish_pass.h"
#include <memory>
#include <string>
#include <vector>
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

void SiluToSwishPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  // 遍历所有节点
  for (auto& node : graph->mutable_nodes()) {
    if (node.IsStmt()) {
      // 查找 silu 节点
      auto* stmt = node.stmt();
      auto op_type = stmt->op_type();
      if (op_type == "silu") {
        // 将 silu 更改为 swish
        auto* op_desc = stmt->mutable_op_info();
        op_desc->SetType("swish");
        LOG(INFO) << "silu operator replaced to swish";
      }
    }
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(silu_to_swish_pass, paddle::lite::mir::SiluToSwishPass)
    .BindTargets({TARGET(kARM),
                 TARGET(kX86),
                 TARGET(kOpenCL)});