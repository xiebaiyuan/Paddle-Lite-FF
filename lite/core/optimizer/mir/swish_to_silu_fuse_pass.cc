// lite/core/optimizer/mir/swish_to_silu_fuse_pass.cc
#include "lite/core/optimizer/mir/swish_to_silu_fuse_pass.h"
#include <memory>
#include <string>
#include <vector>
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

void SwishToSiluFusePass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  // 遍历所有节点
  for (auto& node : graph->mutable_nodes()) {
    if (node.IsStmt()) {
      // 查找 swish 节点
      auto* stmt = node.stmt();
      auto op_type = stmt->op_type();
      if (op_type == "swish") {
        // 将 swish 更改为 silu
        auto* op_desc = stmt->mutable_op_info();
        op_desc->SetType("silu");
        LOG(INFO) << "swish operator replaced to silu";
      }
    }
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(swish_to_silu_fuse_pass, paddle::lite::mir::SwishToSiluFusePass)
    .BindTargets({TARGET(kARM),
                 TARGET(kX86),
                 TARGET(kOpenCL)});