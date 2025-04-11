// lite/core/optimizer/mir/lite_silu_convert_pass.cc
#include "lite/core/optimizer/mir/lite_silu_convert_pass.h"
#include <memory>
#include <vector>
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

void LiteSiluConvertPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  LOG(INFO) << "Applying LiteSiluConvertPass...";
  // 存储需要删除的节点，避免在遍历过程中直接修改图结构
  std::vector<mir::Node*> nodes_to_remove;

  // 遍历图中所有节点
  for (auto* node : graph->StmtTopologicalOrder()) {
    if (!node->IsStmt()) continue;
    auto* stmt = node->stmt();
    if (!stmt) {
      LOG(WARNING) << "Null stmt found, skipping";
      continue;
    }
    auto* op_info = stmt->op_info();
    if (!op_info) {
      LOG(WARNING) << "Null op_info found, skipping";
      continue;
    }
    if (op_info->Type() != "silu") continue;

    LOG(INFO) << "Found silu op, converting to sigmoid + elementwise_mul";

    // 获取silu的输入输出
    auto input_name = op_info->Input("X").front();
    auto output_name = op_info->Output("Out").front();
    LOG(INFO) << "Silu input: " << input_name << ", output: " << output_name;

    // 创建中间输出名称
    auto sigmoid_output_name = input_name + ".sigmoid";

    // 获取相关节点
    auto* input_node = graph->RetrieveArgument(input_name);
    auto* output_node = graph->RetrieveArgument(output_name);
    if (!input_node) {
      LOG(WARNING) << "Input node " << input_name << " not found, skipping";
      continue;
    }
    if (!output_node) {
      LOG(WARNING) << "Output node " << output_name << " not found, skipping";
      continue;
    }

    // 记录待删除节点（确认输入输出都存在后再添加）
    nodes_to_remove.push_back(node);

    // 创建sigmoid op节点
    auto* sigmoid_op_node = graph->NewInstructNode();
    auto sigmoid_op_desc = *op_info;
    sigmoid_op_desc.SetType("sigmoid");
    sigmoid_op_desc.SetInput("X", {input_name});
    sigmoid_op_desc.SetOutput("Out", {sigmoid_output_name});

    // 确保内核设置正确
    try {
      sigmoid_op_node->AsStmt().ResetOp(sigmoid_op_desc, graph->valid_places());
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to create sigmoid op: " << e.what() << ", skipping";
      continue;
    }

    // 创建sigmoid输出变量节点
    auto* sigmoid_out = graph->NewArgumentNode(sigmoid_output_name);
    sigmoid_out->AsArg().name = sigmoid_output_name;

    // 创建elementwise_mul op节点
    auto* mul_op_node = graph->NewInstructNode();
    auto mul_op_desc = *op_info;
    mul_op_desc.SetType("elementwise_mul");
    mul_op_desc.SetInput("X", {input_name});
    mul_op_desc.SetInput("Y", {sigmoid_output_name});
    mul_op_desc.SetOutput("Out", {output_name});

    // 确保内核设置正确
    try {
      mul_op_node->AsStmt().ResetOp(mul_op_desc, graph->valid_places());
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to create elementwise_mul op: " << e.what() << ", skipping";
      // 清理之前创建的节点以避免内存泄漏
      graph->RemoveNode(sigmoid_op_node);
      graph->RemoveNode(sigmoid_out);
      continue;
    }

    LOG(INFO) << "Successfully created new ops, now updating graph connections";

    // 先断开原始连接 - 优先处理输出连接
    for (auto* out : node->outlinks) {
      LOG(INFO) << "Removing link: " << node->AsStmt().op_info()->Type()
                << " -> " << (out->IsArg() ? out->AsArg().name : out->AsStmt().op_info()->Type());
      RemoveDirectedLink(node, out);
    }

    // 再处理输入连接
    for (auto* in : node->inlinks) {
      LOG(INFO) << "Removing link: "
                << (in->IsArg() ? in->AsArg().name : in->AsStmt().op_info()->Type())
                << " -> " << node->AsStmt().op_info()->Type();
      RemoveDirectedLink(in, node);
    }

    // 建立新的图连接
    LOG(INFO) << "Creating new links in graph";
    DirectedLink(input_node, sigmoid_op_node);
    DirectedLink(sigmoid_op_node, sigmoid_out);
    DirectedLink(sigmoid_out, mul_op_node);
    DirectedLink(input_node, mul_op_node);
    DirectedLink(mul_op_node, output_node);
  }

  // 在遍历完成后，删除所有标记的节点
  for (auto* node : nodes_to_remove) {
    LOG(INFO) << "Removing original silu node from graph";
    graph->RemoveNode(node);
  }

  LOG(INFO) << "LiteSiluConvertPass applied successfully";
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(lite_silu_convert_pass, paddle::lite::mir::LiteSiluConvertPass)
    .BindTargets({TARGET(kARM), TARGET(kOpenCL), TARGET(kX86)});