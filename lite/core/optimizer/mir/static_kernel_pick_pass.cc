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

#include "lite/core/optimizer/mir/static_kernel_pick_pass.h"
#include <algorithm>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "lite/core/optimizer/mir/graph_visualize_pass.h"
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

bool KernelScoreCmp(const std::pair<float, std::unique_ptr<KernelBase>>& a,
                    const std::pair<float, std::unique_ptr<KernelBase>>& b) {
  return a.first > b.first;
}

// 在static_kernel_pick_pass.cc文件中添加DetectPattern方法的实现
void StaticKernelPickPass::DetectPattern(const lite::mir::Node* node) {
  if (!node->IsStmt()) return;

  const auto& op_type = node->stmt()->op_type();

  switch (pattern_state_) {
    case PatternState::INIT:
      if (op_type == "split" || op_type == "split_v2") {
        pattern_state_ = PatternState::FOUND_SPLIT;
        VLOG(3) << "Pattern detection: Found split";
      }
    break;
    case PatternState::FOUND_SPLIT:
      if (op_type == "reshape" || op_type == "reshape2") {
        pattern_state_ = PatternState::FOUND_RESHAPE;
        VLOG(3) << "Pattern detection: Found reshape after split";
      } else {
        pattern_state_ = PatternState::INIT;
      }
    break;
    case PatternState::FOUND_RESHAPE:
      if (op_type == "transpose" || op_type == "transpose2") {
        pattern_state_ = PatternState::PATTERN_MATCHED;
        after_pattern_ = true;
        LOG(INFO) << "Detected split+reshape2+transpose2 pattern, forcing ARM kernels afterwards";
      } else {
        pattern_state_ = PatternState::INIT;
      }
    break;
    case PatternState::PATTERN_MATCHED:
      // 已经匹配到模式，保持状态
        break;
  }
}
// 添加模式检测方法
void StaticKernelPickPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  // 重置状态
  pattern_state_ = PatternState::INIT;
  after_pattern_ = false;

  kernel_pick_factors_.ConsiderTarget();
  kernel_pick_factors_.ConsiderPrecision();
  kernel_pick_factors_.ConsiderDataLayout();
  CHECK(kernel_pick_factors_.any_factor_considered())
      << "kernel_pick_factors should be specified first";
  CHECK(graph) << "graph not valid";

  // 使用节点唯一标识符作为键，而不是指针
  std::map<std::string, int> op_index;
  std::set<std::string> pattern_ops; // 记录模式中的操作类型

  int pattern_start_pos = -1;
  int pattern_end_pos = -1;

  if (force_arm_mode_) {
    // 第一遍：按拓扑顺序扫描并检测模式
    auto topo_nodes = graph->StmtTopologicalOrder();
    for (int i = 0; i < topo_nodes.size(); ++i) {
      auto node = topo_nodes[i];
      if (!node->IsStmt()) continue;

      // 生成唯一标识
      std::string op_key = node->AsStmt().op_type() + "_" + std::to_string(node->get_id());
      op_index[op_key] = i;

      // 记录模式状态前的值
      PatternState old_state = pattern_state_;
      DetectPattern(node);

      // 如果刚检测到模式的第一个节点(split)
      if (old_state == PatternState::INIT &&
          pattern_state_ == PatternState::FOUND_SPLIT) {
        pattern_start_pos = i;
        pattern_ops.insert(node->AsStmt().op_type());
        LOG(INFO) << "找到模式起始节点 (split) 在位置 " << i
                  << " 节点ID " << node->get_id();
      }
      // 如果是模式的中间节点(reshape)
      else if (old_state == PatternState::FOUND_SPLIT &&
               pattern_state_ == PatternState::FOUND_RESHAPE) {
        pattern_ops.insert(node->AsStmt().op_type());
        LOG(INFO) << "找到模式中间节点 (reshape) 在位置 " << i
                  << " 节点ID " << node->get_id();
      }
      // 如果是刚刚匹配到完整模式的节点(transpose)
      else if (pattern_state_ == PatternState::PATTERN_MATCHED &&
               old_state == PatternState::FOUND_RESHAPE) {
        pattern_end_pos = i;
        pattern_ops.insert(node->AsStmt().op_type());
        LOG(INFO) << "找到模式结束节点 (transpose) 在位置 " << i
                  << " 节点ID " << node->get_id()
                  << ", 模式检测范围从 " << pattern_start_pos
                  << " 到 " << pattern_end_pos;
      }
    }

    // 如果检测到了完整模式
    if (pattern_start_pos >= 0 && pattern_end_pos >= 0) {
      LOG(INFO) << "成功检测到 split+reshape+transpose 模式"
                << " (位置 " << pattern_start_pos << " 到 " << pattern_end_pos << "), "
                << "将对该模式内及之后的算子强制使用 ARM/Host 内核";
    }

    // 重置状态，准备第二次遍历
    pattern_state_ = PatternState::INIT;
    after_pattern_ = false;
  }

  // 第二遍：根据模式信息选择内核
  for (auto& node : graph->mutable_nodes()) {
    if (!node.IsStmt()) continue;
    auto& instruct = node.AsStmt();

    // 生成唯一标识符
    std::string op_key = instruct.op_type() + "_" + std::to_string(node.get_id());

    VLOG(2) << "为算子选择内核: " << instruct.op_type();

    // 获取输入输出类型信息
    std::map<std::string, PrecisionType> in_types;
    std::map<std::string, PrecisionType> out_types;
    for (auto* in : node.inlinks) {
      if (!in->IsArg()) continue;
      auto& arg = in->AsArg();
      if (arg.is_weight || arg.is_persist) continue;
      if (arg.type && arg.type->precision() != PRECISION(kUnk)) {
        in_types[arg.name] = arg.type->precision();
      }
    }
    for (auto* out : node.outlinks) {
      if (!out->IsArg()) continue;
      auto& arg = out->AsArg();
      if (arg.type && arg.type->precision() != PRECISION(kUnk)) {
        out_types[arg.name] = arg.type->precision();
      }
    }

    // 获取候选内核
    std::vector<std::pair<float, std::unique_ptr<KernelBase>>> scored;
    CHECK(!instruct.kernels().empty()) << "没有找到算子的内核: " << instruct.op_type();

    // 判断该节点是否在模式内或模式后
    bool is_in_or_after_pattern = false;
    if (force_arm_mode_ && pattern_start_pos >= 0) {
      // 方法1: 检查在拓扑顺序中的位置
      auto it = op_index.find(op_key);
      if (it != op_index.end() && it->second >= pattern_start_pos) {
        is_in_or_after_pattern = true;
        LOG(INFO) << "节点 " << instruct.op_type() << " 在位置 " << it->second
                  << " 位于模式内或模式之后 (模式起始位置 " << pattern_start_pos << ")";
      }
      // 方法2: 检查操作类型是否匹配模式中的操作
      else if (pattern_ops.find(instruct.op_type()) != pattern_ops.end()) {
        is_in_or_after_pattern = true;
        LOG(INFO) << "节点 " << instruct.op_type() << " 是检测到的模式的一部分";
      }
    }

    // 如果节点在模式内部或之后，优先选择ARM或host内核
    if (is_in_or_after_pattern) {
      LOG(INFO) << "尝试为算子 " << instruct.op_type() << " 使用 ARM/Host 内核";

      // 检查是否有ARM或Host内核
      bool has_arm = false;
      bool has_host = false;

      for (auto& kernel : instruct.kernels()) {
        if (kernel->target() == TARGET(kARM)) has_arm = true;
        if (kernel->target() == TARGET(kHost)) has_host = true;
      }

      LOG(INFO) << "可用内核: ARM=" << has_arm << ", Host=" << has_host;

      // 添加所有内核，但给ARM和Host内核加分
      for (auto& kernel : instruct.kernels()) {
        float score = KernelGrade(&node,
                               *kernel,
                               graph->valid_places(),
                               in_types,
                               out_types,
                               instruct.op_info()->input_names(),
                               instruct.op_info()->output_names());

        // 提升ARM和Host内核的优先级
        if (kernel->target() == TARGET(kARM)) {
          score += 200000.0f;  // ARM内核优先级最高
          LOG(INFO) << "提升ARM内核分数+2000: " << kernel->name();
        } else if (kernel->target() == TARGET(kHost)) {
          score += 100000.0f;  // Host内核优先级次之
          LOG(INFO) << "提升Host内核分数+1000: " << kernel->name();
        }

        scored.emplace_back(score, std::move(kernel));
      }
    } else {
      // 模式之前的节点使用原有的评分逻辑
      for (auto&& kernel : instruct.kernels()) {
        VLOG(2) << "当前候选内核: " << kernel->summary();
        VLOG(2) << "有效设备数量: " << graph->valid_places().size();
        float score = KernelGrade(&node,
                               *kernel,
                               graph->valid_places(),
                               in_types,
                               out_types,
                               instruct.op_info()->input_names(),
                               instruct.op_info()->output_names());

        scored.emplace_back(score, std::move(kernel));
      }
    }

    // 排序、选择最佳内核
    std::stable_sort(scored.begin(), scored.end(), KernelScoreCmp);
    instruct.kernels().clear();

    if (scored.empty()) {
      LOG(WARNING) << "算子 " << instruct.op_type() << " 没有找到合适的内核";
      continue;
    }

    // 处理非int8情况
    if (!instruct.op_info()->HasAttr("enable_int8")) {
      // 只保留得分最高的内核
      instruct.kernels().emplace_back(std::move(scored.front().second));
      VLOG(3) << "最终选择的内核是 "
              << instruct.kernels().front()->summary() << "\n\n";
    } else {
      // 处理int8量化情况
      bool out_type_int8 = true;
      // 量化的LSTM和GRU有fp32输出
      if (instruct.op_type() == "lstm" || instruct.op_type() == "gru") {
        out_type_int8 = false;
      }

      // 只有当所有连接到该算子输出的算子都有enable_int8属性时，输出类型才是int8，否则为fp32
      for (auto* out_n : node.outlinks) {
        CHECK(out_n->IsArg());
        for (auto* tmp_op : out_n->outlinks) {
          CHECK(tmp_op->IsStmt());
          auto* tmp_op_info = tmp_op->AsStmt().op_info();
          if (!tmp_op_info->HasAttr("enable_int8") ||
              tmp_op_info->Type() == "lstm" || tmp_op_info->Type() == "gru") {
            out_type_int8 = false;
            break;
          }
        }
        if (!out_type_int8) break;
      }

      // 如果out_type_int8为true，表示该算子的输出类型可以是int8
      if (out_type_int8) {
        auto out_node = node.outlinks.front();
        CHECK(out_node->IsArg());
        auto out_node_name = out_node->arg()->name;
        auto one_adj_op_node = out_node->outlinks.front();
        CHECK(one_adj_op_node->IsStmt());
        auto& one_adj_instruct = one_adj_op_node->AsStmt();
        CHECK(one_adj_instruct.op_info()->HasAttr("enable_int8"));
        CHECK(one_adj_instruct.op_info()->HasInputScale(out_node_name));

        instruct.mutable_op_info()->SetOutputScale(
            out_node_name,
            one_adj_instruct.op_info()->GetInputScale(out_node_name));

        auto update_desc = *instruct.mutable_op_info();
        instruct.ResetOp(update_desc, graph->valid_places());
        scored.clear();
        for (auto&& kernel : instruct.kernels()) {
          float score = KernelGrade(&node,
                                    *kernel,
                                    graph->valid_places(),
                                    in_types,
                                    out_types,
                                    instruct.op_info()->input_names(),
                                    instruct.op_info()->output_names());
          scored.emplace_back(score, std::move(kernel));
        }
        std::stable_sort(scored.begin(), scored.end(), KernelScoreCmp);
        instruct.kernels().clear();
      }

      // 根据输出类型选择内核
      auto output_arguments = instruct.op_info()->OutputArgumentNames();
      for (auto& candidate : scored) {
        bool all_output_type_match = true;
        auto expect_output_type =
            out_type_int8 ? PRECISION(kInt8) : PRECISION(kFloat);

        for (auto& arg_name : output_arguments) {
          const Type* out_arg_ty =
              candidate.second->GetOutputDeclType(arg_name);
          if (out_arg_ty->precision() != expect_output_type) {
            all_output_type_match = false;
          }
        }

        if (all_output_type_match) {
          instruct.kernels().emplace_back(std::move(candidate.second));
          VLOG(2) << "选择内核 " << instruct.kernels().front()->name();
          break;
        }
      }
      CHECK(!instruct.kernels().empty())
          << "算子 " << instruct.op_type() << " 没有找到合适的内核";
    }
  }

  // 添加统计信息输出
  if (force_arm_mode_ && pattern_end_pos >= 0) {
    int arm_kernel_count = 0;
    int host_kernel_count = 0;
    int other_kernel_count = 0;

    for (auto& node : graph->mutable_nodes()) {
      if (!node.IsStmt()) continue;
      if (!node.AsStmt().kernels().empty()) {
        auto target = node.AsStmt().kernels().front()->target();
        if (target == TARGET(kARM)) {
          arm_kernel_count++;
        } else if (target == TARGET(kHost)) {
          host_kernel_count++;
        } else {
          other_kernel_count++;
        }
      }
    }

    LOG(INFO) << "内核分布统计: ARM: " << arm_kernel_count
              << ", Host: " << host_kernel_count
              << ", 其他: " << other_kernel_count;
  }

}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(static_kernel_pick_pass,
                  paddle::lite::mir::StaticKernelPickPass)
    .BindTargets({TARGET(kAny)})
    .ExcludeTargets({TARGET(kXPU)});
