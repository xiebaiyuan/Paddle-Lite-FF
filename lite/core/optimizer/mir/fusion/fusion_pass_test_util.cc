// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <cmath>

#include "lite/core/optimizer/mir/fusion/fusion_pass_test_util.h"

namespace paddle {
namespace lite {
namespace mir {

std::shared_ptr<cpp::ProgramDesc> BuildProgramDesc(
    const std::vector<TestOpDesc>& ops,
    const std::set<std::string>& persistable) {
  auto program = std::make_shared<cpp::ProgramDesc>();
  auto* block = program->AddBlock<cpp::BlockDesc>();
  block->SetIdx(0);
  std::set<std::string> all_vars;
  for (const auto& op : ops) {
    auto* op_desc = block->AddOp<cpp::OpDesc>();
    op_desc->SetType(op.type);
    for (const auto& kv : op.inputs) {
      op_desc->SetInput(kv.first, kv.second);
      all_vars.insert(kv.second.begin(), kv.second.end());
    }
    for (const auto& kv : op.outputs) {
      op_desc->SetOutput(kv.first, kv.second);
      all_vars.insert(kv.second.begin(), kv.second.end());
    }
    for (const auto& kv : op.int_attrs) {
      op_desc->SetAttr(kv.first, kv.second);
    }
    for (const auto& kv : op.float_attrs) {
      op_desc->SetAttr(kv.first, kv.second);
    }
    for (const auto& kv : op.bool_attrs) {
      op_desc->SetAttr(kv.first, kv.second);
    }
    for (const auto& kv : op.str_attrs) {
      op_desc->SetAttr(kv.first, kv.second);
    }
    for (const auto& kv : op.int_vector_attrs) {
      op_desc->SetAttr(kv.first, kv.second);
    }
    for (const auto& kv : op.float_vector_attrs) {
      op_desc->SetAttr(kv.first, kv.second);
    }
    for (const auto& kv : op.string_vector_attrs) {
      op_desc->SetAttr(kv.first, kv.second);
    }
  }
  for (const auto& name : all_vars) {
    auto* var = block->AddVar<cpp::VarDesc>();
    var->SetName(name);
    var->SetType(lite::VarDescAPI::Type::DENSE_TENSOR);
    if (persistable.count(name)) {
      var->SetPersistable(true);
    }
  }
  return program;
}

void SetScalar(Scope* scope, const std::string& name, float value) {
  auto* tensor = scope->Var(name)->GetMutable<lite::Tensor>();
  tensor->Resize(DDim({1}));
  tensor->mutable_data<float>()[0] = value;
}

std::unique_ptr<SSAGraph> BuildGraph(const std::vector<TestOpDesc>& ops,
                                     const std::set<std::string>& persistable,
                                     Scope* scope) {
  auto program_desc = BuildProgramDesc(ops, persistable);
  // Kernels are registered via paddle_use_kernels.h (included by the test
  // binaries); SSAGraph::Build requires at least one kernel per op. Host
  // builds register x86 kernels, ARM builds register ARM kernels.
#if defined(LITE_WITH_ARM)
  std::vector<Place> valid_places{Place{TARGET(kARM), PRECISION(kFloat)}};
#else
  std::vector<Place> valid_places{Place{TARGET(kX86), PRECISION(kFloat)}};
#endif
  lite::Program program(
      program_desc, std::shared_ptr<Scope>(scope, [](Scope*) {}), valid_places);
  auto graph = std::unique_ptr<SSAGraph>(new SSAGraph());
  graph->Build(program, valid_places);
  return graph;
}

int CountOp(const SSAGraph& graph, const std::string& op_type) {
  int count = 0;
  for (auto& node : graph.nodes()) {
    if (node.IsStmt() && node.stmt()->op_info()->Type() == op_type) {
      ++count;
    }
  }
  return count;
}

bool OpHasFloatAttr(const SSAGraph& graph,
                    const std::string& op_type,
                    const std::string& name,
                    float value) {
  for (auto& node : graph.nodes()) {
    if (node.IsStmt() && node.stmt()->op_info()->Type() == op_type &&
        node.stmt()->op_info()->HasAttr(name)) {
      if (std::fabs(node.stmt()->op_info()->GetAttr<float>(name) - value) <
          1e-6f) {
        return true;
      }
    }
  }
  return false;
}

bool OpHasBoolAttr(const SSAGraph& graph,
                   const std::string& op_type,
                   const std::string& name,
                   bool value) {
  for (auto& node : graph.nodes()) {
    if (node.IsStmt() && node.stmt()->op_info()->Type() == op_type &&
        node.stmt()->op_info()->HasAttr(name)) {
      if (node.stmt()->op_info()->GetAttr<bool>(name) == value) return true;
    }
  }
  return false;
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
