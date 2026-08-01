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

// Shared helpers for fusion-pass unit tests. Builds a Program + SSAGraph from
// a hand-written op list and lets tests assert on the resulting graph.
#pragma once

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "lite/core/optimizer/mir/ssa_graph.h"
#include "lite/core/program.h"
#include "lite/model_parser/cpp_desc.h"

namespace paddle {
namespace lite {
namespace mir {

// One op in the hand-written graph: type + input/output maps + attrs.
struct TestOpDesc {
  std::string type;
  std::map<std::string, std::vector<std::string>> inputs;
  std::map<std::string, std::vector<std::string>> outputs;
  std::map<std::string, int> int_attrs;
  std::map<std::string, float> float_attrs;
  std::map<std::string, bool> bool_attrs;
  std::map<std::string, std::string> str_attrs;
  std::map<std::string, std::vector<int>> int_vector_attrs;
};

// Build a cpp::ProgramDesc from a list of ops. All intermediate vars are
// created as DENSE_TENSOR (non-persistable) unless listed in `persistable`.
std::shared_ptr<cpp::ProgramDesc> BuildProgramDesc(
    const std::vector<TestOpDesc>& ops,
    const std::set<std::string>& persistable = {});

// Create a float scalar tensor in `scope` under `name`.
void SetScalar(Scope* scope, const std::string& name, float value);

// Build an SSAGraph from ops. `scope` must outlive the returned graph.
std::unique_ptr<SSAGraph> BuildGraph(const std::vector<TestOpDesc>& ops,
                                     const std::set<std::string>& persistable,
                                     Scope* scope);

// Count stmt nodes of a given op type in the graph.
int CountOp(const SSAGraph& graph, const std::string& op_type);

// True if any stmt node of `op_type` has a float attr `name` == `value`.
bool OpHasFloatAttr(const SSAGraph& graph,
                    const std::string& op_type,
                    const std::string& name,
                    float value);

}  // namespace mir
}  // namespace lite
}  // namespace paddle
