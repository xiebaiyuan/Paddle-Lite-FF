// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
// Standalone test for PassManagerV2 - no gtest dependency.
// Compiles directly against Paddle-Lite headers + core.a + utils.a.
#include "lite/core/optimizer/mir/pass_v2.h"
#include "lite/core/optimizer/mir/pass_manager_v2.h"
#include "lite/core/optimizer/mir/ssa_graph.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <memory>

namespace paddle {
namespace lite {
namespace mir {

#define T_CHECK(cond, msg) do { \
  if (!(cond)) { fprintf(stderr, "FAIL: %s\n  %s:%d\n", msg, __FILE__, __LINE__); exit(1); } \
  else { fprintf(stdout, "  PASS: %s\n", msg); } \
} while(0)
#define T_CHECK_EQ(a, b, msg) T_CHECK((a) == (b), msg)

// ============================================================
// Test helpers
// ============================================================
class CountingPass : public PassV2 {
 public:
  CountingPass() : PassV2(Kind::kProgramWise), call_count_(0), did_modify_(true) {}
  void Apply(const std::unique_ptr<SSAGraph>& graph) override { ++call_count_; }
  void ApplyV2(const std::unique_ptr<SSAGraph>& graph, bool& modified) override {
    ++call_count_; modified = did_modify_;
  }
  bool ModifiedGraph() const override { return did_modify_; }
  bool ShouldOnlyApplyOnce() const override { return apply_once_; }
  int call_count() const { return call_count_; }
  void set_did_modify(bool v) { did_modify_ = v; }
  void set_apply_once(bool v) { apply_once_ = v; }
 private:
  int call_count_;
  bool did_modify_;
  bool apply_once_ = false;
};

class OneShotFusionPass : public CountingPass {
 public:
  OneShotFusionPass() = default;  // Inherits did_modify=true from CountingPass
  void ApplyV2(const std::unique_ptr<SSAGraph>& graph, bool& modified) override {
    CountingPass::ApplyV2(graph, modified);
    set_did_modify(false);  // Only modifies the graph on the first invocation
  }
};

class IdempotentEliminationPass : public CountingPass {
 public:
  IdempotentEliminationPass() { set_did_modify(false); }
};
}  // namespace mir
}  // namespace lite
}  // namespace paddle

#define TEST_CASE(name) static void test_##name()
// Functions for SSAGraph creation
// Workaround: program.h includes too many deps, create minimal graph inline
static std::unique_ptr<paddle::lite::mir::SSAGraph> make_graph() {
  auto g = std::unique_ptr<paddle::lite::mir::SSAGraph>(new paddle::lite::mir::SSAGraph());
  return g;
}

// ============================================================
// Test cases
// ============================================================
TEST_CASE(single_pass_one_step) {
  paddle::lite::mir::PassManagerV2 mgr;
  mgr.ClearAll();
  auto* p = new paddle::lite::mir::CountingPass(); p->set_name("counting");
  mgr.AddNewPass("counting", p);
  auto g = make_graph();
  mgr.Run(g, 1);
  T_CHECK_EQ(p->call_count(), 1, "single pass called once");
  mgr.ClearAll();
}

TEST_CASE(early_convergence) {
  paddle::lite::mir::PassManagerV2 mgr;
  mgr.ClearAll();
  auto* p = new paddle::lite::mir::IdempotentEliminationPass();
  p->set_name("idem");
  mgr.AddNewPass("idem", p);
  auto g = make_graph();
  mgr.Run(g, 5);
  T_CHECK_EQ(p->call_count(), 1, "idempotent pass converges after 1 step");
  mgr.ClearAll();
}

TEST_CASE(apply_once) {
  paddle::lite::mir::PassManagerV2 mgr;
  mgr.ClearAll();
  auto* once = new paddle::lite::mir::CountingPass();
  once->set_name("one"); once->set_apply_once(true);
  auto* every = new paddle::lite::mir::CountingPass();
  every->set_name("every");
  mgr.AddNewPass("one", once);
  mgr.AddNewPass("every", every);
  auto g = make_graph();
  mgr.Run(g, 3);
  T_CHECK_EQ(once->call_count(), 1, "apply-once pass called only once");
  T_CHECK_EQ(every->call_count(), 3, "normal pass called every step");
  mgr.ClearAll();
}

TEST_CASE(fusion_chain_convergence) {
  paddle::lite::mir::PassManagerV2 mgr;
  mgr.ClearAll();
  auto* conv_bn = new paddle::lite::mir::OneShotFusionPass();
  conv_bn->set_name("conv_bn");
  auto* conv_act = new paddle::lite::mir::OneShotFusionPass();
  conv_act->set_name("conv_activation");
  auto* cf = new paddle::lite::mir::IdempotentEliminationPass();
  cf->set_name("const_fold"); cf->set_apply_once(true);
  mgr.AddNewPass("conv_bn", conv_bn);
  mgr.AddNewPass("conv_activation", conv_act);
  mgr.AddNewPass("const_fold", cf);
  auto g = make_graph();
  mgr.Run(g, 4);
  T_CHECK_EQ(conv_bn->call_count(), 2, "conv_bn called on step 0 and 1");
  T_CHECK_EQ(conv_act->call_count(), 2, "conv_activation called on step 0 and 1");
  T_CHECK_EQ(cf->call_count(), 1, "const_fold apply-once, only step 0");
  mgr.ClearAll();
}

TEST_CASE(level_filtering) {
  paddle::lite::mir::PassManagerV2 mgr;
  mgr.ClearAll();
  auto* basic = new paddle::lite::mir::CountingPass();
  basic->set_name("basic");
  auto* aggressive = new paddle::lite::mir::CountingPass();
  aggressive->set_name("aggressive"); aggressive->set_did_modify(true);
  mgr.AddNewPass("basic", basic);
  mgr.AddNewPass("aggressive", aggressive);
  mgr.SetLevel("basic", paddle::lite::mir::PassV2::kLevelBasic);
  mgr.SetLevel("aggressive", paddle::lite::mir::PassV2::kLevelAggressive);
  auto g = make_graph();
  mgr.Run(g, 1, 1);
  T_CHECK(basic->call_count() >= 1, "basic pass runs at level 1");
  T_CHECK_EQ(aggressive->call_count(), 0, "aggressive pass skipped at level 1");
  mgr.ClearAll();
}

TEST_CASE(legacy_pass_compatibility) {
  struct LegacyPass : public paddle::lite::mir::StmtPass {
    int* counter;
    explicit LegacyPass(int* c) : StmtPass(), counter(c) {}
    void Apply(const std::unique_ptr<paddle::lite::mir::SSAGraph>& graph) override { (*counter)++; }
  };
  paddle::lite::mir::PassManagerV2 mgr;
  mgr.ClearAll();
  int count = 0;
  auto* lp = new LegacyPass(&count); lp->set_name("legacy");
  mgr.AddNewPass("legacy", lp);
  auto g = make_graph();
  mgr.Run(g, 2);
  T_CHECK_EQ(count, 2, "legacy pass runs every step");
  mgr.ClearAll();
}

TEST_CASE(level_zero_skips_all) {
  paddle::lite::mir::PassManagerV2 mgr;
  mgr.ClearAll();
  auto* p = new paddle::lite::mir::CountingPass();
  p->set_name("any"); p->set_did_modify(true);
  mgr.AddNewPass("any", p);
  mgr.SetLevel("any", paddle::lite::mir::PassV2::kLevelBasic);
  auto g = make_graph();
  mgr.Run(g, 1, 0);
  T_CHECK_EQ(p->call_count(), 0, "no pass runs at level 0");
  mgr.ClearAll();
}

TEST_CASE(pass_promotion) {
  paddle::lite::mir::PassManagerV2 mgr;
  mgr.ClearAll();
  auto* p = new paddle::lite::mir::CountingPass();
  p->set_name("counting");
  mgr.AddNewPass("counting", p);
  T_CHECK(mgr.LookUp("counting") != nullptr, "pass registered and lookup works");
  mgr.ClearAll();
}

int main() {
  fprintf(stdout, "=== PassManagerV2 Standalone Tests ===\n");
  test_single_pass_one_step();
  test_early_convergence();
  test_apply_once();
  test_fusion_chain_convergence();
  test_level_filtering();
  test_level_zero_skips_all();
  test_legacy_pass_compatibility();
  test_pass_promotion();
  fprintf(stdout, "\n=== ALL 8 TESTS PASSED ===\n");
  return 0;
}
