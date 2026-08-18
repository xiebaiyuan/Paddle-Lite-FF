"""Paddle-Lite 可融合模式规则表 + 扫描引擎（netron 风格分析）。

规则来源：lite/core/optimizer/mir/fusion/*_fuser.cc 的 BuildPattern（OpNode/VarNode
链式断言），转译为声明式规则。本文件只做"候选识别"——命中表示拓扑+可检查属性满足，
是否真触发还取决于 kernel 注册/后端（opt 实际跑 pass 才知道）。

规则字段：
    name    规则名（对应 pass 名）
    pass    实际 MIR pass 名
    ops     op type 序列（每项支持 "a|b" 或门）
    edges   相邻 op 的连接：(producer_type, producer_port, consumer_type, consumer_port)
    conds   属性谓词（可选），每项 dict：
              - {"op": 位, "attr": 名, "~=": 值}          attr 近似相等（float 容差 1e-5）
              - {"op": 位, "attr": 名, "==": 值}          attr 精确相等
              - {"op": 位, "attr": "存在"}                  attr 存在
              - {"op": 位, "input": 端口, "persistable": True}  该端口输入 var 全部 persistable
              - {"op": 位, "input": 端口, "only_one_output": True} 该端口输入 var 唯一消费者
              - {"op": 位, "input": 端口, "dims": [[d0,d1,...], ...]} 该端口输入 var 的 dims 满足（任一匹配即通过；0 通配任意值）
    fused   fused op 名（展示用）
    removable   AsIntermediate 顶点数（融合后可省的 op 数）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# conv2d + depthwise_conv2d 族
_CONV = "conv2d|depthwise_conv2d|conv2d_transpose"
_CONV2 = "conv2d|depthwise_conv2d"
# ARM 端激活（conv_activation_fuse_pass.cc ARM 分支；gelu 已注释禁用——性能回退）
_ACT_CONV = "relu|relu6|leaky_relu|hard_swish|hard_sigmoid|sigmoid|tanh|swish|abs"
_ACT_GENERIC = "relu|relu6|leaky_relu|hard_swish|hard_sigmoid|prelu|sigmoid|tanh|swish|abs|relu1"
_ELEM = "elementwise_add|elementwise_mul|elementwise_sub|elementwise_div|elementwise_max|elementwise_min"

FUSION_RULES: list[dict[str, Any]] = [
    # ---------- conv 族 ----------
    {
        "name": "conv_activation",
        "pass": "lite_conv_activation_fuse_pass",
        "ops": [_CONV, _ACT_CONV],
        "edges": [("A", "Output", "B", "X")],
        "conds": [],
        # ARM 分支的 act_types 只含 relu/relu6/leaky_relu/hard_swish（gelu 已
        # 注释禁用）；hard_sigmoid/swish/prelu 等仅 OpenCL/Metal 分支支持。
        # 声明 ARM 可融合的激活类型，扫描时按 ARM 语义过滤（本仓库主目标
        # 是 ARM；OpenCL 场景可手动放开）。
        "arm_act_types": ["relu", "relu6", "leaky_relu", "hard_swish"],
        "fused": "conv2d+fused_act",
        "removable": 1,
    },
    {
        "name": "conv_elementwise",
        "pass": "lite_conv_elementwise_fuse_pass",
        "ops": [_CONV2, "elementwise_add"],
        "edges": [("A", "Output", "B", "X")],
        "conds": [
            {"op": "B", "input": "Y", "persistable": True},
            {"op": "B", "input": "Y", "only_one_output": True},
        ],
        "fused": "conv2d+bias(add)",
        "removable": 1,
    },
    {
        "name": "conv_scale",
        "pass": "lite_conv_scale_fuse_pass",
        "ops": [_CONV2, "scale"],
        "edges": [("A", "Output", "B", "X")],
        "conds": [],
        "fused": "conv2d+scale",
        "removable": 1,
    },
    {
        "name": "conv_bn",
        "pass": "lite_conv_bn_fuse_pass",
        "ops": [_CONV, "batch_norm"],
        "edges": [("A", "Output", "B", "X")],
        "conds": [],
        "fused": "conv2d+bn",
        "removable": 1,
    },
    {
        "name": "conv_conv",
        "pass": "lite_conv_conv_fuse_pass",
        "ops": [_CONV2, _CONV2],
        "edges": [("A", "Output", "B", "Input")],
        # ConvConvFuser requires (mirrors conv_conv_fuser.cc BuildPattern):
        #   - both convs have groups == 1 (depthwise 1x1 → 1x1 excluded)
        #   - neither conv carries a fused activation (with_act missing or
        #     false) — recomputing weights would swallow conv0's activation
        #   - the second conv's filter is 1x1 (kernel size check)
        # The computation-gain check ic0*(oc1-oc0) <= oc0*oc1 depends on
        # weights at runtime and is left to the pass itself.
        "conds": [
            {"op": "A", "attr": "groups", "==": 1},
            {"op": "B", "attr": "groups", "==": 1},
            {"op": "A", "attr": "with_act", "==": False},
            {"op": "B", "attr": "with_act", "==": False},
            {"op": "B", "input": "Filter", "dims": [[0, 0, 1, 1]]},
        ],
        "fused": "conv2d+conv2d",
        "removable": 1,
    },
    {
        "name": "conv_hardswish",
        "pass": "lite_conv_hardswish_fuse_pass",
        "ops": [_CONV2, "hard_swish"],
        "edges": [("A", "Output", "B", "X")],
        "conds": [],
        "fused": "conv2d+hardswish",
        "removable": 1,
    },

    # ---------- elementwise 族 ----------
    {
        "name": "elementwise_add_activation",
        "pass": "lite_elementwise_add_activation_fuse_pass",
        "ops": [_ELEM, _ACT_GENERIC],
        "edges": [("A", "Out", "B", "X")],
        "conds": [],
        # elementwise_add_activation_fuse_pass.cc 的 ARM 分支只支持 relu
        # （"arm not support tanh and abs act fusion"）；sigmoid 等仅 OpenCL。
        "arm_act_types": ["relu"],
        "fused": "elementwise+act",
        "removable": 1,
    },
    {
        "name": "elementwise_add_scale",
        "pass": "lite_elementwise_add_scale_fuse_pass",
        "ops": [_ELEM, "scale"],
        "edges": [("A", "Out", "B", "X")],
        "conds": [],
        "fused": "elementwise+scale",
        "removable": 1,
    },
    {
        "name": "scale_activation",
        "pass": "lite_scale_activation_fuse_pass",
        "ops": ["scale", _ACT_GENERIC],
        "edges": [("A", "Out", "B", "X")],
        "conds": [],
        "fused": "scale+act",
        "removable": 1,
    },
    {
        "name": "scale_clip_mul",
        "pass": "lite_scale_clip_mul_fuse_pass",
        "ops": ["scale", "clip", "elementwise_mul"],
        # The fused pattern is x → scale → clip → mul, where the clip output
        # feeds the mul's Y operand (x * hard_sigmoid(gate)); the main branch
        # feeds X. The edge must therefore connect clip_out to mul's Y port.
        "edges": [("A", "Out", "B", "X"), ("B", "Out", "C", "Y")],
        "conds": [],
        "fused": "fusion_scale_clip_mul",
        "removable": 2,
    },
    {
        "name": "sigmoid_elementmul",
        "pass": "lite_sigmoid_elementmul_fuse_pass",
        "ops": ["sigmoid", "elementwise_mul"],
        "edges": [("A", "Out", "B", "Y")],
        "conds": [{"op": "B", "input": "X", "persistable": False}],
        "fused": "fusion_silu",  # SiLU
        "removable": 1,
    },
    {
        "name": "div_mul",
        "pass": "lite_div_mul_fuse_pass",
        "ops": ["elementwise_div", "elementwise_mul"],
        "edges": [("A", "Out", "B", "X")],
        "conds": [
            {"op": "A", "input": "Y", "persistable": True},
            {"op": "B", "input": "Y", "persistable": True},
        ],
        "fused": "fusion_div_mul",
        "removable": 1,
    },

    # ---------- fc / matmul 族 ----------
    {
        "name": "fc",
        "pass": "lite_fc_fuse_pass",
        "ops": ["mul", "elementwise_add"],
        "edges": [("A", "Out", "B", "X")],
        "conds": [
            {"op": "A", "attr": "x_num_col_dims", "==": 1},
            {"op": "A", "attr": "y_num_col_dims", "==": 1},
            {"op": "B", "input": "Y", "persistable": True},
        ],
        "fused": "fc",
        "removable": 1,
    },
    {
        "name": "fc_activation",
        "pass": "lite_fc_fuse_pass",
        "ops": ["mul", "elementwise_add", _ACT_GENERIC],
        "edges": [("A", "Out", "B", "X"), ("B", "Out", "C", "X")],
        "conds": [
            {"op": "A", "attr": "x_num_col_dims", "==": 1},
            {"op": "A", "attr": "y_num_col_dims", "==": 1},
            {"op": "B", "input": "Y", "persistable": True},
        ],
        "fused": "fc+act",
        "removable": 2,
    },
    {
        "name": "matmul_elementwise_add",
        "pass": "lite_matmul_elementwise_add_fuse_pass",
        "ops": ["matmul|matmul_v2", "elementwise_add"],
        "edges": [("A", "Out", "B", "X")],
        "conds": [
            {"op": "A", "attr": "alpha", "~=": 1.0},
            {"op": "B", "input": "Y", "persistable": True},
        ],
        "fused": "fusion_matmul_add",
        "removable": 1,
    },
    {
        "name": "reshape2_matmul",
        "pass": "lite_reshape2_matmul_fuse_pass",
        "ops": ["reshape2", "matmul|matmul_v2"],
        "edges": [("A", "Out", "B", "X")],
        # Reshape2MatmulFuser requires:
        #   - reshape2's input X has rank 4 with the last two dims == 1
        #     (e.g. [N, C, 1, 1]); a 4D batch matmul operand is NOT this shape.
        #   - matmul's input X (and Y) have rank 2.
        # Static-shape dims are checked conservatively; dynamic shapes (-1)
        # never match, which avoids reporting 4D batch matmuls as fusable.
        "conds": [
            {"op": "A", "input": "X", "dims": [[0, 0, 1, 1], [0, 0, 0, 1]]},
            {"op": "B", "input": "X", "dims": [[0, 0], [1, 1], [1, 0]]},
        ],
        "fused": "reshape2+matmul",
        "removable": 1,
    },
    {
        "name": "squeeze2_matmul",
        "pass": "lite_squeeze2_matmul_fuse_pass",
        "ops": ["squeeze2", "matmul|matmul_v2"],
        "edges": [("A", "Out", "B", "X")],
        "conds": [],
        "fused": "squeeze2+matmul",
        "removable": 1,
    },
    {
        "name": "flatten_fc",
        "pass": "lite_flatten_fc_fuse_pass",
        "ops": ["flatten2|flatten_contiguous_range", "mul"],
        "edges": [("A", "Out", "B", "X")],
        "conds": [],
        "fused": "flatten+fc",
        "removable": 1,
    },

    # ---------- gelu / 其他 ----------
    {
        "name": "gelu",
        "pass": "lite_gelu_fuse_pass",
        "ops": ["elementwise_div", "erf", "elementwise_add", "elementwise_mul", "elementwise_mul"],
        "edges": [
            ("A", "Out", "B", "X"), ("B", "Out", "C", "X"),
            ("C", "Out", "D", "Y"), ("D", "Out", "E", "X"),
        ],
        "conds": [
            {"op": "A", "input": "Y", "persistable": True},
            {"op": "C", "input": "Y", "persistable": True},
            {"op": "E", "input": "Y", "persistable": True},
        ],
        "fused": "gelu",
        "removable": 4,
    },
    {
        "name": "instance_norm_activation",
        "pass": "lite_instance_norm_activation_fuse_pass",
        "ops": ["instance_norm", "relu"],
        "edges": [("A", "Out", "B", "X")],
        "conds": [],
        "fused": "instance_norm+relu",
        "removable": 1,
    },
    {
        "name": "transpose_softmax_transpose",
        "pass": "lite_transpose_softmax_transpose_fuse_pass",
        "ops": ["transpose2", "softmax", "transpose2"],
        "edges": [("A", "Out", "B", "X"), ("B", "Out", "C", "X")],
        "conds": [],
        "fused": "fusion_transpose_softmax",
        "removable": 2,
    },
    {
        "name": "shuffle_channel",
        "pass": "lite_shuffle_channel_fuse_pass",
        "ops": ["reshape2", "transpose2", "reshape2"],
        "edges": [("A", "Out", "B", "X"), ("B", "Out", "C", "X")],
        "conds": [],
        "fused": "shuffle_channel",
        "removable": 2,
    },
    {
        "name": "greater_than_cast",
        "pass": "lite_greater_than_cast_fuse_pass",
        "ops": ["greater_than", "cast"],
        "edges": [("A", "Out", "B", "X")],
        "conds": [],
        "fused": "fusion_greater_than_cast",
        "removable": 1,
    },
    {
        "name": "interpolate",
        "pass": "lite_interpolate_fuse_pass",
        "ops": ["nearest_interp|bilinear_interp|nearest_interp_v2|bilinear_interp_v2", "shape"],
        "edges": [("A", "Out", "B", "X")],
        "conds": [],
        "fused": "interpolate",
        "removable": 1,
    },
    {
        "name": "keepdims_convert",
        "pass": "lite_keepdims_convert_fuse_pass",
        "ops": ["arg_max|arg_min|reduce_max|reduce_min|reduce_sum|reduce_mean|reduce_prod|reduce_any|reduce_all"],
        "edges": [],
        "conds": [{"op": "A", "attr": "keepdims", "==": True}],
        "fused": "keepdims_cleanup",
        "removable": 0,
    },
    {
        "name": "p_norm_fill_constant_max_div",
        "pass": "lite_p_norm_fill_constant_max_div_fuse_pass",
        "ops": ["p_norm", "fill_constant", "elementwise_max", "elementwise_div"],
        "edges": [("B", "Out", "C", "Y"), ("C", "Out", "D", "Y")],
        "conds": [],
        "fused": "p_norm_normalized",
        "removable": 2,
    },
]


@dataclass
class FusionHit:
    rule: dict[str, Any]
    op_idxs: list[int]          # 匹配到的 op idx（按规则顺序）
    op_types: list[str]
    via_single_consumer: bool   # 全部连接走唯一消费者边
    confidence: str             # high / medium / low
    shared_cons_count: int = 0  # 最坏连接的消费者数（1=唯一，>1=共享）

    @property
    def removable_op_count(self) -> int:
        """AsIntermediate 顶点数（融合后可省的 op 数）。"""
        return self.rule.get("removable", 0)


def _type_matches(pattern: str, op_type: str) -> bool:
    return op_type in pattern.split("|")


def _port_args(op, port: str) -> list[str]:
    """取 op 指定端口的 var 名列表（兼容 Out/Output 互称）。"""
    for a in op.inputs:
        if a.parameter == port:
            return a.arguments
    for a in op.outputs:
        if a.parameter == port:
            return a.arguments
    return []


def _attr_check(op, cond: dict[str, Any], m) -> bool:
    a = op.attrs.get(cond["attr"])
    if "~=" in cond:
        if a is None:
            return False
        return abs(a.value - cond["~="]) < 1e-5 if isinstance(a.value, (int, float)) else False
    if "==" in cond:
        # 特殊语义：attr == False 时，attr 不存在也视为通过（如 conv 的
        # with_act 缺省即无融合激活）。
        if cond["=="] is False:
            if a is None:
                return True
            return a.value is False
        if a is None:
            return False
        return a.value == cond["=="]
    # 无比较符：仅要求 attr 存在
    return a is not None


def _input_check(op, cond: dict[str, Any], m) -> bool:
    names = _port_args(op, cond["input"])
    if not names:
        return False
    for name in names:
        v = m.vars.get(name)
        if cond.get("persistable"):
            if not (v and v.persistable):
                return False
        if cond.get("persistable") is False:
            if v and v.persistable:
                return False
        if cond.get("only_one_output"):
            if len(m.var_consumers.get(name, [])) != 1:
                return False
        # dims 检查：输入 var 的静态 shape 必须匹配候选之一。候选中的 0
        # 表示通配（任意维度值），如 [0,0,1,1] 匹配任意 [N,C,1,1]。
        # 动态 shape（dims 为空或含 -1）不匹配，保守排除。
        if "dims" in cond:
            candidates = cond["dims"]
            actual = v.dims if v else None
            if not actual:
                return False
            if -1 in actual:
                return False
            if not any(_dims_match(actual, cand) for cand in candidates):
                return False
    return True


def _dims_match(actual: list[int], pattern: list[int]) -> bool:
    """dims 匹配：pattern 中的 0 通配任意值，其余精确相等。"""
    if len(actual) != len(pattern):
        return False
    for a, p in zip(actual, pattern):
        if p != 0 and a != p:
            return False
    return True


def _eval_conds(m, rule: dict[str, Any], ops: list) -> bool:
    for cond in rule.get("conds", []):
        op = ops[cond["op"]] if isinstance(cond["op"], int) else ops[ord(cond["op"]) - 65]
        if "attr" in cond:
            if not _attr_check(op, cond, m):
                return False
        if "input" in cond:
            if not _input_check(op, cond, m):
                return False
    return True


def _match_chains(m, rule: dict[str, Any], start: int):
    """沿 var_consumers 匹配 op 序列（链式，顶点去重）。

    连接判定用 op_input_producers（该 consumer 实际读到的写者，scope 语义），
    正确处理 x2paddle 寄存器池写覆盖。每个起点只产生一条匹配
    （按拓扑序取第一个合格候选），避免共享 var 导致的路径爆炸。
    yield (seq, via_single)。
    """
    ops = m.ops
    types = rule["ops"]
    edges = rule.get("edges", [])
    if not _type_matches(types[0], ops[start].type):
        return
    seq = [start]
    for i in range(1, len(types)):
        cur = ops[seq[-1]]
        # 当前 op 输出 var → 消费者中匹配下一类型的候选（按 idx 排序取第一个）
        candidates: list[int] = []
        for out in cur.outputs:
            for name in out.arguments:
                for c in m.var_consumers.get(name, []):
                    if c in seq:
                        continue
                    if not _type_matches(types[i], ops[c].type):
                        continue
                    # scope 校验：该 consumer 读到的写者确实是当前 op
                    if m.op_input_producers.get((c, _port_of_input(ops[c], name), name)) == cur.idx:
                        candidates.append(c)
        if not candidates:
            return
        seq.append(sorted(candidates)[0])
    # 校验 edges 端口连接
    if edges:
        for (pi, pp, ci, cp) in edges:
            p_idx = seq[ord(pi) - 65]
            c_idx = seq[ord(ci) - 65]
            p_out = _port_args(ops[p_idx], pp)
            c_in = _port_args(ops[c_idx], cp)
            if not (set(p_out) & set(c_in)):
                return
    # 评估 via_single：每个连接 var 是否唯一消费者
    via_single = True
    for i in range(len(seq) - 1):
        p = ops[seq[i]]
        connected = False
        for a in p.outputs:
            for n in a.arguments:
                cons = m.var_consumers.get(n, [])
                if seq[i + 1] in cons:
                    connected = True
                    if len(cons) != 1:
                        via_single = False
        if not connected:
            via_single = False
    yield seq, via_single


def _port_of_input(op, name: str) -> str:
    """找 op 输入 var 所属端口名。"""
    for a in op.inputs:
        if name in a.arguments:
            return a.parameter
    return ""


def scan_fusion(m, include_shared: bool = False, min_confidence: str = "medium") -> list[FusionHit]:
    """扫描全部规则，返回 FusionHit 列表。

    include_shared: 保留 low 级候选（共享 var 且消费者含同类 op 的假阳性），
                    默认排除。
    min_confidence: 最低 confidence 过滤（high/medium/low）。
    confidence 分级：
        high   = 全部连接 var 唯一消费者（必然可融合）
        medium = 有共享 var，但消费者里没有其他同类可融合 op
                 （如 conv→act 共享输出但不冲突，实际可融合）
        low    = 有共享 var 且消费者里有其他同类 op（疑似假阳性）
    """
    order = {"high": 3, "medium": 2, "low": 1}
    hits: list[FusionHit] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for rule in FUSION_RULES:
        first_types = rule["ops"][0].split("|")
        starts: list[int] = []
        for t in first_types:
            starts.extend(m.ops_by_type.get(t, []))
        for s in sorted(set(starts)):
            for seq, via_single in _match_chains(m, rule, s):
                ops = [m.ops[i] for i in seq]
                # ARM 语义过滤：若规则声明了 arm_act_types，最后一个 op
                # （激活）必须属于该集合；否则运行时 pass 会拒绝。
                arm_acts = rule.get("arm_act_types")
                if arm_acts is not None:
                    if not _type_matches("|".join(arm_acts), ops[-1].type):
                        continue
                if not _eval_conds(m, rule, ops):
                    continue
                # 评估共享连接：消费者里有无其他同类可融合 op（scope 语义下
                # 每个 consumer 读到的是自己的写者，共享 var 多消费者不必然冲突）
                sibling_same_type = False
                worst_cons = 0
                for i in range(len(seq) - 1):
                    p = ops[i]
                    for a in p.outputs:
                        for n in a.arguments:
                            cons = m.var_consumers.get(n, [])
                            if seq[i + 1] in cons:
                                worst_cons = max(worst_cons, len(cons))
                                # 消费者中"读到的写者也是 p"的其他同类 op → 假阳性
                                for c in cons:
                                    if c not in seq and _type_matches(
                                            rule["ops"][i + 1], m.ops[c].type):
                                        if m.op_input_producers.get(
                                                (c, _port_of_input(m.ops[c], n), n)) == p.idx:
                                            sibling_same_type = True
                # confidence：
                #   high   = 全部连接唯一消费者（必然可融合）
                #   medium = 有共享 var，但消费者里没有其他同类可融合 op
                #            （如 conv→act 共享输出但不冲突，实际可融合）
                #   low    = 共享且消费者里有其他同类 op（融合有歧义/假阳性）
                if via_single:
                    confidence = "high"
                elif sibling_same_type:
                    confidence = "low"
                else:
                    confidence = "medium"
                if not include_shared and confidence == "low":
                    continue
                if order[confidence] < order[min_confidence]:
                    continue
                key = (rule["name"], tuple(seq))
                if key in seen:
                    continue
                seen.add(key)
                hits.append(FusionHit(
                    rule=rule, op_idxs=seq, op_types=[o.type for o in ops],
                    via_single_consumer=via_single, confidence=confidence,
                    shared_cons_count=worst_cons))
    return hits


def summarize_hits(hits: list[FusionHit]) -> dict[str, int]:
    """按规则名统计命中数。"""
    from collections import Counter
    return dict(Counter(h.rule["name"] for h in hits))
