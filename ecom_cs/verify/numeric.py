"""数字守卫：回复里出现的每个数字都必须能在证据/工具数据里找到出处。

这是「生成后校验」里性价比最高的一环。客服答错数字（时效、金额、天数）是
最贵的错误，而数字恰恰是最容易核对的东西——模型写的「7 个工作日」如果在
条款和工具返回值里都找不到，几乎可以断定是编的（幻觉或者串台）。

实现上不做语义对齐，只做**存在性核对**，宁可误拦不可误放：误拦的代价是
多一次转人工，误放的代价是赔付纠纷。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

NUM_WITH_UNIT = re.compile(r"(\d+(?:\.\d+)?)\s*(个?工作日|个?自然日|天|小时|分钟|元|块|%|％|折|次|年|个月)")

# 这些数字是格式性的，不参与核对（编号、列表序号等）
IGNORABLE = re.compile(r"^\d{1,2}$")


@dataclass
class NumericViolation:
    token: str
    value: float
    unit: str
    reason: str = "not_found_in_evidence"

    def __str__(self) -> str:
        return f"{self.token}（证据中未出现）"


def extract_claims(reply: str) -> list[tuple[str, float, str]]:
    out: list[tuple[str, float, str]] = []
    for m in NUM_WITH_UNIT.finditer(reply or ""):
        out.append((m.group(0).strip(), float(m.group(1)), m.group(2)))
    return out


def _normalize_unit(unit: str) -> str:
    if unit in {"个工作日", "工作日"}:
        return "工作日"
    if unit in {"个自然日", "自然日", "天"}:
        return "天"
    if unit in {"%", "％"}:
        return "%"
    return unit


def check_numbers(
    reply: str,
    contexts: Iterable[str],
    allow_values: Iterable[float] = (),
) -> list[NumericViolation]:
    """contexts 是证据正文、工具返回值的可读文本、用户输入。"""
    haystack = "\n".join(c for c in contexts if c)
    allowed = set(allow_values)
    violations: list[NumericViolation] = []

    for token, value, unit in extract_claims(reply):
        norm_unit = _normalize_unit(unit)
        # 1) 完全一致的写法
        if token in haystack:
            continue
        # 2) 数值 + 单位分别出现（容忍「7天」vs「7 天」）
        number_forms = {f"{value:g}", f"{value:.0f}", f"{value:.2f}"}
        unit_forms = {norm_unit, unit}
        if any(nf in haystack for nf in number_forms) and any(uf in haystack for uf in unit_forms):
            continue
        # 3) 计算得来的数（如退款金额 = 订单金额），由调用方显式放行
        if value in allowed:
            continue
        if IGNORABLE.match(str(int(value)) if value == int(value) else ""):
            # 单/双位整数且无歧义单位时放宽（「3 步」「2 次」这类操作说明）
            if norm_unit in {"次", "步"}:
                continue
        violations.append(NumericViolation(token=token, value=value, unit=norm_unit))

    return violations


def numeric_context(state) -> list[str]:
    """收集本轮所有可作为数字出处的文本。"""
    parts: list[str] = []
    for ev in getattr(state, "evidence", []) or []:
        parts.append(ev.text or "")
        if getattr(ev, "parent_text", ""):
            parts.append(ev.parent_text)
    for tr in getattr(state, "tool_results", []) or []:
        for v in (tr.data or {}).values():
            if isinstance(v, (int, float, str)):
                parts.append(str(v))
            elif isinstance(v, dict):
                parts.extend(str(x) for x in v.values() if isinstance(x, (int, float, str)))
    for slot in (getattr(state, "slots", {}) or {}).values():
        if isinstance(slot.value, (int, float, str)):
            parts.append(str(slot.value))
    return parts
