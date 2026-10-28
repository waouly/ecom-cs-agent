"""置信度融合：四信号加权 -> 政策状态修正 -> 校准。

简历里写的是「四信号加权置信度评分」，这里把它做成可校准的连续分数：
原始融合分只是排序信号，真正与「答对概率」挂钩的是经过 Platt 校准后的值。
不校准的阈值在业务上是很危险的东西——0.62 在淡季可能对应 95% 正确率，
在大促知识更新后可能只有 80%，而阈值本身不会告诉你这件事。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..types import PolicyStatus, Signals
from ..utils.text import clamp

DEFAULT_WEIGHTS: dict[str, float] = {
    "top1": 0.40,
    "margin": 0.22,
    "intent": 0.22,
    "tool": 0.16,
}


@dataclass
class ConfidenceScorer:
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    def signals(self, state) -> Signals:
        ev = getattr(state, "evidence", None) or []
        top1 = 0.0
        margin = 0.0
        if ev:
            # rerank 分在 [0,1] 但实际集中在 0.2~0.7，开方拉开分辨率
            top1 = clamp(float(ev[0].rerank_score) ** 0.5)
            if len(ev) >= 2 and ev[0].rerank_score > 0:
                margin = clamp((ev[0].rerank_score - ev[1].rerank_score) / ev[0].rerank_score)
            else:
                margin = 0.6  # 只有一条证据时按中等唯一性处理
        return Signals(
            top1=top1,
            margin=margin,
            intent=clamp(float(getattr(state, "intent_confidence", 0.0))),
            tool_completeness=clamp(float((getattr(state, "meta", {}) or {}).get("tool_completeness", 0.0))),
        )

    def fuse(
        self,
        signals: Signals,
        *,
        has_evidence: bool,
        policy_status: Optional[PolicyStatus] = None,
    ) -> float:
        s = signals
        base = (
            self.weights["top1"] * s.top1
            + self.weights["margin"] * s.margin
            + self.weights["intent"] * s.intent
            + self.weights["tool"] * s.tool_completeness
        )
        # 结构化判定优于文本匹配：规则引擎给出确定结论时，应该提升置信度
        if policy_status == PolicyStatus.APPLICABLE:
            base += 0.12
        elif policy_status == PolicyStatus.NOT_APPLICABLE:
            base += 0.04
        elif policy_status == PolicyStatus.UNKNOWN:
            base -= 0.10
        if not has_evidence:
            base = min(base, 0.45)
        return clamp(base)


def describe_confidence(value: float) -> str:
    if value >= 0.8:
        return "高"
    if value >= 0.62:
        return "中"
    if value >= 0.45:
        return "偏低"
    return "低"
