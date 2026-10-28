"""模型级联：把「用哪个模型」变成一个可解释的决策。

判断顺序（越靠前越省）：
1. 工具已经拿到确定性事实，且用户只要事实（订单状态、物流轨迹）-> L0 模板直出；
2. 政策判定为 applicable 且校准置信度足够 -> L1 小模型/模板精修；
3. 其余（意图模糊、要组织语言、要安抚情绪）-> L2 大模型。

值得强调的是 **L0 不是偷懒**：订单状态这种问题用模板回答，正确率 100%、
延迟 2ms、成本 0，比任何模型都强。把「确定性事实」交给模型反而是退步。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..types import Intent, PolicyStatus, Route

L0_INTENTS = {Intent.ORDER_QUERY, Intent.LOGISTICS}


@dataclass
class TierDecision:
    tier: str
    reason: str

    def to_dict(self) -> dict:
        return {"tier": self.tier, "reason": self.reason}


class Cascade:
    def __init__(self, l1_threshold: float = 0.78) -> None:
        self.l1_threshold = l1_threshold

    def decide(self, state) -> TierDecision:
        route = getattr(state, "route", None)
        intent = getattr(state, "intent", None)
        calibrated = float(getattr(state, "calibrated", 0.0))
        policy = getattr(state, "policy", None)
        tools_ok = all(tr.ok for tr in (getattr(state, "tool_results", None) or []))
        has_tools = bool(getattr(state, "tool_results", None))

        if route == Route.SMALLTALK:
            return TierDecision("L1", "闲聊走模板，无需大模型")

        if intent in L0_INTENTS and has_tools and tools_ok:
            return TierDecision("L0", "工具已返回确定性事实，模板直出")

        if policy is not None and policy.status == PolicyStatus.APPLICABLE and calibrated >= self.l1_threshold:
            return TierDecision("L1", f"政策已确定且置信度 {calibrated:.2f} ≥ {self.l1_threshold}")

        return TierDecision("L2", "需要组织语言或意图不确定")
