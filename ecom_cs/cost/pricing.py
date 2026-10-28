"""价格表：按 tier 记账。

离线 mock 的成本记 0，但账本仍然记录 token 与 tier——这样切换到真实模型后，
同一套指标立刻可比。把「成本」当成一等公民指标，是这类系统从 Demo 走向
上线最容易被忽略的一步。
"""

from __future__ import annotations

# 单位：美元 / 1K tokens（输入, 输出）。数值是量级示意，接真实网关时从配置读。
TIER_PRICES: dict[str, tuple[float, float]] = {
    "L0": (0.0, 0.0),          # 工具/模板，无模型调用
    "L1": (0.00015, 0.0006),   # 小模型
    "L2": (0.0025, 0.01),      # 大模型
    "mock": (0.0, 0.0),
}


def estimate_cost(tier: str, prompt_tokens: int, completion_tokens: int) -> float:
    price_in, price_out = TIER_PRICES.get(tier, TIER_PRICES["L2"])
    return (prompt_tokens / 1000.0) * price_in + (completion_tokens / 1000.0) * price_out
