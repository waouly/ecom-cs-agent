"""成本工程：语义缓存 + 模型级联 + 成本账本。

客服场景的成本命门在于「90% 的请求是重复的」：同一句「退款多久到账」一天被问
几千次。所以第一优先级不是换更便宜的模型，而是**根本不调用模型**。

- L0：工具/规则能直接给结论 -> 模板渲染，0 次 LLM 调用；
- L1：高置信度的政策问答 -> 小模型或模板精修；
- L2：意图模糊、需要组织语言 -> 大模型。
这一层的价值在评测里直接可见：平均成本、P95 延迟都会掉一个量级。
"""

from __future__ import annotations

from .cache import CacheHit, SemanticCache
from .cascade import Cascade, TierDecision
from .ledger import CostEntry, CostLedger
from .pricing import TIER_PRICES, estimate_cost

__all__ = [
    "CacheHit", "SemanticCache",
    "Cascade", "TierDecision",
    "CostEntry", "CostLedger",
    "TIER_PRICES", "estimate_cost",
]
