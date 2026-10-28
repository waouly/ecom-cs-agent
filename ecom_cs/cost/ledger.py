"""成本账本：按轮记录 tier / tokens / 金额 / 延迟，并给出分位数。

客服的 SLA 通常看 P95 延迟而不是均值——一条 8 秒的回复会直接拉扯整天的满意度。
所以账本里存原始样本，汇总时算 p50/p95，而不是只给个平均值糊弄过去。
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CostEntry:
    session_id: str
    tier: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usd: float = 0.0
    latency_ms: float = 0.0
    cache_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id, "tier": self.tier,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "usd": round(self.usd, 6), "latency_ms": round(self.latency_ms, 2),
            "cache_hit": self.cache_hit,
        }


@dataclass
class CostLedger:
    entries: list[CostEntry] = field(default_factory=list)

    def record(self, entry: CostEntry) -> None:
        self.entries.append(entry)

    @staticmethod
    def _pct(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        vs = sorted(values)
        idx = min(len(vs) - 1, max(0, int(round(q * (len(vs) - 1)))))
        return vs[idx]

    def summary(self) -> dict[str, Any]:
        n = len(self.entries)
        if n == 0:
            return {"turns": 0, "usd_total": 0.0, "usd_avg": 0.0,
                    "latency_p50_ms": 0.0, "latency_p95_ms": 0.0,
                    "cache_hit_rate": 0.0, "by_tier": {}}

        latencies = [e.latency_ms for e in self.entries]
        by_tier: dict[str, dict[str, Any]] = {}
        for e in self.entries:
            slot = by_tier.setdefault(e.tier, {"n": 0, "usd": 0.0, "tokens": 0})
            slot["n"] += 1
            slot["usd"] = round(slot["usd"] + e.usd, 6)
            slot["tokens"] += e.prompt_tokens + e.completion_tokens

        return {
            "turns": n,
            "usd_total": round(sum(e.usd for e in self.entries), 6),
            "usd_avg": round(sum(e.usd for e in self.entries) / n, 6),
            "latency_p50_ms": round(self._pct(latencies, 0.50), 2),
            "latency_p95_ms": round(self._pct(latencies, 0.95), 2),
            "latency_mean_ms": round(statistics.fmean(latencies), 2),
            "cache_hit_rate": round(sum(1 for e in self.entries if e.cache_hit) / n, 4),
            "by_tier": by_tier,
        }
