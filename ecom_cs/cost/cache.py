"""语义缓存：相同的问法不必走第二遍模型。

三条安全边界（缓存最容易出事的地方）：
1. **只缓存与用户身份无关的答案**。订单状态、物流轨迹绝不能进缓存，
   否则 A 用户会看到 B 用户的订单。所以 scope 只允许 policy / smalltalk。
2. **阈值要高**。0.95 以上才命中，宁可多一次调用也不要答非所问。
3. **带 TTL 且可主动失效**。政策更新后旧答案必须尽快退场，
   ``invalidate`` 供知识更新流程调用。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import numpy as np

from ..utils.text import cosine

CACHEABLE_SCOPES = {"policy", "smalltalk"}


@dataclass
class CacheHit:
    reply: str
    score: float
    citations: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Entry:
    query: str
    vector: np.ndarray
    reply: str
    citations: list[dict[str, Any]]
    meta: dict[str, Any]
    ts: float
    hits: int = 0


class SemanticCache:
    def __init__(self, embedder=None, threshold: float = 0.95, ttl_s: float = 3600.0,
                 max_entries: int = 2048) -> None:
        self.embedder = embedder
        self.threshold = threshold
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self._entries: list[_Entry] = []
        self._exact: dict[str, _Entry] = {}
        self.hits = 0
        self.misses = 0

    # -- 读写 --
    def get(self, query: str, scope: str) -> Optional[CacheHit]:
        if scope not in CACHEABLE_SCOPES:
            self.misses += 1
            return None
        self._evict()
        entry = self._exact.get(query)
        if entry is not None:
            entry.hits += 1
            self.hits += 1
            return CacheHit(reply=entry.reply, score=1.0,
                            citations=entry.citations, meta={**entry.meta, "exact": True})

        if self.embedder is None or not self._entries:
            self.misses += 1
            return None
        qv = np.asarray(self.embedder.encode_one(query), dtype=np.float32)
        best: Optional[_Entry] = None
        best_score = 0.0
        for e in self._entries:
            if e.meta.get("scope") != scope:
                continue
            s = cosine(qv, e.vector)
            if s > best_score:
                best, best_score = e, s
        if best is not None and best_score >= self.threshold:
            best.hits += 1
            self.hits += 1
            return CacheHit(reply=best.reply, score=best_score,
                            citations=best.citations, meta={**best.meta, "exact": False})
        self.misses += 1
        return None

    def put(self, query: str, reply: str, scope: str,
            citations: Optional[list[dict[str, Any]]] = None,
            meta: Optional[dict[str, Any]] = None) -> None:
        if scope not in CACHEABLE_SCOPES or self.embedder is None:
            return
        if query in self._exact:
            return
        vec = np.asarray(self.embedder.encode_one(query), dtype=np.float32)
        entry = _Entry(
            query=query, vector=vec, reply=reply,
            citations=list(citations or []),
            meta={**(meta or {}), "scope": scope},
            ts=time.time(),
        )
        self._entries.append(entry)
        self._exact[query] = entry
        if len(self._entries) > self.max_entries:
            self._entries.sort(key=lambda e: e.hits, reverse=True)
            self._entries = self._entries[: self.max_entries]
            self._exact = {e.query: e for e in self._entries}

    # -- 维护 --
    def _evict(self) -> None:
        if not self._entries:
            return
        now = time.time()
        alive = [e for e in self._entries if now - e.ts <= self.ttl_s]
        if len(alive) != len(self._entries):
            self._entries = alive
            self._exact = {e.query: e for e in alive}

    def invalidate(self) -> int:
        n = len(self._entries)
        self._entries.clear()
        self._exact.clear()
        return n

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "entries": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
            "threshold": self.threshold,
        }
