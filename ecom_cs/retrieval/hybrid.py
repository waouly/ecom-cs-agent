"""混合检索：dense + sparse -> RRF 融合 -> 精排 -> 条款级证据。

几个刻意的选择：

1. **RRF 而不是加权分数融合**。向量余弦相似度和 BM25 分数的量纲完全不同，
   直接加权需要对每个查询做归一化，而归一化本身依赖候选集分布，很不稳定。
   RRF 只用排名，天然免疫量纲问题，代价参数只有一个 k，属于「几乎不会调坏」的方案。
2. **融合后仍然精排**。RRF 解决「谁该进候选」，精排解决「谁该排第一」。
   客服场景里 Top-1 决定答案对错，所以这一步不能省。
3. **召回单元是条款，不是 chunk 窗口**。父块随证据一起返回给生成阶段，
   检索只看小块——召回要精准，生成要上下文（small-to-big）。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np

from ..kb.chunker import build_chunks
from ..kb.facts import parse_date
from ..kb.models import Chunk
from ..kb.policy_engine import PolicyEngine
from ..types import Evidence
from ..utils.text import tokenize
from .bm25 import BM25, normalize_scores
from .rerank import build_reranker
from .vector import build_vector_store


def rrf_fuse(rankings: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion：score(d) = sum 1/(k + rank)。"""
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (idx, _score) in enumerate(ranking):
            scores[idx] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def weighted_fuse(
    rankings: list[list[tuple[int, float]]],
    weights: list[float],
) -> list[tuple[int, float]]:
    scores: dict[int, float] = defaultdict(float)
    for ranking, w in zip(rankings, weights):
        norm = normalize_scores(ranking)
        for idx, s in norm:
            scores[idx] += w * s
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


@dataclass
class RetrievalResult:
    query: str
    evidence: list[Evidence] = field(default_factory=list)
    dense_hits: list[tuple[int, float]] = field(default_factory=list)
    sparse_hits: list[tuple[int, float]] = field(default_factory=list)
    fused: list[tuple[int, float]] = field(default_factory=list)
    rewritten: str = ""
    aliases_hit: list[str] = field(default_factory=list)
    coref_replaced: bool = False
    latency_ms: float = 0.0

    @property
    def top1(self) -> Optional[Evidence]:
        return self.evidence[0] if self.evidence else None

    def clause_ids(self) -> list[str]:
        return [e.clause_id for e in self.evidence]


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        embedder,
        settings,
        engine: Optional[PolicyEngine] = None,
    ) -> None:
        self.chunks = chunks
        self.embedder = embedder
        self.settings = settings
        self.engine = engine
        self._clause_keywords = self._collect_keywords(engine)
        self._build()

    # -- 建索引 --
    def _collect_keywords(self, engine: Optional[PolicyEngine]) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        if engine is None:
            return out
        for c in engine.clauses:
            out[c.clause_id] = c.keywords
        return out

    def _index_text(self, c: Chunk) -> str:
        return f"{c.title} {c.text} {' '.join(c.keywords)}"

    def _build(self) -> None:
        s = self.settings
        self.enable_dense = bool(getattr(s, "enable_dense", True))
        self.enable_sparse = bool(getattr(s, "enable_bm25", True))
        self.enable_rerank = bool(getattr(s, "enable_rerank", True))
        self.top_k_dense = int(getattr(s, "top_k_dense", 20))
        self.top_k_sparse = int(getattr(s, "top_k_sparse", 20))
        self.top_k_final = int(getattr(s, "top_k_final", 5))
        self.rrf_k = int(getattr(s, "rrf_k", 60))
        self.fusion = getattr(s, "fusion", "rrf")

        texts = [self._index_text(c) for c in self.chunks]
        if self.enable_dense and self.chunks:
            matrix = self.embedder.encode(texts)
            self.vectors = np.asarray(matrix, dtype=np.float32)
            self.store = build_vector_store(s, self.vectors)
        else:
            self.store = None
        if self.enable_sparse and self.chunks:
            self.bm25: Optional[BM25] = BM25([tokenize(t) for t in texts])
        else:
            self.bm25 = None
        self.reranker = build_reranker(s) if self.enable_rerank else None

    # -- 过滤 --
    def _is_active(self, chunk: Chunk, today: date) -> bool:
        start = parse_date(chunk.effective_from)
        end = parse_date(chunk.effective_to)
        if start and today < start:
            return False
        if end and today > end:
            return False
        return True

    # -- 检索 --
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        today: Optional[date] = None,
    ) -> tuple[list[Evidence], dict[str, list[tuple[int, float]]]]:
        top_k = top_k or self.top_k_final
        today = today or date.today()
        active = [i for i, c in enumerate(self.chunks) if self._is_active(c, today)]
        if not active:
            return [], {}
        active_set = set(active)

        dense_hits: list[tuple[int, float]] = []
        sparse_hits: list[tuple[int, float]] = []

        if self.store is not None:
            qv = self.embedder.encode_one(query)
            dense_hits = [(i, s) for i, s in self.store.search(qv, self.top_k_dense) if i in active_set]

        if self.bm25 is not None:
            sparse_hits = [(i, s) for i, s in self.bm25.search(tokenize(query), self.top_k_sparse) if i in active_set]

        rankings = [r for r in (dense_hits, sparse_hits) if r]
        if not rankings:
            return [], {"dense": dense_hits, "sparse": sparse_hits}

        if len(rankings) == 1:
            fused = rankings[0]
        elif self.fusion == "weighted":
            fused = weighted_fuse(rankings, [0.6, 0.4] if len(rankings) == 2 else [1.0 / len(rankings)] * len(rankings))
        else:
            fused = rrf_fuse(rankings, k=self.rrf_k)

        pool_size = min(len(fused), max(top_k * 4, top_k + 5))
        pool = fused[:pool_size]

        if self.reranker is not None:
            scored: list[tuple[int, float]] = []
            for idx, _f in pool:
                c = self.chunks[idx]
                score = self.reranker.score(
                    query, c.text, title=c.title,
                    keywords=self._clause_keywords.get(c.chunk_id) or c.keywords,
                )
                scored.append((idx, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            pool = scored
        else:
            # 没精排时用融合分当排序分
            pool = [(i, max(0.0, 1.0 - rank / max(1, pool_size))) for rank, (i, _s) in enumerate(pool)]

        dense_map = dict(dense_hits)
        sparse_map = dict(sparse_hits)
        fuse_map = dict(fused)

        evidence: list[Evidence] = []
        for rank, (idx, score) in enumerate(pool[:top_k]):
            c = self.chunks[idx]
            evidence.append(
                Evidence(
                    doc_id=c.doc_id,
                    clause_id=c.chunk_id,
                    parent_id=c.doc_id,
                    title=c.title,
                    text=c.text,
                    parent_text=c.parent_text,
                    source=c.source,
                    dense_score=round(dense_map.get(idx, 0.0), 4),
                    sparse_score=round(sparse_map.get(idx, 0.0), 4),
                    fusion_score=round(fuse_map.get(idx, 0.0), 6),
                    rerank_score=round(score, 4),
                    rank=rank,
                    effective_from=c.effective_from,
                    effective_to=c.effective_to or "",
                )
            )
        return evidence, {"dense": dense_hits, "sparse": sparse_hits}


def build_retriever(engine: PolicyEngine, embedder, settings) -> HybridRetriever:
    chunks = build_chunks(engine.docs)
    return HybridRetriever(chunks, embedder, settings, engine=engine)
