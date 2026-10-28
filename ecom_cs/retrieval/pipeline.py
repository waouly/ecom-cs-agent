"""检索服务：把「改写 -> 检索 -> 合并」串成一个门面，供编排层调用。"""

from __future__ import annotations

import time
from datetime import date
from typing import Optional

from ..kb.policy_engine import PolicyEngine
from ..types import Evidence
from .hybrid import HybridRetriever, RetrievalResult, build_retriever
from .rewrite import rewrite_query, split_questions


class RetrievalService:
    def __init__(self, retriever: HybridRetriever, engine: PolicyEngine, settings=None) -> None:
        self.retriever = retriever
        self.engine = engine
        self.settings = settings

    @classmethod
    def build(cls, engine: PolicyEngine, embedder, settings) -> "RetrievalService":
        return cls(build_retriever(engine, embedder, settings), engine, settings)

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        order_no: Optional[str] = None,
        today: Optional[date] = None,
    ) -> RetrievalResult:
        started = time.perf_counter()
        rw = rewrite_query(query, order_no=order_no)
        parts = split_questions(rw.rewritten)

        all_evidence: list[Evidence] = []
        merged_dense: dict[int, float] = {}
        merged_sparse: dict[int, float] = {}
        for part in parts:
            ev, dbg = self.retriever.retrieve(part, top_k=top_k, today=today)
            all_evidence.extend(ev)
            for i, s in dbg.get("dense", []):
                merged_dense[i] = max(merged_dense.get(i, 0.0), s)
            for i, s in dbg.get("sparse", []):
                merged_sparse[i] = max(merged_sparse.get(i, 0.0), s)

        # 同一条款被多个子问题命中时取最好名次，避免重复占位
        best: dict[str, Evidence] = {}
        for e in all_evidence:
            cur = best.get(e.clause_id)
            if cur is None or e.rank < cur.rank:
                best[e.clause_id] = e
        final = sorted(best.values(), key=lambda e: (e.rank, -e.rerank_score))
        final = final[:(top_k or getattr(self.settings, "top_k_final", 5))]
        for i, e in enumerate(final):
            e.rank = i

        return RetrievalResult(
            query=query,
            evidence=final,
            dense_hits=sorted(merged_dense.items(), key=lambda kv: -kv[1]),
            sparse_hits=sorted(merged_sparse.items(), key=lambda kv: -kv[1]),
            rewritten=rw.rewritten,
            aliases_hit=rw.aliases_hit,
            coref_replaced=rw.coref_replaced,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    def stats(self) -> dict:
        return {
            "chunks": len(self.retriever.chunks),
            "dense": getattr(self.retriever, "store", None) is not None,
            "sparse": getattr(self.retriever, "bm25", None) is not None,
            "rerank": getattr(self.retriever, "reranker", None) is not None,
        }
