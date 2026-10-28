"""切分策略：条款级小块 + 文档级父块（small-to-big）。

为什么不用定长 token 切分：
一条政策条款是一个不可分割的判定单元。「自签收之日起 7 天内、商品未拆封」
如果被切成两段，召回时可能只拿到「商品未拆封」而没有「7 天内」，模型就会
把适用条件答错。所以这里以 clause 为最小检索单元，父块只用于给生成阶段
提供上下文，不参与相似度计算——召回要精准，生成要上下文。
"""

from __future__ import annotations

from typing import Iterable

from .models import Chunk, PolicyDoc


def build_chunks(docs: Iterable[PolicyDoc]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        for clause in doc.clauses:
            exception_text = " ".join(e.text for e in clause.exceptions if e.text)
            parent = " ".join(x for x in (doc.overview, clause.text, exception_text) if x)
            chunks.append(
                Chunk(
                    chunk_id=clause.clause_id,
                    doc_id=doc.doc_id,
                    title=clause.title,
                    text=clause.text,
                    parent_text=parent,
                    source="policy" if doc.doc_id != "FAQ-001" else "faq",
                    keywords=list(clause.keywords),
                    effective_from=clause.effective_from,
                    effective_to=clause.effective_to,
                )
            )
    return chunks


def chunk_stats(chunks: list[Chunk]) -> dict[str, float]:
    if not chunks:
        return {"n": 0, "avg_len": 0, "max_len": 0, "min_len": 0}
    lens = [len(c.text) for c in chunks]
    return {
        "n": len(chunks),
        "avg_len": sum(lens) / len(lens),
        "max_len": max(lens),
        "min_len": min(lens),
    }
