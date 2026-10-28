"""BM25 稀疏检索（纯 Python 实现，零依赖）。

为什么自己写而不装 rank_bm25：这个项目的卖点是「离线可跑、无重依赖」，
而 BM25 只有 60 行。自己写还顺便能把 IDF 的计算口径暴露出来，方便在评测里
解释「为什么这条召回失败了」——黑盒库里你只能看到分数，看不到分子分母。
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class BM25:
    corpus: list[list[str]]
    k1: float = 1.5
    b: float = 0.75
    tf: list[Counter] = field(default_factory=list)
    idf: dict[str, float] = field(default_factory=dict)
    avgdl: float = 0.0

    def __post_init__(self) -> None:
        self.tf = [Counter(doc) for doc in self.corpus]
        n = len(self.corpus)
        self.avgdl = (sum(len(d) for d in self.corpus) / n) if n else 0.0
        df: Counter = Counter()
        for doc in self.corpus:
            for term in set(doc):
                df[term] += 1
        self.idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def score(self, query_tokens: list[str], idx: int) -> float:
        doc_tf = self.tf[idx]
        dl = len(self.corpus[idx])
        if not dl:
            return 0.0
        total = 0.0
        for term in query_tokens:
            freq = doc_tf.get(term, 0)
            if not freq:
                continue
            idf = self.idf.get(term, 0.0)
            denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
            total += idf * freq * (self.k1 + 1) / denom
        return total

    def search(self, query_tokens: list[str], top_k: int = 20) -> list[tuple[int, float]]:
        if not query_tokens or not self.corpus:
            return []
        scores = [(i, self.score(query_tokens, i)) for i in range(len(self.corpus))]
        scores = [x for x in scores if x[1] > 0]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def normalize_scores(hits: list[tuple[int, float]]) -> list[tuple[int, float]]:
    if not hits:
        return []
    top = max(s for _, s in hits) or 1.0
    return [(i, s / top) for i, s in hits]
