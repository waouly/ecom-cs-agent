"""意图分类：注入检测 -> 词典打分 -> 样例句向量 -> 融合 + softmax 置信度。

置信度不是拍脑袋的常数，而是 softmax 概率——这样它才能进后面的校准环节
（保形预测 / Platt scaling 需要有区分度的连续分数，常数置信度没法校准）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..types import Intent
from ..utils.text import normalize, sigmoid
from .lexicon import INJECTION_PATTERNS, INTENT_EXEMPLARS, INTENT_LEXICON

_INJ = [(re.compile(p, re.IGNORECASE), name) for p, name in INJECTION_PATTERNS]


@dataclass
class IntentResult:
    intent: Intent
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)
    method: str = "lexicon"
    matched: list[str] = field(default_factory=list)
    injection_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "intent": self.intent.value,
            "confidence": round(self.confidence, 4),
            "method": self.method,
            "matched": self.matched,
            "injection_reason": self.injection_reason,
        }


class IntentClassifier:
    def __init__(self, embedder=None, lex_weight: float = 0.65) -> None:
        self.embedder = embedder
        self.lex_weight = lex_weight
        self._centroids: Optional[np.ndarray] = None
        self._intent_order: list[Intent] = []
        if embedder is not None:
            self._build_centroids()

    def _build_centroids(self) -> None:
        vectors = []
        order = []
        for intent, examples in INTENT_EXEMPLARS.items():
            mat = self.embedder.encode(examples)
            centroid = np.asarray(mat, dtype=np.float32).mean(axis=0)
            norm = np.linalg.norm(centroid) or 1.0
            vectors.append(centroid / norm)
            order.append(intent)
        self._centroids = np.vstack(vectors)
        self._intent_order = order

    # -- 信号 --
    def detect_injection(self, text: str) -> Optional[str]:
        for pattern, name in _INJ:
            if pattern.search(text):
                return name
        return None

    def lexical_scores(self, text: str) -> tuple[dict[str, float], list[str]]:
        norm = normalize(text)
        scores: dict[str, float] = {}
        matched: list[str] = []
        for intent, table in INTENT_LEXICON.items():
            total = 0.0
            for word, weight in table.items():
                if word in norm:
                    total += weight
                    matched.append(f"{intent.value}:{word}")
            if total:
                scores[intent.value] = total
        top = max(scores.values()) if scores else 0.0
        if top:
            scores = {k: v / top for k, v in scores.items()}
        return scores, matched

    def embedding_scores(self, text: str) -> dict[str, float]:
        if self.embedder is None or self._centroids is None:
            return {}
        q = np.asarray(self.embedder.encode_one(text), dtype=np.float32)
        norm = np.linalg.norm(q) or 1.0
        q = q / norm
        sims = self._centroids @ q
        # 余弦在 [-1,1]，映射到 [0,1] 便于融合
        scaled = (sims + 1.0) / 2.0
        return {intent.value: float(s) for intent, s in zip(self._intent_order, scaled)}

    # -- 主入口 --
    def classify(self, text: str) -> IntentResult:
        norm = normalize(text)
        reason = self.detect_injection(norm)
        if reason:
            return IntentResult(
                intent=Intent.INJECTION,
                confidence=0.97,
                scores={Intent.INJECTION.value: 1.0},
                method="rule",
                matched=[reason],
                injection_reason=reason,
            )

        lex, matched = self.lexical_scores(norm)
        emb = self.embedding_scores(norm)

        all_intents = [i.value for i in Intent if i not in {Intent.INJECTION}]
        fused: dict[str, float] = {}
        for name in all_intents:
            l = lex.get(name, 0.0)
            e = emb.get(name, 0.0)
            if emb:
                fused[name] = self.lex_weight * l + (1 - self.lex_weight) * e
            else:
                fused[name] = l

        method = "lexicon+embedding" if emb else "lexicon"
        best = max(fused, key=lambda k: fused[k])

        # softmax 让置信度带温度：区分度低的输入自然得到低置信度
        values = np.array([fused[k] for k in all_intents], dtype=np.float32)
        temp = 0.08
        z = np.exp((values - values.max()) / temp)
        probs = z / z.sum()
        conf = float(probs[all_intents.index(best)])

        # 词典完全没命中且向量也不够像 -> 判为超出范围，主动兜底而不是硬猜
        if not matched and (not emb or fused[best] < 0.55):
            return IntentResult(
                intent=Intent.OUT_OF_SCOPE,
                confidence=round(min(0.5, 1.0 - fused[best]), 4),
                scores={k: round(v, 4) for k, v in fused.items()},
                method=method,
            )

        return IntentResult(
            intent=Intent(best),
            confidence=round(min(0.995, max(0.05, conf)), 4),
            scores={k: round(v, 4) for k, v in fused.items()},
            method=method,
            matched=matched[:8],
        )


def margin_confidence(scores: dict[str, float]) -> float:
    """Top1 与 Top2 的间距，越大说明意图越唯一。"""
    if len(scores) < 2:
        return 1.0
    ordered = sorted(scores.values(), reverse=True)
    gap = ordered[0] - ordered[1]
    return float(sigmoid(gap * 8))
