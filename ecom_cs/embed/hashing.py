"""离线哈希向量化（signed feature hashing）。

不依赖任何模型权重，但对中文 bigram/trigram 的词汇重叠敏感，足以支撑
「同一类问法召回同一条政策」这个演示目标，且完全确定性、可复现。
接真实模型时换 ``BGEEmbedder`` 即可，接口一致。
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Sequence

import numpy as np

from ..utils.text import normalize
from .base import l2_normalize


class HashingEmbedder:
    name = "hashing-bigram"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _features(self, text: str) -> Counter:
        t = normalize(text)
        t = "".join(ch for ch in t if not ch.isspace())
        feats: Counter = Counter()
        for n in (2, 3):
            for i in range(max(0, len(t) - n + 1)):
                feats[f"{n}:{t[i:i + n]}"] += 1
        for word in normalize(text).split():
            feats[f"w:{word}"] += 1
        return feats

    def _hash(self, feat: str) -> tuple[int, float]:
        h = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "little") % self.dim
        sign = 1.0 if (h[4] & 1) else -1.0
        return idx, sign

    def encode_one(self, text: str) -> np.ndarray:
        out = np.zeros(self.dim, dtype="float32")
        for feat, count in self._features(text).items():
            idx, sign = self._hash(feat)
            out[idx] += sign * (1.0 + np.log1p(count))
        norm = float(np.linalg.norm(out))
        if norm > 0:
            out /= norm
        return out

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        mat = np.stack([self.encode_one(t) for t in texts])
        return l2_normalize(mat)
