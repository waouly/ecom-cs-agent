"""向量化抽象。"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    dim: int
    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_one(self, text: str) -> np.ndarray: ...


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype("float32")
