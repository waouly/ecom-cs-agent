"""BGE-M3 向量化适配器（可选依赖）。

装好 sentence-transformers 后：
    pip install "ecom-cs-agent[real-embed]"
    set ECOM_CS_EMBED_PROVIDER=bge
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import l2_normalize


class BGEEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - 可选依赖
            raise ImportError(
                "未安装 sentence-transformers，无法使用 BGE 向量化。"
                "请 pip install sentence-transformers，或把 ECOM_CS_EMBED_PROVIDER 设为 hashing。"
            ) from e
        self._model = SentenceTransformer(model_name, device=device)
        self.name = model_name
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        vecs = self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype="float32")

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
