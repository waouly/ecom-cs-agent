from __future__ import annotations

from ..config import Settings
from .base import Embedder, l2_normalize
from .hashing import HashingEmbedder

__all__ = ["Embedder", "HashingEmbedder", "l2_normalize", "build_embedder"]


def build_embedder(settings: Settings) -> Embedder:
    provider = (settings.embed_provider or "hashing").lower()
    if provider == "hashing":
        return HashingEmbedder(dim=settings.embed_dim)
    if provider == "bge":
        from .bge import BGEEmbedder

        return BGEEmbedder(model_name=settings.embed_model)
    raise ValueError(f"未知 embed provider: {settings.embed_provider}")
