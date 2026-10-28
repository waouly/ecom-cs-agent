"""向量库：内存实现（默认，离线可跑）+ Milvus 适配层（配 URI 即切真实）。

内存实现不是玩具：条款语料是几十到几千条量级，numpy 点积在 1ms 内出结果，
比走网络到 Milvus 更快也更稳。真正的价值在于**接口一致**——上层检索流程
不需要知道底下是 numpy 还是 Milvus，切换只改一行配置。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class VectorStore(Protocol):
    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]: ...

    def size(self) -> int: ...


class MemoryVectorStore:
    def __init__(self, vectors: np.ndarray) -> None:
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"vectors 需要二维数组，收到 {arr.shape}")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.vectors = arr / norms

    def search(self, query_vector: np.ndarray, top_k: int = 20) -> list[tuple[int, float]]:
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(q) or 1.0
        q = q / norm
        if self.vectors.shape[0] == 0:
            return []
        sims = self.vectors @ q
        k = min(top_k, sims.shape[0])
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(int(i), float(sims[i])) for i in idx]

    def size(self) -> int:
        return int(self.vectors.shape[0])


class MilvusVectorStore:
    """Milvus 适配层。未安装 pymilvus 或未配置 URI 时抛出可读错误。

    使用前提：``pip install pymilvus`` 且 ``ECOM_CS_VECTOR_BACKEND=milvus``。
    """

    def __init__(
        self,
        uri: str,
        collection: str = "ecom_cs_policy",
        dim: int = 512,
        chunks: list[str] | None = None,
    ) -> None:
        try:
            from pymilvus import (  # type: ignore
                Collection, CollectionSchema, DataType, FieldSchema, connections, utility,
            )
        except ImportError as e:  # pragma: no cover - 可选依赖
            raise RuntimeError(
                "vector_backend=milvus 需要先安装 pymilvus：pip install pymilvus"
            ) from e
        self._conn = connections.connect(alias="default", uri=uri)
        self.collection_name = collection
        self.dim = dim
        if utility.has_collection(collection):
            self.collection = Collection(collection)
        else:
            fields = [
                FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
            ]
            schema = CollectionSchema(fields=fields, description="ecom-cs policy clauses")
            self.collection = Collection(collection, schema=schema)
            self.collection.create_index(
                field_name="vector",
                index_params={"index_type": "FLAT", "metric_type": "COSINE"},
            )
            if chunks:
                self.index_texts(chunks, None)
        self.collection.load()

    def index_texts(self, texts: list[str], vectors: np.ndarray | None = None) -> None:
        if vectors is None:
            raise RuntimeError("Milvus 索引需要显式传入 vectors")
        rows = [[i, vectors[i].tolist(), texts[i][:2048]] for i in range(len(texts))]
        self.collection.insert(rows)

    def search(self, query_vector: np.ndarray, top_k: int = 20) -> list[tuple[int, float]]:
        res = self.collection.search(
            data=[np.asarray(query_vector, dtype=np.float32).reshape(-1).tolist()],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=top_k,
            output_fields=["pk"],
        )
        hits: list[tuple[int, float]] = []
        for hit in res[0]:
            hits.append((int(hit.entity.get("pk")), float(hit.score)))
        return hits

    def size(self) -> int:
        return int(self.collection.num_entities)


def build_vector_store(settings, vectors: np.ndarray, chunks: list[str] | None = None):
    backend = getattr(settings, "vector_backend", "memory")
    if backend == "milvus":
        return MilvusVectorStore(
            uri=getattr(settings, "milvus_uri", "") or "http://127.0.0.1:19530",
            collection=getattr(settings, "milvus_collection", "ecom_cs_policy"),
            dim=int(getattr(settings, "embed_dim", 512)),
            chunks=chunks,
        )
    return MemoryVectorStore(vectors)
