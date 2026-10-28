"""精排层。

两个实现：
- ``HeuristicReranker``：无训练参数的轻量精排，离线默认。核心信号是
  「查询词覆盖率 + 政策条款关键词命中 + 数字一致性」。
- ``CrossEncoderReranker``：BGE-reranker 交叉编码器，装 transformers + 模型
  即可切换到真实精排（对应简历里的 BGE-Reranker 精排）。

为什么默认不用交叉编码器：模型权重几百 MB，会让「clone 即跑」失效。
但接口留好，切一行配置就能上——这正是简历里那套链路该有的样子。

数字一致性这一项很重要：客服问句里「7 天」「15 天」「48 小时」是判定核心，
查询里的数字在条款里找不到时，说明召回错了，应该直接压分。
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

from ..utils.text import overlap_ratio

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


class HeuristicReranker:
    def __init__(self, w_coverage: float = 0.55, w_keyword: float = 0.25,
                 w_title: float = 0.20, number_penalty: float = 0.45) -> None:
        self.w_coverage = w_coverage
        self.w_keyword = w_keyword
        self.w_title = w_title
        self.number_penalty = number_penalty

    def score(
        self,
        query: str,
        text: str,
        title: str = "",
        keywords: Optional[Sequence[str]] = None,
    ) -> float:
        if not text:
            return 0.0
        coverage = overlap_ratio(query, text)
        kw_hit = 0.0
        if keywords:
            hits = sum(1 for k in keywords if k and k in query)
            kw_hit = min(1.0, hits / 2.0)
        title_hit = 0.0
        if title:
            title_hit = overlap_ratio(title, query)
        base = (
            self.w_coverage * coverage
            + self.w_keyword * kw_hit
            + self.w_title * title_hit
        )

        q_nums = set(_NUM_RE.findall(query))
        if q_nums:
            hit = sum(1 for n in q_nums if n in text)
            if hit == 0:
                base -= self.number_penalty
            else:
                base += 0.10 * (hit / len(q_nums))
        return max(0.0, min(1.0, base))


class CrossEncoderReranker:
    """BGE reranker 交叉编码器。首次调用才加载模型，避免影响启动速度。"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", max_length: int = 512) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self._model = None
        self._tokenizer = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # type: ignore
            from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
        except ImportError as e:  # pragma: no cover - 可选依赖
            raise RuntimeError(
                "CrossEncoderReranker 需要 torch + transformers：pip install torch transformers"
            ) from e
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.eval()
        self._torch = torch

    def score(self, query: str, text: str, title: str = "", keywords=None) -> float:
        self._ensure()
        pair = f"{title}\n{text}" if title else text
        inputs = self._tokenizer(
            [[query, pair]], padding=True, truncation=True, max_length=self.max_length,
            return_tensors="pt",
        )
        with self._torch.no_grad():
            logits = self._model(**inputs).logits.view(-1)
        return float(self._torch.sigmoid(logits[0]).item())


def build_reranker(settings):
    kind = getattr(settings, "rerank_provider", "heuristic")
    if kind in {"bge", "cross-encoder"}:
        return CrossEncoderReranker(model_name=getattr(settings, "rerank_model", "BAAI/bge-reranker-base"))
    return HeuristicReranker()
