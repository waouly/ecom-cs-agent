"""LLM 客户端抽象与用量统计。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol, runtime_checkable


class LLMError(RuntimeError):
    pass


def count_tokens(text: str) -> int:
    """粗略 token 估算：CJK 字符按 1 token，其它按 1/3.5 token。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    rest = len(text) - cjk
    return max(1, cjk + int(rest / 3.5) + 1)


@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    cost_usd: float = 0.0

    def add(self, prompt: str, completion: str, price_in: float, price_out: float) -> None:
        pt = count_tokens(prompt)
        ct = count_tokens(completion)
        self.prompt_tokens += pt
        self.completion_tokens += ct
        self.calls += 1
        self.cost_usd += pt / 1000 * price_in + ct / 1000 * price_out

    def merge(self, other: "LLMUsage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.calls += other.calls
        self.cost_usd += other.cost_usd

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@runtime_checkable
class LLMClient(Protocol):
    name: str

    def complete(self, prompt: str, **kwargs) -> str: ...

    def chat(self, messages: list[dict], **kwargs) -> str: ...

    def stream(self, prompt: str, **kwargs) -> Iterator[str]: ...
