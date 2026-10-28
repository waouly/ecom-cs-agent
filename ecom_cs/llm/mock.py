"""离线 Mock LLM。

只用于「原始 chat 接口」的兜底演示与测试：它不会真的推理，只会返回确定性
的结构化回显，保证在没有 API Key 的环境里整条链路依然可跑、可断言。
真正的离线生成能力在 ``ecom_cs.llm.tasks.HeuristicAnswerer`` 里。
"""

from __future__ import annotations

import hashlib
from typing import Iterator

from .base import count_tokens


class MockLLM:
    name = "mock-llm-1"

    def __init__(self, model: str = "mock-llm-1", seed: str = "") -> None:
        self.model = model
        self.seed = seed

    # -- 内部 --
    def _fingerprint(self, text: str) -> str:
        h = hashlib.blake2b((self.seed + text).encode("utf-8"), digest_size=6).hexdigest()
        return h

    def _render(self, messages: list[dict]) -> str:
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = str(m.get("content", ""))
                break
        fp = self._fingerprint(last_user)
        head = last_user.strip().replace("\n", " ")[:60]
        return f"[MOCK:{fp}] 针对「{head}」的离线模拟回复。配置 ECOM_CS_LLM_PROVIDER=openai 可切换真实模型。"

    # -- 接口 --
    def complete(self, prompt: str, **kwargs) -> str:
        return self._render([{"role": "user", "content": prompt}])

    def chat(self, messages: list[dict], **kwargs) -> str:
        return self._render(messages)

    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        text = self.complete(prompt)
        step = max(1, len(text) // 8)
        for i in range(0, len(text), step):
            yield text[i : i + step]

    def usage_hint(self, prompt: str, completion: str) -> int:
        return count_tokens(prompt) + count_tokens(completion)
