"""OpenAI 兼容协议客户端（httpx 实现，不依赖 openai SDK）。

对接 DeepSeek / Qwen / Moonshot / vLLM / Ollama 等任何 OpenAI 兼容端点。
"""

from __future__ import annotations

import json
from typing import Any, Iterator

import httpx

from .base import LLMError


class OpenAICompatLLM:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.0,
        timeout_s: float = 30.0,
    ) -> None:
        if not base_url:
            raise LLMError("ECOM_CS_LLM_BASE_URL 未配置")
        if not model:
            raise LLMError("ECOM_CS_LLM_MODEL 未配置")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.name = f"openai-compat:{model}"

    # -- 内部 --
    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _payload(self, messages: list[dict], stream: bool) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": stream,
        }

    # -- 接口 --
    def chat(self, messages: list[dict], **kwargs) -> str:
        url = f"{self.base_url}/chat/completions"
        try:
            resp = httpx.post(
                url,
                headers=self._headers(),
                json=self._payload(messages, stream=False),
                timeout=kwargs.get("timeout", self.timeout_s),
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"] or ""
        except httpx.HTTPStatusError as e:
            raise LLMError(f"LLM HTTP {e.response.status_code}: {e.response.text[:300]}") from e
        except Exception as e:  # noqa: BLE001
            raise LLMError(f"LLM 调用失败: {e}") from e

    def complete(self, prompt: str, **kwargs) -> str:
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        url = f"{self.base_url}/chat/completions"
        messages = [{"role": "user", "content": prompt}]
        try:
            with httpx.stream(
                "POST",
                url,
                headers=self._headers(),
                json=self._payload(messages, stream=True),
                timeout=kwargs.get("timeout", self.timeout_s),
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        obj = json.loads(chunk)
                        delta = obj["choices"][0].get("delta", {}).get("content")
                        if delta:
                            yield delta
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
        except Exception as e:  # noqa: BLE001
            raise LLMError(f"LLM 流式调用失败: {e}") from e
