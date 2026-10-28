"""单轮上下文：把依赖、错误、耗时收在一处，供节点函数使用。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .deps import Deps


@dataclass
class TurnContext:
    deps: Deps
    errors: list[dict[str, str]] = field(default_factory=list)
    started_at: float = field(default_factory=time.perf_counter)

    def record_error(self, where: str, message: str, tb: str = "") -> None:
        self.errors.append({"where": where, "message": message, "traceback": tb[-800:]})

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started_at) * 1000

    @property
    def settings(self):
        return self.deps.settings

    @property
    def repo(self):
        return self.deps.repo

    @property
    def today(self):
        return self.deps.today

    def to_dict(self) -> dict[str, Any]:
        return {"errors": self.errors, "elapsed_ms": round(self.elapsed_ms(), 2)}
