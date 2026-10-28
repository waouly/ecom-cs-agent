"""工具层骨架：统一的调用协议、权限上下文、注册表。

为什么要一个独立工具层而不是在编排里直接写 SQL：
1. **权限边界**。用户能看到哪张订单、能不能改，必须在数据出口处判定，
   而不是在 prompt 里「请勿泄露他人订单」——那是靠模型自觉，等于没有。
2. **副作用可控**。写操作要幂等、要确认、要审计、要可补偿，这些逻辑
   放在工具里有唯一入口，编排层只负责「什么时候调」。
3. **可评测**。工具是纯函数式的（除写操作），可以在离线评测里直接打桩。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional, Protocol, runtime_checkable

from ..db.repo import Repository
from ..types import SessionState, ToolResult


class ToolError(Exception):
    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code
        self.message = message or code


@dataclass
class ToolSpec:
    name: str
    kind: str                      # read | write
    description: str
    args: dict[str, str] = field(default_factory=dict)
    required: list[str] = field(default_factory=list)
    mutates: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
            "args": self.args,
            "required": self.required,
            "mutates": self.mutates,
        }


@dataclass
class ToolContext:
    session: SessionState
    repo: Repository
    today: date = field(default_factory=date.today)
    settings: Any = None

    @property
    def user_id(self) -> str:
        return self.session.user_id

    @property
    def session_id(self) -> str:
        return self.session.session_id


@runtime_checkable
class Tool(Protocol):
    spec: ToolSpec

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult: ...


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.spec.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def specs(self, kind: Optional[str] = None) -> list[dict[str, Any]]:
        return [
            t.spec.to_dict() for t in self._tools.values()
            if kind is None or t.spec.kind == kind
        ]

    def names(self) -> list[str]:
        return list(self._tools)

    def execute(self, name: str, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        started = time.perf_counter()
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(name=name, ok=False, error=f"unknown_tool:{name}")
        try:
            result = tool.run(args, ctx)
        except ToolError as e:
            result = ToolResult(name=name, ok=False, error=e.code, data={"message": e.message})
        except Exception as e:  # 兜底：工具异常不能把整条会话打断
            result = ToolResult(name=name, ok=False, error="tool_exception",
                                data={"message": f"{type(e).__name__}: {e}"})
        result.latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return result


def build_registry() -> ToolRegistry:
    from .read_tools import AfterSalesLookupTool, LogisticsLookupTool, OrderLookupTool
    from .write_tools import ChangeAddressTool, CreateReturnTool, HandoffTool

    registry = ToolRegistry()
    for tool in (
        OrderLookupTool(),
        LogisticsLookupTool(),
        AfterSalesLookupTool(),
        ChangeAddressTool(),
        CreateReturnTool(),
        HandoffTool(),
    ):
        registry.register(tool)
    return registry
