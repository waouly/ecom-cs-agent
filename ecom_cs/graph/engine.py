"""自研图引擎：约 100 行，替代 LangGraph。

为什么不用 LangGraph：
- 依赖重（连 torch 生态一起拖），会让「clone 即跑」失效；
- 真正的价值在**编排语义**——节点、条件边、步数上限、可观测事件流，
  这些用几十行就能表达清楚，理解了反而更容易讲明白。

保留的语义：
- 节点函数 ``(state, ctx) -> next_node_name``，直接改 state；
- 条件边用 router 决定去向，router 只读 state；
- 全局步数上限，防止环形图死循环（真实客服系统最常见的线上事故）；
- 每个节点产出 trace 事件，串起来就是一次可回放的会话；
- 节点异常被捕获并记录，不炸整个会话——客服系统「挂了」比「答错」更糟。
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

END = "__end__"

NodeFn = Callable[[Any, Any], str]
RouterFn = Callable[[Any], str]


@dataclass
class GraphEvent:
    node: str
    next: str
    duration_ms: float
    ok: bool = True
    error: str = ""
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "next": self.next,
            "duration_ms": round(self.duration_ms, 2),
            "ok": self.ok,
            "error": self.error,
            "summary": self.summary,
        }


@dataclass
class GraphRun:
    final_node: str
    steps: int
    stopped_reason: str           # end | max_steps | error
    events: list[GraphEvent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_node": self.final_node,
            "steps": self.steps,
            "stopped_reason": self.stopped_reason,
            "events": [e.to_dict() for e in self.events],
        }


class StateGraph:
    def __init__(self, name: str = "graph") -> None:
        self.name = name
        self._nodes: dict[str, NodeFn] = {}
        self._edges: dict[str, str] = {}
        self._routers: dict[str, RouterFn] = {}
        self._entry: Optional[str] = None

    # -- 构建 --
    def add_node(self, name: str, fn: NodeFn) -> "StateGraph":
        if name in self._nodes:
            raise ValueError(f"节点重复注册: {name}")
        self._nodes[name] = fn
        return self

    def add_edge(self, src: str, dst: str) -> "StateGraph":
        self._edges[src] = dst
        return self

    def add_conditional_edge(self, src: str, router: RouterFn) -> "StateGraph":
        self._routers[src] = router
        return self

    def set_entry(self, name: str) -> "StateGraph":
        self._entry = name
        return self

    # -- 校验 --
    def validate(self) -> list[str]:
        problems: list[str] = []
        if self._entry is None:
            problems.append("未设置 entry")
        for name in self._nodes:
            has_edge = name in self._edges
            has_router = name in self._routers
            if not has_edge and not has_router:
                problems.append(f"节点 {name} 没有出边")
        targets = set(self._edges.values())
        for router in self._routers.values():
            targets |= set(getattr(router, "__graph_targets__", set()))
        for t in targets:
            if t != END and t not in self._nodes:
                problems.append(f"边指向不存在的节点: {t}")
        if self._entry and self._entry not in self._nodes:
            problems.append(f"entry 不存在: {self._entry}")
        return problems

    # -- 执行 --
    def run(self, state: Any, ctx: Any = None, max_steps: int = 16,
            on_event: Optional[Callable[[GraphEvent], None]] = None) -> GraphRun:
        if self._entry is None:
            raise RuntimeError("图未设置 entry")
        current = self._entry
        events: list[GraphEvent] = []
        steps = 0
        stopped = "max_steps"

        while steps < max_steps:
            if current == END:
                stopped = "end"
                break
            fn = self._nodes.get(current)
            if fn is None:
                events.append(GraphEvent(node=current, next=END, duration_ms=0.0,
                                         ok=False, error="unknown_node"))
                stopped = "error"
                break
            started = time.perf_counter()
            steps += 1
            try:
                nxt = fn(state, ctx) or END
                ok, err = True, ""
            except Exception as e:
                nxt, ok = END, False
                err = f"{type(e).__name__}: {e}"
                if ctx is not None and hasattr(ctx, "record_error"):
                    ctx.record_error(f"node:{current}", err, traceback.format_exc())

            # 条件边优先于静态边
            if ok and nxt not in (END, None) and nxt == "__router__":
                router = self._routers.get(current)
                nxt = router(state) if router else END
            elif ok and current in self._routers and nxt == END:
                nxt = self._routers[current](state)

            event = GraphEvent(
                node=current, next=nxt,
                duration_ms=(time.perf_counter() - started) * 1000,
                ok=ok, error=err,
                summary=self._summary(state, current),
            )
            events.append(event)
            if on_event is not None:
                on_event(event)
            current = nxt if nxt else END
        else:
            stopped = "max_steps"

        return GraphRun(final_node=current, steps=steps, stopped_reason=stopped, events=events)

    @staticmethod
    def _summary(state: Any, node: str) -> dict[str, Any]:
        """只取少量关键字段，避免 trace 被大对象撑爆。"""
        getter = getattr(state, "trace_summary", None)
        if callable(getter):
            try:
                return getter(node)
            except Exception:
                return {}
        return {}

    # -- 可视化 --
    def mermaid(self) -> str:
        lines = ["graph TD"]
        for src, dst in self._edges.items():
            lines.append(f"    {src} --> {dst if dst != END else 'END'}")
        for src in self._routers:
            lines.append(f"    {src} -.->|条件| ROUTER_{src}")
        return "\n".join(lines)
