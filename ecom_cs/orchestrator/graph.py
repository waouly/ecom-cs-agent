"""把节点接成图。

图是「文档」也是「可执行物」：``mermaid()`` 输出的就是 README 里的架构图，
而实际执行走的是同一份定义——不会出现「图画得漂亮，代码是另一回事」。
"""

from __future__ import annotations

from .nodes import (
    decision_router,
    n_cache,
    n_clarify,
    n_confirm,
    n_escalate,
    n_evaluate,
    n_finalize,
    n_generate,
    n_guard,
    n_retrieve,
    n_smalltalk,
    n_tools,
    n_understand,
    n_verify,
    route_router,
)
from ..graph.engine import END, StateGraph

NODES = {
    "guard": n_guard,
    "confirm": n_confirm,
    "understand": n_understand,
    "cache": n_cache,
    "dispatch": None,          # 纯路由节点，返回 __router__ 后由图引擎查 router
    "tools": n_tools,
    "retrieve": n_retrieve,
    "evaluate": n_evaluate,
    "smalltalk": n_smalltalk,
    "clarify": n_clarify,
    "escalate": n_escalate,
    "generate": n_generate,
    "verify": n_verify,
    "finalize": n_finalize,
}

EDGES = [
    ("guard", "confirm"),
    ("confirm", "understand"),
    ("understand", "cache"),
    ("cache", "dispatch"),
    ("tools", "retrieve"),
    ("retrieve", "evaluate"),
    ("smalltalk", "finalize"),
    ("clarify", "finalize"),
    ("escalate", "finalize"),
    ("generate", "verify"),
    ("verify", "finalize"),
    ("finalize", END),
]

ROUTER_TARGETS = {
    "dispatch": ("escalate", "smalltalk", "tools", "retrieve"),
    "evaluate": ("clarify", "escalate", "generate"),
}


def build_graph() -> StateGraph:
    graph = StateGraph("ecom_cs")
    for name, fn in NODES.items():
        graph.add_node(name, fn or (lambda state, ctx: "__router__"))
    for src, dst in EDGES:
        graph.add_edge(src, dst)
    graph.add_conditional_edge("dispatch", route_router)
    graph.add_conditional_edge("evaluate", decision_router)
    graph.set_entry("guard")
    return graph


route_router.__graph_targets__ = set(ROUTER_TARGETS["dispatch"])   # type: ignore[attr-defined]
decision_router.__graph_targets__ = set(ROUTER_TARGETS["evaluate"])  # type: ignore[attr-defined]
