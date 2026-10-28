"""对外服务门面：``ChatService.respond()`` 是整条链路的唯一入口。

返回值刻意做成「可直接序列化的字典」，包含回复、决策、置信度、证据、成本、
图事件——HTTP 层、CLI、评测脚本共用同一个结构，避免三处各自拼一份。
"""

from __future__ import annotations

from typing import Any, Optional

from ..types import Decision
from .context import TurnContext
from .deps import Deps, build_deps
from .graph import build_graph
from .store import SessionStore


class ChatService:
    def __init__(self, deps: Optional[Deps] = None, store: Optional[SessionStore] = None) -> None:
        self.deps = deps or build_deps()
        self.store = store or SessionStore()
        self.graph = build_graph()

    # ------------------------------------------------------------------
    def respond(self, session_id: str, user_id: str, text: str) -> dict[str, Any]:
        state = self.store.get_or_create(session_id, user_id)

        # 每轮重置「一次性」字段，保留 slots/turns/summary/pending_action
        state.user_input = text
        state.evidence = []
        state.tool_results = []
        state.claims = []
        state.numeric_violations = []
        state.clarify_fields = []
        state.clarify_question = ""
        state.cache_hit = False
        state.reply = ""
        state.citations = []
        state.escalation_reason = ""
        state.meta.pop("need_order", None)
        state.meta.pop("tool_error", None)
        state.meta.pop("order", None)

        ctx = TurnContext(deps=self.deps)
        run = self.graph.run(state, ctx, max_steps=16)
        state.meta["graph"] = run.to_dict()
        state.errors.extend(e["message"] for e in ctx.errors)
        self.store.put(state)

        return self.payload(state, ctx, run)

    # ------------------------------------------------------------------
    def payload(self, state, ctx: TurnContext, run) -> dict[str, Any]:
        policy = state.policy
        return {
            "session_id": state.session_id,
            "trace_id": state.trace_id,
            "reply": state.reply,
            "decision": state.decision.value,
            "intent": state.intent.value,
            "intent_confidence": round(state.intent_confidence, 4),
            "route": state.route.value,
            "confidence": state.confidence,
            "calibrated": state.calibrated,
            "threshold": state.threshold,
            "tier": state.tier,
            "cache_hit": state.cache_hit,
            "citations": state.citations,
            "clarify_fields": state.clarify_fields,
            "escalation_reason": state.escalation_reason,
            "policy": None if policy is None else {
                "clause_id": policy.clause_id,
                "title": policy.title,
                "action": policy.action,
                "status": policy.status.value,
                "satisfied": policy.satisfied,
                "violated": policy.violated,
                "missing": policy.missing,
                "exception_id": policy.exception_id,
            },
            "verification": state.meta.get("verification"),
            "safety": state.meta.get("safety"),
            "tier_reason": state.meta.get("tier_reason"),
            "tool_results": [
                {"name": tr.name, "ok": tr.ok, "error": tr.error,
                 "data": tr.data, "cached": tr.cached}
                for tr in state.tool_results
            ],
            "cost_usd": state.cost_usd,
            "tokens": state.tokens,
            "latency_ms": state.latency_ms,
            "nodes": [e.node for e in run.events],
            "stopped_reason": run.stopped_reason,
            "errors": state.errors[-5:],
        }

    # ------------------------------------------------------------------
    def reset(self, session_id: str) -> None:
        self.store.delete(session_id)

    def stats(self) -> dict[str, Any]:
        return {
            "deps": self.deps.summarize(),
            "sessions": len(self.store),
            "cache": self.deps.cache.stats(),
            "cost": self.deps.ledger.summary(),
            "parsers": self.deps.engine.stats(),
            "retrieval": self.deps.retrieval.stats(),
        }
