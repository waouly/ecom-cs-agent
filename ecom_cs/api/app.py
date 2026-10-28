"""FastAPI 应用。

启动时构建一次依赖（政策引擎、向量索引、DB、图），请求间复用：
- 索引重建是重操作，放在 lifespan 而不是每次请求里；
- 会话状态在进程内存（SessionStore），生产环境应换成 Redis，
  这里保持「同一进程内可跑」的最小实现，并在 stats 里暴露会话数。

为什么返回这么多字段：客服系统的每一次回答都要能被复盘——
decision 解释「为什么答/为什么转人工」，threshold/calibrated 解释置信度口径，
tier/cost_usd 解释成本，nodes 解释走了哪条链路。接口层不做裁剪。
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import get_settings
from ..orchestrator import ChatService, build_deps

STATIC_DIR = Path(__file__).resolve().parent / "static"


class ChatRequest(BaseModel):
    session_id: str = Field(default="s1", description="会话 id，多轮上下文按它隔离")
    user_id: str = Field(default="u_demo", description="当前登录用户，用于越权校验")
    message: str = Field(min_length=1, max_length=1000)


class FeedbackRequest(BaseModel):
    session_id: str
    rating: int = Field(ge=-1, le=1, description="1 赞 / -1 踩 / 0 中立")
    comment: str = ""


def _public(payload: dict) -> dict:
    """去掉体积大、只对内部复盘有用的字段。"""
    out = dict(payload)
    out.pop("nodes", None) if len(out.get("nodes") or []) > 20 else None
    return out


def _order_card(order: dict) -> dict:
    """订单 → 前端商品卡片。

    只保留「用户看得懂、界面用得上」的字段；类目/状态的中文标签由 repo 层补齐，
    免得前端再维护一份映射表（映射漂移是客服系统最常见的脏数据来源）。
    """
    return {
        "order_no": order.get("order_no"),
        "title": order.get("title") or "",
        "category": order.get("category") or "",
        "category_label": order.get("category_label") or order.get("category") or "",
        "amount": order.get("amount"),
        "status": order.get("status") or "",
        "status_label": order.get("status_label") or order.get("status") or "",
        "paid_at": order.get("paid_at") or "",
        "signed_at": order.get("signed_at") or "",
        "address": order.get("address") or "",
        "opened": bool(order.get("opened")),
        "is_custom": bool(order.get("is_custom")),
        "can_change_address": order.get("status") == "paid",
    }


def _attach_cards(out: dict, repo) -> dict:
    """把工具结果里的订单/物流摊成前端可直接渲染的卡片，不改动 Agent 原输出。"""
    for t in out.get("tool_results") or []:
        if not t.get("ok"):
            continue
        data = t.get("data") or {}
        if t.get("name") == "lookup_order":
            order = repo.get_order(str(data.get("order_no") or ""))
            if order:
                out["product"] = _order_card(order)
        elif t.get("name") == "lookup_logistics" and data.get("has_logistics"):
            out["logistics"] = {
                "order_no": data.get("order_no"),
                "carrier": data.get("carrier") or "",
                "tracking_no": data.get("tracking_no") or "",
                "status_label": data.get("status_label") or "",
                "description": data.get("description") or "",
                "last_update": data.get("last_update") or "",
            }
    return out


def create_app(service: Optional[ChatService] = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.svc = service or ChatService(build_deps(get_settings()))
        yield

    app = FastAPI(
        title="电商智能客服 Agent",
        version="1.0.0",
        description="政策规则引擎 + RAG + 生成后核验的客服问答/处置服务",
        lifespan=lifespan,
    )

    def svc() -> ChatService:
        return app.state.svc

    # 演示台是单文件静态页，不引入构建链；直接挂在 / 上，serve 起来打开就是 UI。
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": "ecom-cs-agent"}

    @app.post("/chat")
    def chat(req: ChatRequest) -> dict:
        out = svc().respond(req.session_id, req.user_id, req.message)
        return _attach_cards(_public(out), svc().deps.repo)

    @app.get("/orders")
    def orders(user_id: str = "u_demo", limit: int = 8) -> dict:
        """当前用户的历史订单（右侧栏）。

        user_id 由前端显式传入是因为演示台允许切用户看越权拦截；
        真实系统里它必须来自登录态，绝不能信前端。
        """
        rows = svc().deps.repo.list_orders(user_id, limit=max(1, min(limit, 50)))
        return {"count": len(rows), "orders": [_order_card(r) for r in rows]}

    @app.get("/orders/{order_no}")
    def order_detail(order_no: str, user_id: str = "u_demo") -> dict:
        deps = svc().deps
        order = deps.repo.get_order(order_no)
        # 不存在与不属于当前用户返回同一个 404：避免拿订单号探测他人订单是否存在
        if order is None or order.get("user_id") != user_id:
            raise HTTPException(404, "order not found")
        log = deps.repo.get_logistics(order_no)
        aftersales = [
            {
                "aftersale_no": r.get("aftersale_no"),
                "type_label": r.get("type_label", r.get("type")),
                "status_label": r.get("status_label", r.get("status")),
                "created_at": r.get("created_at"),
                "eta": r.get("eta", ""),
            }
            for r in deps.repo.list_aftersales(user_id)
            if r.get("order_no") == order_no
        ]
        return {
            "order": _order_card(order),
            "logistics": (
                {
                    "carrier": log.get("carrier") or "",
                    "tracking_no": log.get("tracking_no") or "",
                    "status": log.get("status") or "",
                    "status_label": log.get("status_label") or "",
                    "description": log.get("description") or "",
                    "updated_at": log.get("updated_at") or "",
                }
                if log
                else None
            ),
            "aftersales": aftersales,
        }

    @app.post("/chat/stream")
    def chat_stream(req: ChatRequest) -> StreamingResponse:
        out = _attach_cards(svc().respond(req.session_id, req.user_id, req.message), svc().deps.repo)

        def gen():
            # 先发一个 meta 事件，前端可以立刻渲染决策标签/转人工提示
            meta = {
                "decision": out["decision"],
                "intent": out["intent"],
                "escalation_reason": out.get("escalation_reason", ""),
                "trace_id": out.get("trace_id"),
            }
            yield f"event: meta\ndata: {json.dumps(meta, ensure_ascii=False)}\n\n"
            for i in range(0, len(out["reply"]), 12):
                chunk = out["reply"][i : i + 12]
                yield f"event: delta\ndata: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
            tail = {
                "decision": out["decision"],
                "citations": out.get("citations", []),
                "confidence": out.get("confidence"),
                "calibrated": out.get("calibrated"),
                "cost_usd": out.get("cost_usd"),
                "tier": out.get("tier"),
                "product": out.get("product"),
                "logistics": out.get("logistics"),
            }
            yield f"event: done\ndata: {json.dumps(tail, ensure_ascii=False)}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/sessions/{session_id}")
    def session(session_id: str) -> dict:
        state = svc().store.get(session_id)
        if state is None:
            raise HTTPException(404, "session not found")
        return state.to_dict()

    @app.post("/feedback")
    def feedback(req: FeedbackRequest) -> dict:
        deps = svc().deps
        state = svc().store.get(req.session_id)
        deps.repo.add_feedback(
            session_id=req.session_id,
            query=(state.user_input if state else ""),
            reply=(state.reply if state else ""),
            rating=req.rating,
            comment=req.comment,
        )
        return {"ok": True}

    @app.get("/policies")
    def policies() -> dict:
        engine = svc().deps.engine
        return {
            "count": len(engine.clauses),
            "clauses": [
                {"clause_id": c.clause_id, "title": c.title, "action": c.action,
                 "policy_type": c.policy_type, "applies_to": c.applies_to}
                for c in engine.clauses
            ],
        }

    @app.get("/policies/{clause_id}")
    def policy(clause_id: str) -> dict:
        clause = svc().deps.engine.by_id(clause_id)
        if clause is None:
            raise HTTPException(404, "clause not found")
        return clause.to_dict()

    @app.get("/stats")
    def stats() -> dict:
        deps = svc().deps
        return {
            "deps": deps.summarize(),
            "cost": deps.ledger.summary(),
            "cache": deps.cache.stats(),
            "sessions": len(svc().store),
            "gaps": deps.repo.stats().get("gaps", 0),
        }

    return app


app = create_app()
