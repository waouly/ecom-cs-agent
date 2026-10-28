"""写工具：改地址、发起退货、转人工。

这是简历里「只会答不会办」的缺口。补上之后立刻出现四个真问题，这里逐个处理：

1. **重复执行**。用户说「改一下」两次，不能改两次。用 (session, 参数) 派生的
   幂等键，命中已执行记录就直接返回缓存结果，附带 ``cached=True``。
2. **误执行**。写操作默认不落库，先返回 ``needs_confirm`` 预览，用户确认后
   再带 ``confirm=True`` 调一次。危险操作不能由模型「觉得用户想改」就执行。
3. **越权**。改地址前必须校验订单归属，越权返回「订单不存在」。
4. **失败补偿**。创建售后单 + 改订单状态是两步写，第二步失败要把第一步回滚，
   否则会出现「有售后单但订单还是已签收」的脏数据。
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from ..types import ToolResult
from .base import ToolContext, ToolError, ToolSpec
from .read_tools import _require_order


def _idem_key(*parts: Any) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _seen(ctx: ToolContext, action: str, key: str) -> dict[str, Any] | None:
    """在审计日志里找同幂等键的历史执行记录。"""
    row = ctx.repo.conn.execute(
        "SELECT payload, result FROM audit_log WHERE action = ? AND payload LIKE ?"
        " ORDER BY id DESC LIMIT 1",
        (action, f"%{key}%"),
    ).fetchone()
    if not row:
        return None
    try:
        payload = json.loads(row["payload"])
    except (json.JSONDecodeError, TypeError):
        return None
    payload["_previous_result"] = row["result"]
    return payload


class ChangeAddressTool:
    spec = ToolSpec(
        name="change_address",
        kind="write",
        description="修改未发货订单的收货地址（需用户二次确认）",
        args={"order_no": "str", "new_address": "str", "confirm": "bool"},
        required=["order_no", "new_address"],
        mutates=True,
    )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        order = _require_order(ctx, str(args.get("order_no", "")))
        new_address = str(args.get("new_address", "")).strip()
        if not new_address:
            raise ToolError("missing_address", "缺少新地址")

        if order.get("shipped_at"):
            raise ToolError("already_shipped", "订单已发货，无法修改收货地址")
        if new_address == (order.get("address") or "").strip():
            return ToolResult(name=self.spec.name, ok=True,
                              data={"changed": False, "message": "地址与当前一致，无需修改"})

        key = _idem_key(ctx.session_id, order["order_no"], new_address)
        confirm = bool(args.get("confirm"))

        if not confirm:
            return ToolResult(
                name=self.spec.name, ok=True,
                data={
                    "needs_confirm": True,
                    "idempotency_key": key,
                    "preview": {
                        "order_no": order["order_no"],
                        "old_address": order.get("address"),
                        "new_address": new_address,
                    },
                    "message": f"将把订单 {order['order_no']} 的收货地址改为：{new_address}，确认操作请回复「确认」",
                },
                idempotency_key=key,
            )

        previous = _seen(ctx, "change_address", key)
        if previous is not None:
            return ToolResult(
                name=self.spec.name, ok=True, cached=True, idempotency_key=key,
                data={"changed": True, "reused": True, "order_no": order["order_no"],
                      "new_address": new_address, "message": "该修改已执行过，未重复提交"},
            )

        old_address = order.get("address")
        ctx.repo.update_order_address(order["order_no"], new_address)
        ctx.repo.add_audit(
            actor=ctx.user_id, action="change_address", target=order["order_no"],
            payload={"idempotency_key": key, "old": old_address, "new": new_address},
            result="ok",
        )
        ctx.repo.conn.commit()
        return ToolResult(
            name=self.spec.name, ok=True, idempotency_key=key,
            data={"changed": True, "order_no": order["order_no"],
                  "old_address": old_address, "new_address": new_address,
                  "message": "地址已修改成功"},
        )


class CreateReturnTool:
    spec = ToolSpec(
        name="create_return",
        kind="write",
        description="为订单发起退货/退款申请（需用户二次确认）",
        args={"order_no": "str", "reason": "str", "aftersale_type": "str", "confirm": "bool"},
        required=["order_no", "reason"],
        mutates=True,
    )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        order = _require_order(ctx, str(args.get("order_no", "")))
        reason = str(args.get("reason", "")).strip() or "用户申请"
        type_ = str(args.get("aftersale_type", "return_refund"))
        amount = float(order.get("amount") or 0.0)
        key = _idem_key(ctx.session_id, order["order_no"], type_, reason)

        existing = ctx.repo.find_aftersale_by_idem(key)
        if existing:
            return ToolResult(
                name=self.spec.name, ok=True, cached=True, idempotency_key=key,
                data={"aftersale_no": existing["aftersale_no"], "status": existing.get("status"),
                      "reused": True, "message": "该退货申请已提交过，无需重复提交"},
            )

        if not bool(args.get("confirm")):
            return ToolResult(
                name=self.spec.name, ok=True, idempotency_key=key,
                data={
                    "needs_confirm": True,
                    "preview": {"order_no": order["order_no"], "amount": amount,
                                "reason": reason, "aftersale_type": type_},
                    "message": (f"将为订单 {order['order_no']} 发起退货申请，"
                                f"预计退款 {amount:.2f} 元，审核 1-3 个工作日。确认请回复「确认」"),
                },
            )

        aftersale_no = f"AS{int(time.time() * 1000) % 10**10}"
        old_status = order.get("status")
        created: dict[str, Any] | None = None
        try:
            created = ctx.repo.create_aftersale(
                aftersale_no=aftersale_no,
                order_no=order["order_no"],
                user_id=ctx.user_id,
                type_=type_,
                amount=amount,
                idempotency_key=key,
                note=reason,
            )
            ctx.repo.set_order_status(order["order_no"], "refunding")
            ctx.repo.add_audit(
                actor=ctx.user_id, action="create_return", target=order["order_no"],
                payload={"idempotency_key": key, "aftersale_no": aftersale_no,
                         "amount": amount, "reason": reason},
                result="ok",
            )
            ctx.repo.conn.commit()
        except Exception as e:
            # 补偿：回滚已写入的售后单与订单状态
            ctx.repo.conn.rollback()
            if created:
                ctx.repo.conn.execute(
                    "DELETE FROM aftersales WHERE aftersale_no = ?", (aftersale_no,)
                )
            ctx.repo.set_order_status(order["order_no"], old_status or "signed")
            ctx.repo.add_audit(
                actor=ctx.user_id, action="create_return", target=order["order_no"],
                payload={"idempotency_key": key, "error": str(e)}, result="failed",
            )
            ctx.repo.conn.commit()
            raise ToolError("create_return_failed", "退货申请提交失败，请稍后重试或转人工") from e

        return ToolResult(
            name=self.spec.name, ok=True, idempotency_key=key,
            data={"aftersale_no": aftersale_no, "amount": amount, "status": "pending_review",
                  "status_label": "待审核", "eta": "1-3 个工作日",
                  "message": f"退货申请已提交，售后单号 {aftersale_no}，预计 1-3 个工作日审核"},
        )


class HandoffTool:
    spec = ToolSpec(
        name="handoff_to_human",
        kind="write",
        description="转交人工客服，创建工单并记录原因",
        args={"reason": "str", "order_no": "str"},
        required=["reason"],
        mutates=True,
    )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        reason = str(args.get("reason", "")).strip() or "未说明原因"
        order_no = str(args.get("order_no", "")).strip()
        ticket = f"CS{int(time.time()) % 10**8}"
        ctx.repo.add_audit(
            actor=ctx.user_id, action="handoff", target=order_no or ctx.session_id,
            payload={"ticket": ticket, "reason": reason, "session_id": ctx.session_id},
            result="ok",
        )
        ctx.repo.conn.commit()
        return ToolResult(
            name=self.spec.name, ok=True,
            data={"ticket": ticket, "reason": reason,
                  "message": f"已转人工客服，工单号 {ticket}，预计 10 分钟内响应"},
        )
