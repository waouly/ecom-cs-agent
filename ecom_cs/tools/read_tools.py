"""读工具：订单、物流、售后。

权限判定放在这里，而不是靠 prompt 约束。越权访问返回的是「不存在」而不是
「无权查看」——后者会泄露「这个订单号是有效的」这一信息，属于典型的越权侧信道。
"""

from __future__ import annotations

from typing import Any

from ..db.labels import AFTERSALE_STATUS_LABEL, LOGISTICS_STATUS_LABEL, ORDER_STATUS_LABEL
from ..types import ToolResult
from .base import ToolContext, ToolError, ToolSpec


def _require_order(ctx: ToolContext, order_no: str) -> dict[str, Any]:
    if not order_no:
        raise ToolError("missing_order_no", "缺少订单号")
    order = ctx.repo.get_order(order_no)
    if not order:
        raise ToolError("order_not_found", f"订单 {order_no} 不存在")
    if order.get("user_id") != ctx.user_id:
        # 与「不存在」表现一致，避免越权探测
        raise ToolError("order_not_found", f"订单 {order_no} 不存在")
    return order


class OrderLookupTool:
    spec = ToolSpec(
        name="lookup_order",
        kind="read",
        description="按订单号查询订单详情（含状态、类目、金额、签收时间）",
        args={"order_no": "str"},
        required=["order_no"],
    )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        order = _require_order(ctx, str(args.get("order_no", "")))
        data = {
            "order_no": order["order_no"],
            "status": order.get("status"),
            "status_label": ORDER_STATUS_LABEL.get(order.get("status", ""), order.get("status")),
            "category": order.get("category"),
            "amount": order.get("amount"),
            "paid_at": order.get("paid_at"),
            "shipped_at": order.get("shipped_at"),
            "signed_at": order.get("signed_at"),
            "opened": bool(order.get("opened")),
            "is_custom": bool(order.get("is_custom")),
            "address": order.get("address"),
        }
        return ToolResult(name=self.spec.name, ok=True, data=data, tier="L0")


class LogisticsLookupTool:
    spec = ToolSpec(
        name="lookup_logistics",
        kind="read",
        description="查询订单物流轨迹与最新状态",
        args={"order_no": "str"},
        required=["order_no"],
    )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        order = _require_order(ctx, str(args.get("order_no", "")))
        log = ctx.repo.get_logistics(order["order_no"])
        if not log:
            return ToolResult(name=self.spec.name, ok=True,
                              data={"order_no": order["order_no"], "has_logistics": False})
        return ToolResult(
            name=self.spec.name,
            ok=True,
            data={
                "order_no": order["order_no"],
                "has_logistics": True,
                "carrier": log.get("carrier"),
                "tracking_no": log.get("tracking_no"),
                "status": log.get("status"),
                "status_label": LOGISTICS_STATUS_LABEL.get(log.get("status", ""), log.get("status")),
                "last_update": log.get("updated_at"),
                "description": log.get("description", ""),
            },
            tier="L0",
        )


class AfterSalesLookupTool:
    spec = ToolSpec(
        name="lookup_aftersale",
        kind="read",
        description="查询订单关联的售后单（退款/退货/换货进度）",
        args={"order_no": "str"},
        required=["order_no"],
    )

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        order = _require_order(ctx, str(args.get("order_no", "")))
        rows = [r for r in ctx.repo.list_aftersales(ctx.user_id)
                if r.get("order_no") == order["order_no"]]
        items = [
            {
                "aftersale_no": r.get("aftersale_no"),
                "type": r.get("type"),
                "type_label": r.get("type_label", r.get("type")),
                "status": r.get("status"),
                "status_label": AFTERSALE_STATUS_LABEL.get(r.get("status", ""), r.get("status")),
                "amount": r.get("amount"),
                "created_at": r.get("created_at"),
                "eta": r.get("eta", ""),
                "note": r.get("note", ""),
            }
            for r in rows
        ]
        return ToolResult(
            name=self.spec.name, ok=True,
            data={"order_no": order["order_no"], "count": len(items), "items": items},
            tier="L0",
        )
