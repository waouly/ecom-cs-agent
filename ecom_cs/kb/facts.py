"""事实装配：把订单、物流、用户和对话槽位合并成规则引擎能消费的 facts。

为什么要单独一层：规则引擎只认扁平的 {字段: 值}，而真实系统里这些值散落在
数据库、工具返回值、用户口述里。把它们统一成一份 facts，才能让「订单事实」
和「用户补充说明」在同一个逻辑平面上被求值（比如用户在对话里说「我已经拆了」，
这个信息要能覆盖订单里的 opened 字段）。
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

from ..types import Slot

DATE_FMTS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d")


def parse_date(value: Any) -> Optional[date]:
    if not value:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    for fmt in DATE_FMTS:
        try:
            return datetime.strptime(str(value), fmt).date()
        except ValueError:
            continue
    return None


def _slot_value(slots: dict[str, Any] | None, name: str) -> Any:
    if not slots:
        return None
    s = slots.get(name)
    if s is None:
        return None
    return s.value if isinstance(s, Slot) else s


def build_facts(
    order: Optional[dict[str, Any]] = None,
    user: Optional[dict[str, Any]] = None,
    slots: Optional[dict[str, Any]] = None,
    today: Optional[date] = None,
) -> dict[str, Any]:
    today = today or date.today()
    facts: dict[str, Any] = {}

    if order:
        facts["order_no"] = order.get("order_no")
        facts["category"] = order.get("category")
        facts["amount"] = order.get("amount")
        facts["order_status"] = order.get("status")
        facts["opened"] = bool(order.get("opened"))
        facts["is_custom"] = bool(order.get("is_custom"))
        facts["shipped"] = bool(order.get("shipped_at"))
        facts["signed"] = bool(order.get("signed_at"))

        paid = parse_date(order.get("paid_at"))
        signed = parse_date(order.get("signed_at"))
        facts["days_since_paid"] = (today - paid).days if paid else None
        facts["days_since_sign"] = (today - signed).days if signed else None
    else:
        # 没有订单时字段保持缺失（None），规则引擎会走 UNKNOWN
        facts.update(
            {
                "order_no": None, "category": None, "amount": None, "order_status": None,
                "opened": None, "is_custom": None, "shipped": None, "signed": None,
                "days_since_paid": None, "days_since_sign": None,
            }
        )

    facts["member_level"] = (user or {}).get("level", "normal")

    # 对话中补充的信息可以覆盖订单事实
    for name in ("opened", "is_custom", "category", "has_quality_issue", "has_evidence"):
        v = _slot_value(slots, name)
        if v is not None:
            facts[name] = v

    facts.setdefault("has_quality_issue", None)
    facts.setdefault("has_evidence", None)
    return facts


def fact_summary(facts: dict[str, Any]) -> dict[str, Any]:
    """给 trace / 调试看的精简事实视图。"""
    keys = (
        "order_no", "category", "order_status", "days_since_sign",
        "opened", "is_custom", "has_quality_issue", "has_evidence",
        "member_level", "amount",
    )
    return {k: facts.get(k) for k in keys}
