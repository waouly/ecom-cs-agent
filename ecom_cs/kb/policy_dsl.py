"""政策 DSL：条件表达式的解析、校验与三值求值。

三值逻辑是这套系统的关键：``True`` 表示条件满足，``False`` 表示明确不满足，
``None`` 表示「事实缺失」。缺失不能当 False——它必须变成一次追问。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Optional

from .models import Clause, Condition, ExceptionRule

VALID_OPS = {"==", "!=", "<", "<=", ">", ">=", "in", "not_in", "exists"}
VALID_ACTIONS = {
    "refund_full", "refund_partial", "refund_shipping", "exchange",
    "repair", "reject", "compensate", "special_handling", "request_evidence",
}

TriBool = Optional[bool]


class DSLError(ValueError):
    pass


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------


def parse_condition(raw: dict[str, Any]) -> Condition:
    if "field" not in raw or "op" not in raw:
        raise DSLError(f"条件缺少 field/op: {raw}")
    op = raw["op"]
    if op not in VALID_OPS:
        raise DSLError(f"不支持的操作符 {op}: {raw}")
    return Condition(
        field=raw["field"],
        op=op,
        value=raw.get("value"),
        label=raw.get("label") or raw["field"],
    )


def parse_exception(raw: dict[str, Any]) -> ExceptionRule:
    if not raw.get("exception_id"):
        raise DSLError(f"例外缺少 exception_id: {raw}")
    action = raw.get("action", "special_handling")
    if action not in VALID_ACTIONS:
        raise DSLError(f"例外 action 非法: {action}")
    return ExceptionRule(
        exception_id=raw["exception_id"],
        when=[parse_condition(c) for c in raw.get("when", [])],
        action=action,
        text=raw.get("text", ""),
    )


def parse_clause(raw: dict[str, Any], doc_id: str, doc_title: str) -> Clause:
    cid = raw.get("clause_id")
    if not cid:
        raise DSLError(f"条款缺少 clause_id: {raw}")
    action = raw.get("action", "special_handling")
    if action not in VALID_ACTIONS:
        raise DSLError(f"{cid} action 非法: {action}")
    conditions = [parse_condition(c) for c in raw.get("conditions", [])]
    applies_to = raw.get("applies_to") or {}
    # 没有条件也没有类目约束的条款只能是「信息型」，不能用来下处置结论
    default_type = "info" if (not conditions and not applies_to) else "eligibility"
    return Clause(
        clause_id=cid,
        title=raw.get("title", cid),
        text=raw.get("text", ""),
        action=action,
        doc_id=doc_id,
        doc_title=doc_title,
        applies_to=applies_to,
        conditions=conditions,
        exceptions=[parse_exception(e) for e in raw.get("exceptions", [])],
        effective_from=raw.get("effective_from", ""),
        effective_to=raw.get("effective_to"),
        priority=int(raw.get("priority", 50)),
        keywords=list(raw.get("keywords", [])),
        policy_type=raw.get("policy_type", default_type),
    )


def validate_clause(clause: Clause) -> list[str]:
    """静态校验，返回错误列表（空表示通过）。"""
    errors: list[str] = []
    if not clause.text:
        errors.append(f"{clause.clause_id}: text 为空")
    if clause.policy_type not in {"eligibility", "info"}:
        errors.append(f"{clause.clause_id}: policy_type 非法")
    seen_exc: set[str] = set()
    for ex in clause.exceptions:
        if ex.exception_id in seen_exc:
            errors.append(f"{clause.clause_id}: 例外 {ex.exception_id} 重复")
        seen_exc.add(ex.exception_id)
        if not ex.when:
            errors.append(f"{clause.clause_id}: 例外 {ex.exception_id} 的 when 为空")
    for cond in clause.conditions:
        if cond.op in {"in", "not_in"} and not isinstance(cond.value, (list, tuple, set)):
            errors.append(f"{clause.clause_id}: {cond.field} 的 {cond.op} 需要列表值")
    return errors


# ---------------------------------------------------------------------------
# 三值求值
# ---------------------------------------------------------------------------


def _as_number(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v))
    except (TypeError, ValueError):
        return None


def eval_condition(cond: Condition, facts: dict[str, Any]) -> TriBool:
    if cond.field not in facts:
        return None
    actual = facts[cond.field]
    if actual is None:
        return None

    op, target = cond.op, cond.value
    try:
        if op == "exists":
            return True
        if op == "==":
            if isinstance(target, bool) or isinstance(actual, bool):
                return bool(actual) == bool(target)
            if isinstance(target, (int, float)) and isinstance(actual, (int, float)):
                return float(actual) == float(target)
            return str(actual) == str(target)
        if op == "!=":
            r = eval_condition(Condition(cond.field, "==", target, cond.label), facts)
            return None if r is None else (not r)
        if op in {"<", "<=", ">", ">="}:
            a, b = _as_number(actual), _as_number(target)
            if a is None or b is None:
                return None
            return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]
        if op == "in":
            return actual in (target or [])
        if op == "not_in":
            return actual not in (target or [])
    except TypeError:
        return None
    raise DSLError(f"未知操作符 {op}")


def eval_all(conds: list[Condition], facts: dict[str, Any]) -> TriBool:
    """AND 的三值语义：任一 False -> False；否则任一 None -> None；否则 True。"""
    if not conds:
        return True
    unknown = False
    for c in conds:
        r = eval_condition(c, facts)
        if r is False:
            return False
        if r is None:
            unknown = True
    return None if unknown else True


# ---------------------------------------------------------------------------
# 生效期
# ---------------------------------------------------------------------------


def _parse_date(s: str | None) -> Optional[date]:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def is_effective(clause: Clause, today: Optional[date] = None) -> bool:
    today = today or date.today()
    start = _parse_date(clause.effective_from)
    end = _parse_date(clause.effective_to)
    if start and today < start:
        return False
    if end and today > end:
        return False
    return True


def matches_scope(clause: Clause, facts: dict[str, Any]) -> bool:
    cats = clause.scope_categories
    if not cats:
        return True
    return facts.get("category") in cats


def format_value(v: Any) -> str:
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if v is None:
        return "未知"
    return str(v)


@dataclass
class CondDetail:
    """单条件求值明细，给「为什么」提供依据。"""

    label: str
    field: str
    satisfied: TriBool
    actual: Any
