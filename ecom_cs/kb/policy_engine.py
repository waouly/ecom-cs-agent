"""政策引擎：加载条款，按事实求值，选出最该被采纳的那一条。

设计取舍（面试可展开）：
1. 常见做法是把政策文本当 RAG 语料直接喂给模型，让模型「读懂政策再回答」。
   问题是政策是**判定规则**，不是知识——模型只能给出「大概符合/大概不符合」，
   无法保证「签收第 8 天」这种边界情况不出错。这里把条款编译成条件表达式，
   由确定性规则引擎判定，模型只负责把结论说人话。
2. 三值逻辑：缺字段返回 UNKNOWN 而不是 False。False 意味着「明确不给办」，
   UNKNOWN 意味着「先问一句」。这两者混淆是客服系统最常见的体验事故。
3. 例外条款（quality 覆盖 opened）单独建模，而不是把条件写成一个巨大 OR 表达式，
   因为例外需要独立引用与独立生效期，也要能在答案里说清「按哪条例外办」。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Optional

from ..types import PolicyOutcome, PolicyStatus
from .models import Clause, PolicyDoc
from .policy_dsl import (
    CondDetail,
    DSLError,
    eval_all,
    eval_condition,
    format_value,
    is_effective,
    matches_scope,
    parse_clause,
    validate_clause,
)

STATUS_RANK = {
    PolicyStatus.APPLICABLE: 3,
    PolicyStatus.UNKNOWN: 2,
    PolicyStatus.NOT_APPLICABLE: 1,
}


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------


def load_docs(policies_dir: Path | str) -> tuple[list[PolicyDoc], list[str]]:
    """读取目录下所有政策 JSON。返回 (文档列表, 错误列表)。"""
    docs: list[PolicyDoc] = []
    errors: list[str] = []
    for path in sorted(Path(policies_dir).glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            errors.append(f"{path.name}: JSON 解析失败 {e}")
            continue
        try:
            doc = _parse_doc(raw)
        except DSLError as e:
            errors.append(f"{path.name}: {e}")
            continue
        for clause in doc.clauses:
            errors.extend(validate_clause(clause))
        docs.append(doc)
    return docs, errors


def _parse_doc(raw: dict[str, Any]) -> PolicyDoc:
    doc_id = raw.get("doc_id") or ""
    doc_title = raw.get("title") or doc_id
    if not doc_id:
        raise DSLError("文档缺少 doc_id")
    clauses = [parse_clause(c, doc_id, doc_title) for c in raw.get("clauses", [])]
    ids = [c.clause_id for c in clauses]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise DSLError(f"{doc_id}: 条款 ID 重复 {sorted(dup)}")
    return PolicyDoc(
        doc_id=doc_id,
        title=doc_title,
        category=raw.get("category", ""),
        version=raw.get("version", ""),
        updated_at=raw.get("updated_at", ""),
        overview=raw.get("overview", ""),
        clauses=clauses,
    )


# ---------------------------------------------------------------------------
# 引擎
# ---------------------------------------------------------------------------


@dataclass
class EvaluationBundle:
    eligibility: list[PolicyOutcome] = field(default_factory=list)
    info: list[PolicyOutcome] = field(default_factory=list)

    @property
    def primary(self) -> Optional[PolicyOutcome]:
        """用来决定 answer/clarify/escalate 的那一条。UNKNOWN 优先于 NOT_APPLICABLE，
        因为「缺信息」比「明确拒绝」更需要对用户负责。

        没有资格类结论时退回到信息类（例如「退款多久到账」这种纯时效问题）——
        这类问题本来就不产生资格结论，不该被当成「无法回答」。
        """
        if self.eligibility:
            # 先比「限定性」：只对某类商品生效的条款，压过不限类目的通用条款。
            # 再比状态（UNKNOWN 优先于 NOT_APPLICABLE，因为缺信息比直接拒绝更该问一句），
            # 最后用原始顺序做稳定排序。
            return max(
                self.eligibility,
                key=lambda o: (o.scope_specific, STATUS_RANK[o.status],
                               -(self.eligibility.index(o))),
            )
        applicable_info = [o for o in self.info if o.status == PolicyStatus.APPLICABLE]
        return applicable_info[0] if applicable_info else None

    @property
    def has_eligibility(self) -> bool:
        return bool(self.eligibility)

    @property
    def info_applicable(self) -> Optional[PolicyOutcome]:
        """信息类里能直接回答的那条（例如「退款 1-3 个工作日到账」）。"""
        for o in self.info:
            if o.status == PolicyStatus.APPLICABLE:
                return o
        return None

    @property
    def needs_clarify(self) -> bool:
        return self.primary is not None and self.primary.status == PolicyStatus.UNKNOWN


class PolicyEngine:
    def __init__(
        self,
        docs: Iterable[PolicyDoc],
        today: Optional[date] = None,
        clarify_on_exception: bool = True,
    ) -> None:
        self.docs: list[PolicyDoc] = list(docs)
        self.today = today or date.today()
        # 基础条件明确不满足、但例外条款的前置事实未知时，选择「先问一句」
        # 而不是直接拒绝。例：已拆封（明确不满足）+ 是否有质量问题（未知）
        # -> 拆封本身是拒绝理由，但质量问题可以翻盘，所以该问。
        self.clarify_on_exception = clarify_on_exception
        self.load_errors: list[str] = []
        self._clauses: dict[str, Clause] = {}
        for d in self.docs:
            for c in d.clauses:
                self._clauses[c.clause_id] = c

    @classmethod
    def load(
        cls,
        policies_dir: Path | str,
        today: Optional[date] = None,
        clarify_on_exception: bool = True,
    ) -> "PolicyEngine":
        docs, errors = load_docs(policies_dir)
        engine = cls(docs, today=today, clarify_on_exception=clarify_on_exception)
        engine.load_errors = errors
        return engine

    # -- 查询 --
    @property
    def clauses(self) -> list[Clause]:
        return list(self._clauses.values())

    def by_id(self, clause_id: str) -> Optional[Clause]:
        return self._clauses.get(clause_id)

    def candidates(self, facts: dict[str, Any]) -> list[Clause]:
        """按类目与生效期筛出候选条款（检索失败时的安全网）。

        排序让「限定类目」和「优先级高」的条款靠前：定制/内衣这类特例条款
        必须压过七天无理由的通用条款，否则先命中的永远是通用款。
        """
        found = [
            c for c in self._clauses.values()
            if is_effective(c, self.today) and matches_scope(c, facts)
        ]
        found.sort(key=lambda c: (not bool(c.applies_to), -c.priority, c.clause_id))
        return found

    # -- 求值 --
    def evaluate_clause(self, clause: Clause, facts: dict[str, Any]) -> Optional[PolicyOutcome]:
        if not is_effective(clause, self.today):
            return None
        if not matches_scope(clause, facts):
            # 类目未知（用户没给订单）时不能判「不适用」，否则「能退吗」会被直接拒答。
            # 正确做法是先问订单/类目，所以这里降级成 UNKNOWN 并带上缺失字段。
            if clause.applies_to and not facts.get("category") and not facts.get("is_custom"):
                return PolicyOutcome(
                    clause_id=clause.clause_id,
                    title=clause.title,
                    action=clause.action,
                    status=PolicyStatus.UNKNOWN,
                    missing=["category"],
                    explanation="商品类目未知，无法判断该条款是否适用于此订单",
                    effective_from=clause.effective_from,
                    scope_specific=bool(clause.applies_to),
                )
            return None

        details = [
            CondDetail(
                label=c.label or c.field,
                field=c.field,
                satisfied=eval_condition(c, facts),
                actual=facts.get(c.field),
            )
            for c in clause.conditions
        ]
        base = eval_all(clause.conditions, facts)

        satisfied = [d.label for d in details if d.satisfied is True]
        violated = [
            f"{d.label}，实际为 {format_value(d.actual)}"
            for d in details if d.satisfied is False
        ]
        missing = [d.field for d in details if d.satisfied is None]

        true_exs: list[Any] = []
        unknown_exs: list[Any] = []
        for ex in clause.exceptions:
            r = eval_all(ex.when, facts)
            if r is True:
                true_exs.append(ex)
            elif r is None:
                unknown_exs.append(ex)

        def build(status: PolicyStatus, action: str, exception_id: str = "", explanation: str = "") -> PolicyOutcome:
            extra_missing: list[str] = []
            if status == PolicyStatus.UNKNOWN and not missing:
                for ex in unknown_exs:
                    extra_missing.extend(
                        c.field for c in ex.when if eval_condition(c, facts) is None
                    )
            return PolicyOutcome(
                clause_id=clause.clause_id,
                title=clause.title,
                action=action,
                status=status,
                satisfied=list(satisfied),
                violated=list(violated),
                missing=list(dict.fromkeys(missing + extra_missing)),
                exception_id=exception_id,
                explanation=explanation or clause.text,
                effective_from=clause.effective_from,
                raw_conditions=[c.to_dict() for c in clause.conditions],
                scope_specific=bool(clause.applies_to),
            )

        # 例外命中：优先级最高，直接覆盖基础条件（质量问题可突破「未拆封」限制）
        if true_exs:
            ex = true_exs[0]
            out = build(
                PolicyStatus.APPLICABLE,
                ex.action,
                exception_id=ex.exception_id,
                explanation=ex.text,
            )
            out.satisfied.insert(0, f"命中例外条款：{ex.exception_id}")
            return out

        if base is True:
            return build(PolicyStatus.APPLICABLE, clause.action)
        if base is False:
            # 「已拆封 / 定制」这类结果是**确定性的拒绝**：再问一句也没有意义，
            # 直接给否定结论并附替代路径（质量售后）。只有「超期」这种还可能
            # 走质量退货窗口的情形，才值得停下来问一句。
            definitive = any(("拆封" in v) or ("定制" in v) for v in violated)
            if unknown_exs and self.clarify_on_exception and not definitive:
                return build(PolicyStatus.UNKNOWN, clause.action)
            return build(PolicyStatus.NOT_APPLICABLE, "reject")
        return build(PolicyStatus.UNKNOWN, clause.action)

    def evaluate(
        self,
        facts: dict[str, Any],
        clause_ids: Optional[list[str]] = None,
    ) -> EvaluationBundle:
        """优先只评估「被检索命中」的条款，命中为空时退化为按类目全量评估。"""
        clauses: list[Clause] = []
        if clause_ids:
            for cid in clause_ids:
                c = self._clauses.get(cid)
                if c is not None:
                    clauses.append(c)
        else:
            clauses = self.candidates(facts)

        bundle = EvaluationBundle()
        for c in clauses:
            out = self.evaluate_clause(c, facts)
            if out is None:
                continue
            (bundle.eligibility if c.policy_type == "eligibility" else bundle.info).append(out)
        return bundle

    # -- 自检 --
    def stats(self) -> dict[str, Any]:
        types: dict[str, int] = {}
        for c in self._clauses.values():
            types[c.policy_type] = types.get(c.policy_type, 0) + 1
        return {
            "docs": len(self.docs),
            "clauses": len(self._clauses),
            "by_type": types,
            "categories": sorted(
                {cat for c in self._clauses.values() for cat in c.scope_categories}
            ),
        }
