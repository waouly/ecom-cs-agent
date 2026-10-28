"""知识库数据模型：政策文档 -> 条款 -> 检索块。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Condition:
    field: str
    op: str
    value: Any = None
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"field": self.field, "op": self.op, "value": self.value, "label": self.label}


@dataclass
class ExceptionRule:
    exception_id: str
    when: list[Condition]
    action: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "exception_id": self.exception_id,
            "when": [c.to_dict() for c in self.when],
            "action": self.action,
            "text": self.text,
        }


@dataclass
class Clause:
    clause_id: str
    title: str
    text: str
    action: str
    doc_id: str
    doc_title: str
    applies_to: dict[str, Any] = field(default_factory=dict)
    conditions: list[Condition] = field(default_factory=list)
    exceptions: list[ExceptionRule] = field(default_factory=list)
    effective_from: str = ""
    effective_to: Optional[str] = None
    priority: int = 50
    keywords: list[str] = field(default_factory=list)
    policy_type: str = "eligibility"     # eligibility | info
    item_id: str = ""                    # 知识闭环更新后的版本占位

    @property
    def scope_categories(self) -> list[str]:
        cats = self.applies_to.get("category") or []
        return cats if isinstance(cats, list) else [cats]

    def to_dict(self) -> dict[str, Any]:
        return {
            "clause_id": self.clause_id,
            "title": self.title,
            "text": self.text,
            "action": self.action,
            "doc_id": self.doc_id,
            "applies_to": self.applies_to,
            "conditions": [c.to_dict() for c in self.conditions],
            "exceptions": [e.to_dict() for e in self.exceptions],
            "effective_from": self.effective_from,
            "effective_to": self.effective_to,
            "priority": self.priority,
            "keywords": self.keywords,
            "policy_type": self.policy_type,
        }


@dataclass
class PolicyDoc:
    doc_id: str
    title: str
    category: str
    version: str
    updated_at: str
    overview: str
    clauses: list[Clause] = field(default_factory=list)


@dataclass
class Chunk:
    """检索单元：小块 = 条款正文；父块 = 政策概览 + 条款 + 例外说明。"""

    chunk_id: str
    doc_id: str
    title: str
    text: str
    parent_text: str
    source: str = "policy"
    keywords: list[str] = field(default_factory=list)
    effective_from: str = ""
    effective_to: Optional[str] = None
