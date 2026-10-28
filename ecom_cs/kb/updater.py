"""知识闭环：未解决问题 -> 人工审核 -> 政策库更新 -> 生效。

常见 RAG 系统在这里是断的：badcase 只存在日志里，知识更新靠人改文档重跑。
这里把闭环做成显式的「覆盖层」：基础政策 JSON 只读，增量更新写在 overlay 里，
带审核人、时间、原因和关联的 gap id，可回滚、可追溯——这正好对上简历里
「未解决问题记录 -> 人工审核 -> 知识库更新」的描述，但把它做实了。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

from .models import PolicyDoc
from .policy_dsl import parse_clause
from .policy_engine import load_docs


@dataclass
class KBUpdate:
    clause_id: str
    kind: str                       # update | retire
    patch: dict[str, Any] = field(default_factory=dict)
    reviewer: str = ""
    reviewed_at: str = ""
    reason: str = ""
    gap_ids: list[int] = field(default_factory=list)
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "clause_id": self.clause_id,
            "kind": self.kind,
            "patch": self.patch,
            "reviewer": self.reviewer,
            "reviewed_at": self.reviewed_at,
            "reason": self.reason,
            "gap_ids": self.gap_ids,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "KBUpdate":
        return cls(
            clause_id=raw["clause_id"],
            kind=raw.get("kind", "update"),
            patch=raw.get("patch", {}),
            reviewer=raw.get("reviewer", ""),
            reviewed_at=raw.get("reviewed_at", ""),
            reason=raw.get("reason", ""),
            gap_ids=raw.get("gap_ids", []),
            version=int(raw.get("version", 1)),
        )


def apply_overlay(docs: list[PolicyDoc], updates: list[KBUpdate]) -> list[PolicyDoc]:
    """把覆盖层合并进基础文档。retire 用 effective_to 标记失效，不删历史版本。"""
    by_id = {u.clause_id: u for u in updates}
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    for doc in docs:
        for i, clause in enumerate(doc.clauses):
            upd = by_id.get(clause.clause_id)
            if upd is None:
                continue
            if upd.kind == "retire":
                clause.effective_to = yesterday
                continue
            merged = {**clause.to_dict(), **upd.patch}
            new_clause = parse_clause(merged, clause.doc_id, clause.doc_title)
            doc.clauses[i] = new_clause
    return docs


class KnowledgeUpdater:
    """管理 overlay 文件，并对外提供待审核 gap 列表。"""

    def __init__(self, overlay_path: Path | str, repo: Any = None) -> None:
        self.overlay_path = Path(overlay_path)
        self.repo = repo
        self._updates: list[KBUpdate] = []
        self.load()

    def load(self) -> None:
        if self.overlay_path.exists():
            raw = json.loads(self.overlay_path.read_text(encoding="utf-8"))
            self._updates = [KBUpdate.from_dict(x) for x in raw.get("updates", [])]
        else:
            self._updates = []

    def save(self) -> None:
        self.overlay_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"updates": [u.to_dict() for u in self._updates]}
        self.overlay_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @property
    def updates(self) -> list[KBUpdate]:
        return list(self._updates)

    # -- 闭环入口 --
    def record_gap(
        self,
        query: str,
        intent: str = "",
        confidence: float = 0.0,
        decision: str = "",
        note: str = "",
    ) -> int:
        if self.repo is None:
            raise RuntimeError("KnowledgeUpdater 需要 Repository 才能记录 gap")
        return self.repo.add_gap(query=query, intent=intent, confidence=confidence,
                                 decision=decision, cluster=note)

    def pending_gaps(self, limit: int = 50) -> list[dict[str, Any]]:
        if self.repo is None:
            return []
        return self.repo.list_gaps(limit=limit, unresolved_only=True)

    def apply_update(
        self,
        clause_id: str,
        patch: dict[str, Any],
        reviewer: str,
        reason: str = "",
        gap_ids: Optional[list[int]] = None,
    ) -> KBUpdate:
        if not patch:
            raise ValueError("patch 不能为空")
        version = 1 + max(
            [u.version for u in self._updates if u.clause_id == clause_id] or [0]
        )
        upd = KBUpdate(
            clause_id=clause_id,
            kind="update",
            patch=patch,
            reviewer=reviewer,
            reviewed_at=date.today().isoformat(),
            reason=reason,
            gap_ids=list(gap_ids or []),
            version=version,
        )
        self._updates.append(upd)
        self.save()
        self._after_change(upd)
        return upd

    def retire(self, clause_id: str, reviewer: str, reason: str = "") -> KBUpdate:
        upd = KBUpdate(
            clause_id=clause_id,
            kind="retire",
            reviewer=reviewer,
            reviewed_at=date.today().isoformat(),
            reason=reason,
        )
        self._updates.append(upd)
        self.save()
        self._after_change(upd)
        return upd

    def rollback(self, clause_id: str) -> int:
        before = len(self._updates)
        self._updates = [u for u in self._updates if u.clause_id != clause_id]
        self.save()
        return before - len(self._updates)

    def _after_change(self, upd: KBUpdate) -> None:
        if self.repo is None:
            return
        for gid in upd.gap_ids:
            self.repo.resolve_gap(gid)
        self.repo.conn.commit()
        self.repo.add_audit(
            actor="kb_reviewer",
            action=upd.kind,
            target=upd.clause_id,
            payload=upd.to_dict(),
            result="applied",
        )
        self.repo.conn.commit()

    def merged_engine(self, policies_dir: Path | str, today: Optional[date] = None):
        """生成合并了覆盖层的引擎实例（供 CLI / API 重建）。"""
        from .policy_engine import PolicyEngine

        docs, errors = load_docs(policies_dir)
        docs = apply_overlay(docs, self._updates)
        engine = PolicyEngine(docs, today=today)
        engine.load_errors = errors
        return engine
