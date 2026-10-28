"""离线评测：把「感觉还行」变成可回归的数字。

为什么不用检索指标当主指标：Recall@5 98% 这种数字说明「条款被召回了很多次」，
但不说明「用户的问题被解决了」。客服真正关心的是：答对没有、该问的时候有没有问、
不该转人工的时候有没有转。所以这里以**端到端口径**为主，检索指标作为解释性辅助。
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Case:
    case_id: str
    query: str
    user: str = "u_demo"
    session: list[str] = field(default_factory=list)   # 多轮前文
    expect: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


def load_cases(path: Path | str) -> list[Case]:
    cases: list[Case] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        raw = json.loads(line)
        cases.append(Case(
            case_id=raw.get("id") or f"case-{len(cases) + 1}",
            query=raw["query"],
            user=raw.get("user", "u_demo"),
            session=raw.get("session", []),
            expect=raw.get("expect", {}),
            tags=raw.get("tags", []),
        ))
    return cases


@dataclass
class CaseResult:
    case: Case
    output: dict[str, Any]
    hits: dict[str, bool]

    @property
    def all_hit(self) -> bool:
        return all(self.hits.values()) if self.hits else True


@dataclass
class EvalReport:
    results: list[CaseResult]
    metrics: dict[str, float]
    slices: dict[str, dict[str, float]]
    cost: dict[str, Any]
    failures: list[CaseResult]

    @property
    def passed(self) -> bool:
        return (
            self.metrics.get("decision_accuracy", 0) >= 0.80
            and self.metrics.get("clause_hit", 0) >= 0.70
            and self.metrics.get("verify_pass_rate", 0) >= 0.85
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_cases": len(self.results),
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "slices": {k: {kk: round(vv, 4) for kk, vv in v.items()}
                       for k, v in self.slices.items()},
            "cost": self.cost,
            "passed": self.passed,
            "failures": [
                {"id": r.case.case_id, "query": r.case.query,
                 "expect": r.case.expect, "got": {
                     "decision": r.output.get("decision"),
                     "intent": r.output.get("intent"),
                     "clause": (r.output.get("policy") or {}).get("clause_id"),
                     "reply": r.output.get("reply"),
                 }}
                for r in self.failures
            ],
        }

    def to_markdown(self) -> str:
        lines = ["## 评测报告", "", f"- 用例数：{len(self.results)}", ""]
        lines += ["| 指标 | 值 |", "| --- | --- |"]
        for k, v in self.metrics.items():
            lines.append(f"| {k} | {v:.3f} |")
        lines += ["", "### 分层切片", "", "| 切片 | 决策准确率 | 条款命中 | 用例数 |",
                  "| --- | --- | --- | --- |"]
        for name, m in self.slices.items():
            lines.append(f"| {name} | {m.get('decision', 0):.3f} | "
                         f"{m.get('clause', 0):.3f} | {int(m.get('n', 0))} |")
        lines += ["", "### 成本与延迟", "", "```",
                  json.dumps(self.cost, ensure_ascii=False, indent=2), "```", ""]
        if self.failures:
            lines += ["### 失败用例（前 10 条）", ""]
            for r in self.failures[:10]:
                got_clause = (r.output.get("policy") or {}).get("clause_id")
                lines.append(
                    f"- `{r.case.case_id}` {r.case.query} → decision={r.output.get('decision')} "
                    f"clause={got_clause} 期望={r.case.expect}"
                )
        return "\n".join(lines)


def _score_case(case: Case, output: dict[str, Any]) -> dict[str, bool]:
    expect = case.expect
    hits: dict[str, bool] = {}
    if "decision" in expect:
        hits["decision"] = output.get("decision") == expect["decision"]
    if "intent" in expect:
        hits["intent"] = output.get("intent") == expect["intent"]
    if "route" in expect:
        hits["route"] = output.get("route") == expect["route"]
    if "clause" in expect:
        got = (output.get("policy") or {}).get("clause_id")
        hits["clause"] = got == expect["clause"]
    reply = output.get("reply") or ""
    for kw in expect.get("must_contain", []):
        hits[f"contain:{kw}"] = kw in reply
    for kw in expect.get("must_not_contain", []):
        hits[f"not:{kw}"] = kw not in reply
    if expect.get("verify", True) and output.get("decision") == "answer":
        hits["verified"] = bool((output.get("verification") or {}).get("passed", True))
    return hits


def summarize(results: Iterable[CaseResult], cost: dict[str, Any]) -> EvalReport:
    results = list(results)
    n = len(results) or 1

    def rate(key: str) -> float:
        vals = [r.hits[key] for r in results if key in r.hits]
        return sum(1 for v in vals if v) / len(vals) if vals else 0.0

    metrics = {
        "decision_accuracy": rate("decision"),
        "intent_accuracy": rate("intent"),
        "route_accuracy": rate("route"),
        "clause_hit": rate("clause"),
        "verify_pass_rate": rate("verified"),
        "escalate_rate": sum(1 for r in results if r.output.get("decision") == "escalate") / n,
        "clarify_rate": sum(1 for r in results if r.output.get("decision") == "clarify") / n,
        "cache_hit_rate": sum(1 for r in results if r.output.get("cache_hit")) / n,
        "avg_latency_ms": sum(r.output.get("latency_ms", 0) for r in results) / n,
    }

    slices: dict[str, dict[str, float]] = {}
    by_intent: dict[str, list[CaseResult]] = defaultdict(list)
    by_decision: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        by_intent[r.output.get("intent", "unknown")].append(r)
        by_decision[r.case.expect.get("decision", "any")].append(r)

    def slice_metrics(group: list[CaseResult]) -> dict[str, float]:
        dn = [r for r in group if "decision" in r.hits]
        cn = [r for r in group if "clause" in r.hits]
        return {
            "decision": sum(1 for r in dn if r.hits["decision"]) / len(dn) if dn else 0.0,
            "clause": sum(1 for r in cn if r.hits["clause"]) / len(cn) if cn else 0.0,
            "n": len(group),
        }

    for name, group in list(by_intent.items()) + list(by_decision.items()):
        slices[f"intent={name}" if name in by_intent else f"expect={name}"] = slice_metrics(group)

    failures = [r for r in results if not r.all_hit]
    return EvalReport(results=results, metrics=metrics, slices=slices,
                      cost=cost, failures=failures)
