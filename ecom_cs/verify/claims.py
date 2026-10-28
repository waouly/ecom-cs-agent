"""答案级核验：把回复拆成 claim，逐条判断有无证据支撑。

简历里的「生成前证据校验」只保证「答案方向对」，管不了「答案里的每一句话」。
最常见的线上事故是：政策说对了，但顺手多编了一句「我们会补偿您 50 元」。
所以要在生成后做 claim 级核验。

离线版用词面覆盖 + 条款锚点做 entailment 近似：回复里政策类断言必须能对上
某个条款 ID 或条款正文的高覆盖片段。接真实模型时把 ``_score`` 换成
NLI/交叉编码器即可，接口不变——这正是我想在面试里讲的「用低成本代理实现，
但保留升级路径」。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

from ..types import ClaimCheck, Evidence
from ..utils.text import overlap_ratio, split_sentences

CLAUSE_ID_RE = re.compile(r"(?:RET|QUA|REF|LOG|PRI|INV|MEM|FAQ)-[A-Z0-9]+-\d+")
# 「没有记录」类表述不是在断言某个事实，而是在描述数据缺口，
# 拿证据覆盖度去验证它必然误杀，所以直接跳过。
SKIP_PREFIXES = ("您好", "好的", "请问", "如果您", "如需", "如需进一步", "我帮您", "已为您", "您还",
                 "该订单目前没有", "暂无", "目前没有")


@dataclass
class VerificationResult:
    claims: list[ClaimCheck]
    passed: bool
    failed_claims: list[str]
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "n_claims": len(self.claims),
            "failed": self.failed_claims,
            "reason": self.reason,
            "claims": [
                {"claim": c.claim, "supported": c.supported, "score": round(c.score, 3),
                 "evidence": c.evidence_ids, "reason": c.reason}
                for c in self.claims
            ],
        }


class ClaimVerifier:
    def __init__(self, threshold: float = 0.42, max_claims: int = 12) -> None:
        self.threshold = threshold
        self.max_claims = max_claims

    # -- claim 抽取 --
    def extract(self, reply: str) -> list[str]:
        claims: list[str] = []
        for sent in split_sentences(reply):
            s = sent.strip(" 　-•·\t")
            if len(s) < 6:
                continue
            if s.startswith(SKIP_PREFIXES):
                continue
            if s.endswith(("？", "?")):
                continue
            claims.append(s)
            if len(claims) >= self.max_claims:
                break
        return claims

    # -- 支持度 --
    def _score(self, claim: str, evidence: list[Evidence]) -> tuple[float, list[str], str]:
        if not evidence:
            return 0.0, [], "no_evidence"

        best = 0.0
        best_ids: list[str] = []
        for ev in evidence:
            text = ev.parent_text or ev.text or ""
            score = overlap_ratio(claim, text)
            # 条款 ID 被直接引用，是强证据
            if ev.clause_id and ev.clause_id in claim:
                score = max(score, 0.85)
            # 工具事实（如「已签收」「退款中」）通常短语短、覆盖低，给一点补偿
            if score > best:
                best, best_ids = score, [ev.clause_id]
            elif score == best and ev.clause_id not in best_ids:
                best_ids.append(ev.clause_id)

        reason = "ok" if best >= self.threshold else "low_overlap"
        return best, best_ids, reason

    def _score_extra(self, claim: str, extras: list[str]) -> float:
        best = 0.0
        for text in extras:
            if not text:
                continue
            best = max(best, overlap_ratio(claim, text))
            if best >= 0.999:
                break
        return best

    def verify(
        self,
        reply: str,
        evidence: Iterable[Evidence],
        extras: Iterable[str] = (),
        trusted: Iterable[str] = (),
        strict: bool = False,
    ) -> VerificationResult:
        """``extras`` 是「非检索来的」事实文本：条款正文、工具返回、槽位值。

        只拿知识库证据做核验会误杀两类正确内容：
        1. 规则引擎判出来的结论（「因此您可以申请全额退款」）——它来自条款的
           action，而不是条款原文的某个片段；
        2. 工具查出来的事实（「实付 ¥299.0」「已签收」）——它根本不在知识库里。
        所以这里把这两类作为 ``extras`` 参与覆盖度计算，``trusted`` 则是
        确定性引擎直接产出的短语，命中即视为已核实（reason=engine_fact）。
        """
        evidence = list(evidence)
        extras = [e for e in extras if e]
        trusted = [t for t in trusted if t]
        claims = self.extract(reply)
        checks: list[ClaimCheck] = []
        failed: list[str] = []

        for claim in claims:
            score, ids, reason = self._score(claim, evidence)
            extra_score = self._score_extra(claim, extras)
            if extra_score > score:
                score, ids, reason = extra_score, ["engine"], "ok"
            for t in trusted:
                if t in claim:
                    score, ids, reason = 1.0, ["engine"], "engine_fact"
                    break
            supported = score >= self.threshold
            checks.append(ClaimCheck(claim=claim, supported=supported, score=score,
                                     evidence_ids=ids, reason=reason))
            if not supported:
                failed.append(claim)

        passed = not failed
        if strict and len(failed) > 0:
            passed = False
        return VerificationResult(claims=checks, passed=passed, failed_claims=failed,
                                  reason="" if passed else "unsupported_claims")


def strip_clause_ids(reply: str) -> str:
    """面向用户的最终回复里去掉内部条款编号（除非是政策问答场景显式保留）。"""
    return CLAUSE_ID_RE.sub("", reply).strip()
