"""评测执行器：逐条跑用例，收集输出，汇总成报告。"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

from ..config import Settings, get_settings
from ..db.seed import seed
from ..orchestrator import ChatService, build_deps
from .metrics import Case, CaseResult, EvalReport, load_cases, summarize, _score_case


def _reset_world(deps) -> None:
    """把业务库恢复到种子状态。

    评测里有写操作用例（改地址、确认下单等），如果不隔离，前一条用例
    真实写入的数据会让后一条用例看到「地址已经是新的」——第二轮 replay
    就再也弹不出确认，测试结果取决于执行顺序。这里每条用例重置一次，
    用一点重灌成本换「任意顺序都可复跑」的确定性。
    """
    deps.repo.reset()
    deps.repo.init_schema()
    seed(deps.repo, deps.today)


def run_eval(
    settings: Optional[Settings] = None,
    dataset: Path | str = "configs/eval/scenarios.jsonl",
    today: Optional[date] = None,
    verbose: bool = False,
    limit: int = 0,
) -> EvalReport:
    settings = settings or get_settings()
    deps = build_deps(settings=settings, today=today)
    svc = ChatService(deps=deps)
    cases = load_cases(dataset)
    if limit:
        cases = cases[:limit]

    results: list[CaseResult] = []
    for case in cases:
        _reset_world(deps)
        svc.store = type(svc.store)()  # 会话也按用例隔离，避免上一条的槽位泄漏
        session_id = f"eval-{case.case_id}"
        # 多轮用例：先按顺序重放前文，只对最后一句做判定
        for text in case.session:
            svc.respond(session_id, case.user, text)
        output = svc.respond(session_id, case.user, case.query)
        hits = _score_case(case, output)
        results.append(CaseResult(case=case, output=output, hits=hits))
        if verbose:
            status = "PASS" if all(hits.values()) else "FAIL"
            detail = ", ".join(f"{k}={v}" for k, v in hits.items())
            print(f"[{status}] {case.case_id} {case.query}")
            print(f"        {detail}")
            print(f"        reply: {output.get('reply')}")

    cost = deps.ledger.summary()
    cost["retrieval"] = deps.retrieval.stats()
    return summarize(results, cost)
