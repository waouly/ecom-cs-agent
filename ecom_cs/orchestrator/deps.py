"""依赖装配：把检索、引擎、工具、模型、成本、校准连成一张网。

全部依赖显式注入（不搞全局单例），这样评测里可以只换掉一个组件做消融，
比如把 reranker 换成恒等函数，看指标掉多少。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Optional

from ..config import Settings, get_settings
from ..cost import Cascade, CostLedger, SemanticCache
from ..db.repo import Repository
from ..db.seed import seed
from ..embed import build_embedder
from ..kb.chunker import build_chunks
from ..kb.models import Chunk
from ..kb.policy_engine import PolicyEngine, load_docs
from ..kb.updater import KnowledgeUpdater
from ..llm.tasks import build_answerer
from ..nlu.intent import IntentClassifier
from ..nlu.slots import SlotExtractor
from ..retrieval.pipeline import RetrievalService
from ..tools.base import ToolRegistry, build_registry
from ..verify.calibration import CalibrationBundle
from ..verify.claims import ClaimVerifier


@dataclass
class Deps:
    settings: Settings
    repo: Repository
    engine: PolicyEngine
    chunks: list[Chunk]
    embedder: Any
    retrieval: RetrievalService
    answerer: Any
    registry: ToolRegistry
    classifier: IntentClassifier
    slots: SlotExtractor
    cache: SemanticCache
    cascade: Cascade
    ledger: CostLedger
    calibration: CalibrationBundle
    claims: ClaimVerifier
    updater: KnowledgeUpdater
    today: date

    def summarize(self) -> dict[str, Any]:
        return {
            "today": self.today.isoformat(),
            "clauses": len(self.engine.clauses),
            "chunks": len(self.chunks),
            "tools": self.registry.names(),
            "llm": self.settings.llm_provider,
            "embed": self.settings.embed_provider,
            "vector": self.settings.vector_backend,
        }


def build_deps(
    settings: Optional[Settings] = None,
    today: Optional[date] = None,
    repo: Optional[Repository] = None,
    seed_data: bool = True,
) -> Deps:
    settings = settings or get_settings()
    settings.ensure_dirs()
    today = today or date.today()

    repo = repo or Repository(settings.db_path)
    repo.init_schema()
    if seed_data and repo.stats().get("users", 0) == 0:
        seed(repo, today=today)

    docs, _errors = load_docs(settings.policies_path)
    engine = PolicyEngine(docs, today)
    chunks = build_chunks(docs)

    embedder = build_embedder(settings)
    retrieval = RetrievalService.build(engine, embedder, settings)
    answerer = build_answerer(settings)
    classifier = IntentClassifier(embedder=embedder)

    cache = SemanticCache(embedder, threshold=settings.cache_threshold,
                          ttl_s=settings.cache_ttl_s)
    calibration = CalibrationBundle.load(settings.calibration_path)

    return Deps(
        settings=settings,
        repo=repo,
        engine=engine,
        chunks=chunks,
        embedder=embedder,
        retrieval=retrieval,
        answerer=answerer,
        registry=build_registry(),
        classifier=classifier,
        slots=SlotExtractor(),
        cache=cache,
        cascade=Cascade(),
        ledger=CostLedger(),
        calibration=calibration,
        claims=ClaimVerifier(),
        updater=KnowledgeUpdater(
            overlay_path=settings.data_dir / "kb_overlay.json",
            repo=repo,
        ),
        today=today,
    )
