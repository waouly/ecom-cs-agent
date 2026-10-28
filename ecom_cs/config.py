"""全局配置。

所有配置项都可以被环境变量覆盖，命名规则：``ECOM_CS_<FIELD_NAME>``。
默认值全部指向离线可跑（mock LLM / hashing embedding / 内存向量库 / sqlite）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_ROOT.parent


def _coerce(default: object, raw: str) -> object:
    if isinstance(default, bool):
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(default, int) and not isinstance(default, bool):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, Path):
        return Path(raw).expanduser()
    return raw


@dataclass(frozen=True)
class Settings:
    # ---- 路径 ----
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    configs_dir: Path = PROJECT_ROOT / "configs"
    db_path: Path = PROJECT_ROOT / "data" / "app.db"
    index_dir: Path = PROJECT_ROOT / "data" / "index"
    policies_path: Path = PROJECT_ROOT / "configs" / "policies"
    intents_path: Path = PROJECT_ROOT / "configs" / "intents.json"
    calibration_path: Path = PROJECT_ROOT / "configs" / "calibration.json"
    traces_dir: Path = PROJECT_ROOT / "data" / "traces"
    reports_dir: Path = PROJECT_ROOT / "data" / "reports"
    eval_dir: Path = PROJECT_ROOT / "data" / "eval"

    # ---- 模型 ----
    llm_provider: str = "mock"          # mock | openai
    llm_model: str = "mock-llm-1"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_temperature: float = 0.0
    llm_timeout_s: float = 30.0

    embed_provider: str = "hashing"     # hashing | bge
    embed_model: str = "BAAI/bge-m3"
    embed_dim: int = 512

    vector_backend: str = "memory"      # memory | milvus
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "ecom_policy"

    # ---- 检索 ----
    top_k_dense: int = 20
    top_k_sparse: int = 20
    rrf_k: int = 60
    top_k_final: int = 5
    fusion: str = "rrf"                 # rrf | weighted
    weighted_dense: float = 0.6
    rerank_threshold: float = 0.05

    # ---- 语义缓存 ----
    cache_enabled: bool = True
    cache_threshold: float = 0.95
    cache_ttl_s: int = 3600

    # ---- 风险控制 ----
    conformal_alpha: float = 0.05
    default_threshold: float = 0.62
    max_retry_before_escalate: int = 1

    # ---- 消融开关（评测时需要能单独关掉）----
    enable_rewrite: bool = True
    enable_coref: bool = True
    enable_dense: bool = True
    enable_bm25: bool = True
    enable_rerank: bool = True
    enable_claims: bool = True
    enable_numeric_guard: bool = True
    enable_safety: bool = True
    enable_cache: bool = True
    enable_cascade: bool = True

    # ---- Agent ----
    agent_max_steps: int = 4
    agent_timeout_s: float = 8.0

    # ---- 成本（美元 / 千 token，仅用于估算展示）----
    price_in_per_1k: float = 0.0005
    price_out_per_1k: float = 0.0015

    @classmethod
    def load(cls) -> "Settings":
        base = cls()
        overrides: dict[str, object] = {}
        for f in fields(cls):
            env_name = "ECOM_CS_" + f.name.upper()
            raw = os.getenv(env_name)
            if raw is None or raw == "":
                continue
            overrides[f.name] = _coerce(f.default, raw)
        return replace(base, **overrides)

    def ensure_dirs(self) -> None:
        for p in (
            self.data_dir,
            self.index_dir,
            self.traces_dir,
            self.reports_dir,
            self.eval_dir,
            self.db_path.parent,
        ):
            Path(p).mkdir(parents=True, exist_ok=True)

    def ablation(self, **flags: bool) -> "Settings":
        """返回一份关掉指定开关的配置副本，用于消融实验。"""
        return replace(self, **flags)


_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = Settings.load()
        _SETTINGS.ensure_dirs()
    return _SETTINGS


def reset_settings() -> None:
    global _SETTINGS
    _SETTINGS = None
