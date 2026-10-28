"""贯穿全链路的数据模型。

设计原则：所有跨模块传递的对象都在这里定义，避免各模块私造 dict。
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class Intent(str, Enum):
    """9 类意图。"""

    POLICY_QA = "policy_qa"          # 售后政策问答（能不能退/多久到账）
    ORDER_QUERY = "order_query"      # 订单信息查询
    LOGISTICS = "logistics"          # 物流/发货进度
    AFTERSALE = "aftersale"          # 售后单进度
    REFUND = "refund"                # 申请退款/退货（写操作）
    COMPLAINT = "complaint"          # 投诉/情绪爆发
    CHITCHAT = "chitchat"            # 寒暄/闲聊
    INJECTION = "injection"          # 提示注入/越权尝试
    OUT_OF_SCOPE = "out_of_scope"    # 非电商域问题


class Route(str, Enum):
    """5 条处理路径。"""

    TOOL_READ = "tool_read"      # 只读工具直出（订单/物流/售后单）
    TOOL_WRITE = "tool_write"    # 写操作（需确认 + 幂等 + 权限）
    POLICY = "policy"            # 政策检索 + 规则引擎判定
    SMALLTALK = "smalltalk"      # 闲聊/兜底话术
    ESCALATE = "escalate"        # 直接转人工


class Decision(str, Enum):
    ANSWER = "answer"
    CLARIFY = "clarify"
    ESCALATE = "escalate"


class PolicyStatus(str, Enum):
    APPLICABLE = "applicable"          # 条件全部满足
    NOT_APPLICABLE = "not_applicable"  # 有明确不满足的条件
    UNKNOWN = "unknown"                # 缺字段，需要追问


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Turn:
    role: str
    content: str
    ts: float = field(default_factory=time.time)


@dataclass
class Slot:
    name: str
    value: Any
    confidence: float = 1.0
    source: str = "user"          # user | tool | coref
    turn_index: int = -1


@dataclass
class Evidence:
    """一条检索到的证据（条款级小块 + 父块上下文）。"""

    doc_id: str
    clause_id: str
    parent_id: str
    title: str
    text: str
    parent_text: str = ""
    source: str = "policy"        # policy | faq
    dense_score: float = 0.0
    sparse_score: float = 0.0
    fusion_score: float = 0.0
    rerank_score: float = 0.0
    rank: int = 0
    effective_from: str = ""
    effective_to: str = ""

    def as_citation(self) -> dict[str, Any]:
        return {
            "clause_id": self.clause_id,
            "title": self.title,
            "source": self.source,
            "snippet": self.text[:160],
            "effective_from": self.effective_from,
        }


@dataclass
class ToolResult:
    name: str
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    latency_ms: float = 0.0
    idempotency_key: str = ""
    tier: str = "L0"
    cached: bool = False


@dataclass
class Signals:
    """四信号置信度输入。每个信号都在 [0,1]。"""

    top1: float = 0.0             # 精排 Top-1 归一化分数
    margin: float = 0.0           # (top1-top2)/top1，区分唯一命中与歧义命中
    intent: float = 0.0           # 意图分类置信度
    tool_completeness: float = 0.0  # 工具返回必需字段完整度

    def as_vector(self) -> list[float]:
        return [self.top1, self.margin, self.intent, self.tool_completeness]


@dataclass
class ClaimCheck:
    claim: str
    supported: bool
    score: float
    evidence_ids: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class PolicyOutcome:
    clause_id: str
    title: str
    action: str
    status: PolicyStatus
    satisfied: list[str] = field(default_factory=list)
    violated: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    exception_id: str = ""
    explanation: str = ""
    effective_from: str = ""
    raw_conditions: list[dict[str, Any]] = field(default_factory=list)
    # 条款是否按商品类目/属性精确限定。用于决定哪条结论优先：
    # 「数码拆封不可退」（限定 digital，判 not_applicable）应该压过
    # 「发票 30 天内可申请」（不限类目，判 applicable），否则短问题会被
    # 通用条款抢答。
    scope_specific: bool = False
    # 用户问的是「规则是什么」而不是「我能不能办」：此时不能用资格结论
    # 去回答（会变成追问），也不能说「因此您可以…」，只做条款转述。
    info_only: bool = False


@dataclass
class SessionState:
    """一次会话的完整状态。图引擎的 state 就是这个对象。"""

    session_id: str
    user_id: str
    user_input: str = ""
    normalized_input: str = ""
    rewritten_query: str = ""

    turns: list[Turn] = field(default_factory=list)
    slots: dict[str, Slot] = field(default_factory=dict)
    summary: str = ""                    # 三层上下文的摘要层
    raw_window: int = 6                  # 原始层保留轮数

    intent: Intent = Intent.CHITCHAT
    intent_confidence: float = 0.0
    route: Route = Route.SMALLTALK

    query: str = ""
    evidence: list[Evidence] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    policy: Optional[PolicyOutcome] = None
    escalation_reason: str = ""

    signals: Signals = field(default_factory=Signals)
    confidence: float = 0.0
    calibrated: float = 0.0
    threshold: float = 0.0

    claims: list[ClaimCheck] = field(default_factory=list)
    numeric_violations: list[str] = field(default_factory=list)

    decision: Decision = Decision.ESCALATE
    reply: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    clarify_fields: list[str] = field(default_factory=list)
    clarify_question: str = ""

    tier: str = "L2"
    cache_hit: bool = False
    retries: int = 0
    cost_usd: float = 0.0
    tokens: int = 0
    latency_ms: float = 0.0
    trace_id: str = ""
    errors: list[str] = field(default_factory=list)
    audit: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    # ---- 便捷方法 ----
    def slot_value(self, name: str) -> Any:
        s = self.slots.get(name)
        return s.value if s else None

    def set_slot(self, name: str, value: Any, confidence: float = 1.0, source: str = "user") -> None:
        self.slots[name] = Slot(
            name=name, value=value, confidence=confidence, source=source,
            turn_index=len(self.turns),
        )

    def missing_slots(self) -> list[str]:
        return [n for n, s in self.slots.items() if s.value in (None, "", [])]

    def recent_turns(self) -> list[Turn]:
        return self.turns[-self.raw_window:]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["intent"] = self.intent.value
        d["route"] = self.route.value
        d["decision"] = self.decision.value
        if self.policy:
            d["policy"]["status"] = self.policy.status.value
        return d
