"""任务层：把「生成回答 / 追问 / 摘要 / 评判」抽象成协议。

- ``HeuristicAnswerer`` 完全离线，零依赖，确定性输出，用于默认演示与单测。
- ``LLMAnswerer`` 用真实模型，prompt 里强制要求引用条款 ID。

整条链路只依赖 ``Answerer`` 协议，因此离线与在线可以无缝切换。
"""

from __future__ import annotations

import json
from typing import Any, Optional, Protocol

from ..config import Settings
from ..types import Decision, Evidence, Intent, PolicyOutcome, PolicyStatus, ToolResult, Turn
from ..utils.text import char_jaccard, clamp, normalize_light, truncate

# ---------------------------------------------------------------------------
# 文案映射
# ---------------------------------------------------------------------------

ACTION_LABEL = {
    "refund_full": "申请全额退款",
    "refund_partial": "申请部分退款",
    "refund_shipping": "由商家承担退回运费",
    "exchange": "申请换货",
    "repair": "申请维修",
    "reject": "无法办理该申请",
    "compensate": "获得相应补偿",
    "special_handling": "转人工特殊处理",
    "request_evidence": "补充凭证后重新申请",
}

TOOL_LABEL = {
    "query_order": "订单",
    "query_logistics": "物流",
    "query_aftersale": "售后单",
    "apply_refund": "退款申请",
    "change_address": "修改收货地址",
    "escalate_to_human": "人工客服",
}

FIELD_QUESTION = {
    "order_no": "请提供一下订单号（在「我的订单」里可以看到）",
    "days_since_sign": "订单大概是什么时候签收的",
    "opened": "商品目前是否已经拆封使用",
    "is_custom": "这件商品是否属于定制款",
    "category": "这是哪一类商品",
    "has_quality_issue": "商品是否存在质量问题",
    "has_evidence": "是否已拍摄商品问题照片或视频",
    "paid_at": "方便告知下单时间吗",
    "sign_date": "方便告知签收日期吗",
    "reason": "方便说一下具体原因吗",
}

FIELD_LABEL = {
    "days_since_sign": "签收天数",
    "opened": "是否拆封",
    "is_custom": "是否定制",
    "order_no": "订单号",
    "category": "商品类目",
    "has_quality_issue": "是否有质量问题",
    "has_evidence": "是否有凭证",
    "reason": "申请原因",
}


def condition_text(conds: list[dict[str, Any]]) -> str:
    if not conds:
        return ""
    return "，".join(str(c.get("label") or c.get("field")) for c in conds)


class Answerer(Protocol):
    name: str

    def generate(
        self,
        *,
        question: str,
        evidence: list[Evidence],
        policy: Optional[PolicyOutcome],
        tool_results: list[ToolResult],
        slots: Optional[dict[str, Any]] = None,
        summary: str = "",
    ) -> str: ...

    def clarify(self, fields: list[str]) -> str: ...

    def summarize(self, turns: list[Turn], prev: str = "") -> str: ...

    def judge(self, question: str, answer: str, reference: str) -> float: ...

    def smalltalk(self, intent: Intent) -> str: ...


# ---------------------------------------------------------------------------
# 离线启发式实现
# ---------------------------------------------------------------------------


class HeuristicAnswerer:
    """把结构化结果（政策判定 / 工具返回）渲染成自然语言。

    它不「创造」，只做「忠实渲染」——这正好是客服场景想要的：结论来自规则
    引擎和工具，语言模型只负责把它说人话。
    """

    name = "heuristic-v1"

    # -- 政策 --
    def _policy_block(self, p: PolicyOutcome) -> str:
        label = ACTION_LABEL.get(p.action, p.action)
        # 信息型条款（没有判定条件，只给出规则/时效）不能套「因此您可以…」的句式，
        # 否则「退款审核时效」会被渲染成「因此您可以申请全额退款」，语义完全错位。
        # info_only 是「用户在问规则本身」的情形：同样只转述条款，不下资格结论。
        if p.info_only or (p.status == PolicyStatus.APPLICABLE and not p.raw_conditions):
            body = truncate(p.explanation or p.title, 300)
            return f"根据《{p.title}》（条款 {p.clause_id}）：{body}"
        if p.status == PolicyStatus.APPLICABLE:
            cond = "，".join(p.satisfied) if p.satisfied else "相关条件均已满足"
            text = f"依据《{p.title}》（条款 {p.clause_id}），{cond}，因此您可以{label}。"
            if p.exception_id:
                text += f"（已按例外条款 {p.exception_id} 处理）"
            return text
        if p.status == PolicyStatus.NOT_APPLICABLE:
            vio = "；".join(p.violated) if p.violated else "不满足该政策条件"
            text = f"很抱歉，依据《{p.title}》（条款 {p.clause_id}），当前订单不满足「{vio}」，因此暂时无法办理。"
            hint = self._hint(p.violated)
            if hint:
                text += hint
            return text
        need = "、".join(FIELD_LABEL.get(f, f) for f in p.missing) or "相关信息"
        return f"根据《{p.title}》（条款 {p.clause_id}），还需要确认{need}才能给出结论。"

    @staticmethod
    def _hint(violated: list[str]) -> str:
        joined = " ".join(violated)
        if "拆封" in joined:
            return "如商品已拆封但存在质量问题，可走「质量问题退货」，请提供照片或视频凭证。"
        if "定制" in joined:
            return "定制类商品不支持无理由退货，但质量问题仍可在三包期内申请售后。"
        if "天" in joined or "期" in joined:
            return "已超出无理由退货期，如为质量问题可在三包期内申请售后；也可转人工做特殊处理。"
        if "类目" in joined or "不支持" in joined:
            return "该商品类目不在无理由退货范围内，如为质量问题请走质量售后流程。"
        return ""

    # -- 工具 --
    def _tool_block(self, r: ToolResult) -> str:
        d = r.data
        if not r.ok:
            label = TOOL_LABEL.get(r.name, r.name)
            return f"暂时无法获取{label}信息（{r.error or '服务异常'}），建议稍后重试，或转人工为您核实。"
        if r.name == "lookup_order":
            bits = [f"订单 {d.get('order_no', '-')} 当前状态：{d.get('status_label', d.get('status', '-'))}"]
            if d.get("signed_at"):
                bits.append(f"签收时间 {d.get('signed_at')}")
            if d.get("amount") is not None:
                bits.append(f"实付 ¥{d.get('amount')}")
            if d.get("address"):
                bits.append(f"收货地址 {d.get('address')}")
            return "，".join(bits) + "。"
        if r.name == "lookup_logistics":
            if not d.get("has_logistics", True):
                return "暂未查询到该订单的物流信息，可能还未发货。"
            carrier = d.get("carrier") or "承运商"
            status = d.get("status_label") or d.get("status") or "-"
            desc = d.get("description") or ""
            when = d.get("last_update") or ""
            text = f"{carrier} 当前状态：{status}"
            if desc:
                text += f"，最新进展：{desc}"
            if when:
                text += f"（更新于 {when}）"
            return text + "。"
        if r.name == "lookup_aftersale":
            items = d.get("items") or []
            if not items:
                return "该订单目前没有进行中的售后单。"
            lines = []
            for it in items[:3]:
                lines.append(
                    f"售后单 {it.get('aftersale_no', '-')}："
                    f"{it.get('type_label', it.get('type', '-'))}，"
                    f"状态 {it.get('status_label', it.get('status', '-'))}"
                )
            return "；".join(lines) + "。"
        if r.name == "change_address":
            if d.get("changed") is False:
                return "收货地址与原来一致，无需修改。"
            return f"收货地址已更新为：{d.get('address', '-')}。"
        if r.name == "create_return":
            return (
                f"已为您创建售后单 {d.get('aftersale_no', '-')}"
                f"（处理方式：{d.get('type_label', d.get('type', '-'))}），"
                "审核通过后会按原支付渠道退回。"
            )
        if r.name == "handoff_to_human":
            return f"已为您转接人工客服，工单号 {d.get('ticket_no', '-')}，请稍候。"
        return f"已获取{TOOL_LABEL.get(r.name, r.name)}信息。"

    def _faq_block(self, e: Evidence) -> str:
        body = e.parent_text or e.text
        return f"根据知识库《{truncate(e.title, 40)}》（{e.clause_id}）：{truncate(body, 260)}"

    # -- 主接口 --
    def generate(
        self,
        *,
        question: str,
        evidence: list[Evidence],
        policy: Optional[PolicyOutcome],
        tool_results: list[ToolResult],
        slots: Optional[dict[str, Any]] = None,
        summary: str = "",
    ) -> str:
        blocks: list[str] = []
        if policy is not None:
            blocks.append(self._policy_block(policy))
        for r in tool_results:
            blocks.append(self._tool_block(r))
        if not blocks and evidence:
            blocks.append(self._faq_block(evidence[0]))
        if not blocks:
            return "抱歉，我暂时没有查到与您问题相关的信息，建议转人工客服为您进一步核实。"
        return "\n".join(blocks)

    def clarify(self, fields: list[str]) -> str:
        qs = [FIELD_QUESTION.get(f, f"能否补充一下{FIELD_LABEL.get(f, f)}") for f in fields]
        if not qs:
            return "能否再具体描述一下您遇到的问题？"
        if len(qs) == 1:
            return f"为了准确判断，还需要确认一下：{qs[0]}？"
        body = "；".join(f"{i + 1}) {q}" for i, q in enumerate(qs))
        return f"为了准确判断，还需要确认两点：{body}？"

    def summarize(self, turns: list[Turn], prev: str = "") -> str:
        """规则摘要：抽订单号、诉求、已给结论。真实模型下替换为 LLM 摘要。"""
        import re

        order_nos: list[str] = []
        intent_words: list[str] = []
        answers: list[str] = []
        for t in turns:
            if t.role == "user":
                order_nos.extend(re.findall(r"(?:订单|order)\D{0,4}(\d{6,})", t.content))
                for kw in ("退货", "退款", "换货", "物流", "发货", "地址", "质量", "发票", "投诉"):
                    if kw in t.content and kw not in intent_words:
                        intent_words.append(kw)
            elif t.role == "assistant":
                answers.append(truncate(t.content, 60))
        bits: list[str] = []
        if prev:
            bits.append(prev)
        if order_nos:
            bits.append("涉及订单 " + "/".join(dict.fromkeys(order_nos)))
        if intent_words:
            bits.append("诉求关键词：" + "、".join(intent_words))
        if answers:
            bits.append("客服已答复：" + answers[-1])
        text = "；".join(bits) if bits else ""
        return truncate(text, 300)

    def judge(self, question: str, answer: str, reference: str) -> float:
        if not reference:
            return 0.5
        base = char_jaccard(answer, reference)
        bonus = 0.0
        for kw in ("可以申请", "无法办理", "不满足", "已为您创建", "全额退款", "不支持"):
            if kw in reference and kw in answer:
                bonus += 0.08
        return clamp(base * 1.15 + bonus)

    def smalltalk(self, intent: Intent) -> str:
        if intent == Intent.COMPLAINT:
            return "非常抱歉给您带来不好的体验，我已经记录您的问题，会优先转接人工客服跟进处理。"
        if intent == Intent.CHITCHAT:
            return "您好，我是电商售后客服助手，可以帮您查询订单、物流进度，或解答退换货政策。请问有什么可以帮您？"
        return "抱歉，这个问题超出了我的服务范围。我可以帮您处理订单、物流和售后相关问题，或为您转人工。"


# ---------------------------------------------------------------------------
# 真实模型实现
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """你是电商售后客服助手。必须严格遵守：
1. 只能依据【证据】和【结构化判定】作答，不得引入任何外部知识或臆测。
2. 引用条款时必须写明条款 ID，格式为（条款 XXX-XXX-000）。
3. 回答中出现的任何天数、金额、比例、时效，必须能在证据或工具结果中原样找到。
4. 若证据不足，直接回答「信息不足，建议转人工」，不要编造。
5. 中文作答，简洁，不超过 120 字。"""


class LLMAnswerer:
    name = "llm-v1"

    def __init__(self, client, fallback: Optional[HeuristicAnswerer] = None) -> None:
        self.client = client
        self.fallback = fallback or HeuristicAnswerer()
        self.total_cost_usd = 0.0
        self.total_tokens = 0

    @staticmethod
    def _render_context(
        evidence: list[Evidence],
        policy: Optional[PolicyOutcome],
        tool_results: list[ToolResult],
    ) -> str:
        lines: list[str] = []
        if policy is not None:
            lines.append("【结构化判定】")
            lines.append(
                json.dumps(
                    {
                        "clause_id": policy.clause_id,
                        "title": policy.title,
                        "status": policy.status.value,
                        "action": policy.action,
                        "satisfied": policy.satisfied,
                        "violated": policy.violated,
                        "missing": policy.missing,
                        "exception": policy.exception_id,
                    },
                    ensure_ascii=False,
                )
            )
        if tool_results:
            lines.append("【工具结果】")
            for r in tool_results:
                lines.append(json.dumps({"tool": r.name, "ok": r.ok, **r.data}, ensure_ascii=False))
        if evidence:
            lines.append("【证据】")
            for i, e in enumerate(evidence, 1):
                lines.append(f"[{i}] ({e.clause_id}) {e.title}: {truncate(e.parent_text or e.text, 240)}")
        return "\n".join(lines)

    def generate(
        self,
        *,
        question: str,
        evidence: list[Evidence],
        policy: Optional[PolicyOutcome],
        tool_results: list[ToolResult],
        slots: Optional[dict[str, Any]] = None,
        summary: str = "",
    ) -> str:
        ctx = self._render_context(evidence, policy, tool_results)
        user = f"【历史摘要】{summary or '无'}\n{ctx}\n\n【用户问题】{question}\n请作答。"
        try:
            return self.client.chat(
                [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": user}]
            ).strip()
        except Exception:  # noqa: BLE001
            return self.fallback.generate(
                question=question, evidence=evidence, policy=policy,
                tool_results=tool_results, slots=slots, summary=summary,
            )

    def clarify(self, fields: list[str]) -> str:
        return self.fallback.clarify(fields)

    def summarize(self, turns: list[Turn], prev: str = "") -> str:
        body = "\n".join(f"{t.role}: {truncate(t.content, 120)}" for t in turns)
        prompt = (
            f"已有摘要：{prev or '无'}\n新增对话：\n{body}\n\n"
            "请合并生成不超过 120 字的新摘要，必须保留订单号、商品、诉求、已给出的结论。只输出摘要。"
        )
        try:
            return self.client.complete(prompt).strip()
        except Exception:  # noqa: BLE001
            return self.fallback.summarize(turns, prev)

    def judge(self, question: str, answer: str, reference: str) -> float:
        prompt = (
            "你是严格的评测员。判断【回答】是否与【参考答案】在事实上等价。\n"
            f"问题：{question}\n参考答案：{reference}\n回答：{answer}\n"
            "只输出 0 到 1 之间的一个数字，表示事实一致程度。"
        )
        try:
            raw = self.client.complete(prompt).strip()
            import re

            m = re.search(r"\d+(?:\.\d+)?", raw)
            return clamp(float(m.group(0))) if m else self.fallback.judge(question, answer, reference)
        except Exception:  # noqa: BLE001
            return self.fallback.judge(question, answer, reference)

    def smalltalk(self, intent: Intent) -> str:
        return self.fallback.smalltalk(intent)


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------


def build_answerer(settings: Settings) -> Answerer:
    provider = (settings.llm_provider or "mock").lower()
    if provider in {"mock", "heuristic", "offline"}:
        return HeuristicAnswerer()
    if provider in {"openai", "openai-compat", "compat"}:
        from .openai_compat import OpenAICompatLLM

        client = OpenAICompatLLM(
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            temperature=settings.llm_temperature,
            timeout_s=settings.llm_timeout_s,
        )
        return LLMAnswerer(client)
    raise ValueError(f"未知 LLM provider: {settings.llm_provider}")
