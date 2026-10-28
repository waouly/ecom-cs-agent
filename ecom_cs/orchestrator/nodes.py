"""编排节点：一轮会话被拆成 guard -> 理解 -> 取数 -> 检索 -> 判定 -> 决策 -> 生成 -> 核验。

几个刻意的顺序选择：
- **先取数后检索**：订单事实（是否签收、几天、拆没拆）决定该检索哪类政策，
  反过来先检索会拿到一堆无关条款，浪费 rerank 预算。
- **先判定后生成**：政策结论由规则引擎给出，模型只负责「把这句结论说人话」。
  生成在判定之后，模型就没有机会「自由发挥」出一个错误结论。
- **生成后再核验**：claim 级核验 + 数字守卫 + 合规检查，任何一项不过就降级转人工。
"""

from __future__ import annotations

import re
import time
import uuid
from datetime import date
from typing import Any, Optional

from ..cost.pricing import estimate_cost
from ..kb.facts import build_facts
from ..llm.tasks import ACTION_LABEL, HeuristicAnswerer
from ..nlu.slots import missing_fields, question_for, slot_value
from ..types import (
    Decision,
    Intent,
    PolicyStatus,
    Route,
    Slot,
    ToolResult,
    Turn,
)
from ..utils.text import char_jaccard, normalize
from ..verify.claims import ClaimVerifier
from ..tools.base import ToolContext
from ..verify.numeric import check_numbers, numeric_context
from ..verify.safety import check_input, check_output
from .context import TurnContext


def _tool_ctx(state, ctx: TurnContext) -> ToolContext:
    # 工具层只认 SessionState（要读 user_id / slots），编排层的 TurnContext 不能直接传进去
    return ToolContext(session=state, repo=ctx.repo, today=ctx.today, settings=ctx.settings)


def _flatten_values(obj: Any) -> str:
    """把工具返回的嵌套 dict/list 摊平成一段文本，供 claim 核验当「事实来源」。"""
    parts: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if value in (None, "", [], {}):
                continue
            parts.append(_flatten_values(value))
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            parts.append(_flatten_values(value))
    elif obj is not None:
        parts.append(str(obj))
    return " ".join(p for p in parts if p)


def _flat_scalars(obj: Any) -> list[str]:
    out: list[str] = []
    if isinstance(obj, dict):
        for value in obj.values():
            out.extend(_flat_scalars(value))
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            out.extend(_flat_scalars(value))
    elif obj is not None and not isinstance(obj, bool):
        out.append(str(obj))
    return out


CONFIRM_WORDS = ("确认", "确定", "是的", "是", "好的", "好", "可以", "同意", "提交", "没问题", "对")
DENY_WORDS = ("取消", "算了", "不用", "不要", "别改", "不改了", "否")

ROUTE_BY_INTENT: dict[Intent, Route] = {
    Intent.ORDER_QUERY: Route.TOOL_READ,
    Intent.LOGISTICS: Route.TOOL_READ,
    Intent.AFTERSALE: Route.TOOL_WRITE,
    Intent.REFUND: Route.TOOL_WRITE,
    Intent.POLICY_QA: Route.POLICY,
    Intent.COMPLAINT: Route.ESCALATE,
    Intent.CHITCHAT: Route.SMALLTALK,
    Intent.INJECTION: Route.ESCALATE,
    Intent.OUT_OF_SCOPE: Route.ESCALATE,
}

ESCALATE_TEXT = {    "injection": "抱歉，我只能协助处理订单、退款、售后和物流相关的问题。如果您有其他需求，可以回复「转人工」。",
    "complaint": "非常抱歉给您带来了不好的体验。我这边先为您转接人工客服，会有专员跟进处理。",
    "low_confidence": "关于这个问题，为了避免给您错误信息，我先为您转接人工客服（已记录工单），稍后会有专员跟进。",
    "not_found": "抱歉，我在知识库里没有找到能确定回答这个问题的依据，已为您转接人工客服，避免给您错误的信息。",
    "verify_failed": "这条回复里有无法核实的信息，为了稳妥起见，已为您转交人工客服确认。",
}


# ----------------------------------------------------------------------
# 节点实现
# ----------------------------------------------------------------------
def n_guard(state, ctx: TurnContext) -> str:
    """入口安全检查：批量用户数据请求直接拦截。"""
    report = check_input(state.user_input)
    if report.blocked:
        state.route = Route.ESCALATE
        state.escalation_reason = "批量数据请求被拦截"
        state.reply = ESCALATE_TEXT["injection"]
        state.decision = Decision.ESCALATE
        return "finalize"
    state.user_input = report.redacted_reply
    state.normalized_input = normalize(state.user_input)
    return "confirm"


def n_confirm(state, ctx: TurnContext) -> str:
    """处理上一轮的待确认写操作。

    写操作分两步：生成预览 -> 用户确认。这一步就是「确认」的回执。
    用户说别的，就当作普通输入继续走（不能因为挂着待确认就把后续问题吞掉）。
    """
    pending = state.meta.pop("pending_action", None)
    if not pending:
        return "understand"

    text = normalize(state.user_input)
    if any(w in text for w in DENY_WORDS):
        state.reply = "好的，操作已取消，没有做任何修改。"
        state.decision = Decision.ANSWER
        state.tier = "L0"
        return "finalize"

    if any(w in text for w in CONFIRM_WORDS):
        args = {**pending.get("args", {}), "confirm": True}
        result = ctx.deps.registry.execute(pending["tool"], args, _tool_ctx(state, ctx))
        state.tool_results = [result]
        state.tier = "L0"
        if result.ok:
            state.reply = str(result.data.get("message") or "操作已完成。")
            state.decision = Decision.ANSWER
            state.meta["write_action"] = {"tool": pending["tool"], "ok": True,
                                          "idempotency_key": result.idempotency_key,
                                          "cached": result.cached}
        else:
            state.reply = (f"操作失败了：{result.data.get('message', result.error)}。"
                           "已记录该问题，您也可以回复「转人工」让客服协助。")
            state.decision = Decision.ESCALATE
            state.escalation_reason = f"写操作失败:{result.error}"
        return "finalize"

    # 不是确认/取消，恢复 pending 并继续理解
    state.meta["pending_action"] = pending
    return "understand"


def n_understand(state, ctx: TurnContext) -> str:
    from ..retrieval.rewrite import rewrite_query

    state.turns.append(Turn(role="user", content=state.user_input))

    order_no = slot_value(state.slots, "order_no")
    rw = rewrite_query(state.normalized_input, order_no=order_no)
    state.rewritten_query = rw.rewritten
    state.query = rw.rewritten or state.normalized_input
    state.meta["rewrite"] = {"aliases": rw.aliases_hit, "coref": rw.coref_replaced}

    result = ctx.deps.classifier.classify(state.normalized_input)
    state.intent = result.intent
    state.intent_confidence = result.confidence
    state.meta["intent"] = {"method": result.method, "matched": result.matched[:6],
                            "injection": result.injection_reason}

    # 带了订单号却落到「超范围」的，多半是「O10001 现在什么状态」这种短句——
    # 词典没覆盖到「状态」这类词，但订单号本身就是强信号，兜底成订单查询。
    has_order = bool(re.search(r"[Oo]\s?[-_]?\s?\d{5,}", state.normalized_input))
    if has_order and state.intent in (Intent.OUT_OF_SCOPE, Intent.CHITCHAT):
        state.intent = Intent.ORDER_QUERY
        state.intent_confidence = max(state.intent_confidence, 0.55)
        state.meta["intent"]["fallback"] = "order_number_present"

    # 「这个订单能改地址吗」问的是政策，「帮我改成 XX」才是办事。
    # 词典里「改地址」权重更高，但句子里有「订单」时会被订单查询词压过去，
    # 这里按「有无明确的处置动词」再校准一次。
    if re.search(r"改.{0,2}地址|修改.{0,2}地址|换.{0,2}地址", state.normalized_input) and not re.search(
        r"帮我|我要|请帮|麻烦|改成|改到|改为|换成|换到", state.normalized_input
    ):
        state.intent = Intent.POLICY_QA
        state.intent_confidence = max(state.intent_confidence, 0.6)
        state.meta["intent"]["fallback"] = "address_policy"

    new_slots = ctx.deps.slots.extract(state.normalized_input, turn_index=len(state.turns) - 1)
    for name, slot in new_slots.items():
        old = state.slots.get(name)
        if old is None or slot.confidence >= old.confidence:
            state.slots[name] = slot

    state.route = ROUTE_BY_INTENT.get(state.intent, Route.POLICY)
    # 槽位在本轮可能刚被抽出来（上面合并过），所以要在这里重新读一次
    order_no = slot_value(state.slots, "order_no")
    # 有订单号且问题涉及「这单能不能办」时，必须先取订单事实再判定；
    # 反过来，没有订单号的退款问题多半是在问时效/规则，走政策问答而不是追问订单。
    if order_no and state.intent in (Intent.POLICY_QA, Intent.AFTERSALE, Intent.REFUND):
        state.route = Route.TOOL_WRITE
    elif not order_no and state.intent == Intent.REFUND:
        state.route = Route.POLICY
    if result.injection_reason:
        state.route = Route.ESCALATE
        state.escalation_reason = f"检测到注入意图：{result.injection_reason}"

    # 写操作（改地址）在这里就进「预览→确认」流程，不进政策判定：
    # 用户要的是「把这单的地址改掉」，不是听一遍政策解释。
    if state.route == Route.TOOL_WRITE and state.intent == Intent.AFTERSALE and order_no:
        address = slot_value(state.slots, "new_address")
        if address and re.search(r"地址|寄到|送到", state.normalized_input):
            args = {"order_no": order_no, "new_address": address}
            res = ctx.deps.registry.execute("change_address", {**args, "confirm": False}, _tool_ctx(state, ctx))
            state.tool_results = [res]
            state.decision = Decision.ANSWER
            state.tier = "L0"
            # 这一分支的回复是「前置条件校验 + 模板」直出的，没有检索也没调模型，
            # 统计置信度不适用；给一个固定值只是为了让日志里不出现 0.00 的假象。
            state.confidence = 0.9
            state.calibrated = 0.9
            if res.ok and res.data.get("needs_confirm"):
                state.meta["pending_action"] = {"tool": "change_address", "args": args}
                state.reply = (
                    f"即将把订单 {order_no} 的收货地址改为「{address}」，"
                    "请回复「确认」执行，或回复「取消」。"
                )
            else:
                state.reply = str(res.data.get("message") or res.error or "地址修改请求未受理。")
            return "finalize"
    return "cache"


def n_cache(state, ctx: TurnContext) -> str:
    """语义缓存：只对「与用户身份无关」的政策问答生效。"""
    if not (ctx.settings.cache_enabled and ctx.settings.enable_cache):
        return "dispatch"
    if state.route != Route.POLICY:
        return "dispatch"
    if slot_value(state.slots, "order_no"):
        return "dispatch"   # 带订单号的回答是个性化的，绝不能命中缓存

    hit = ctx.deps.cache.get(state.query, "policy")
    if hit is None:
        return "dispatch"
    state.reply = hit.reply
    state.citations = hit.citations
    state.cache_hit = True
    state.decision = Decision.ANSWER
    state.tier = "L0"
    state.meta["cache"] = {"score": round(hit.score, 4), "exact": hit.meta.get("exact", False)}
    return "finalize"


def n_dispatch(state, ctx: TurnContext) -> str:
    return "__router__"


def route_router(state) -> str:
    if state.route == Route.ESCALATE:
        return "escalate"
    if state.route == Route.SMALLTALK:
        return "smalltalk"
    if state.route in (Route.TOOL_READ, Route.TOOL_WRITE):
        return "tools"
    return "retrieve"


def n_tools(state, ctx: TurnContext) -> str:
    order_no = slot_value(state.slots, "order_no")
    if not order_no:
        state.meta["need_order"] = True
        return "retrieve"

    result = ctx.deps.registry.execute("lookup_order", {"order_no": order_no}, _tool_ctx(state, ctx))
    state.tool_results.append(result)
    if not result.ok:
        state.meta["tool_error"] = result.error
        return "retrieve"

    order = result.data
    state.meta["order"] = order
    # 工具返回的事实写入槽位，供规则引擎求值
    for field, value in (
        ("category", order.get("category")),
        ("days_since_sign", None),
        ("opened", order.get("opened")),
        ("is_custom", order.get("is_custom")),
        ("shipped", bool(order.get("shipped_at"))),
        ("signed", bool(order.get("signed_at"))),
    ):
        if value is not None:
            state.slots[field] = Slot(name=field, value=value, confidence=1.0, source="tool")

    if state.intent == Intent.LOGISTICS:
        state.tool_results.append(
            ctx.deps.registry.execute("lookup_logistics", {"order_no": order_no}, _tool_ctx(state, ctx))
        )
    if state.intent in (Intent.AFTERSALE, Intent.REFUND):
        state.tool_results.append(
            ctx.deps.registry.execute("lookup_aftersale", {"order_no": order_no}, _tool_ctx(state, ctx))
        )

    known = sum(1 for k in ("category", "shipped", "signed", "opened", "is_custom")
                if k in state.slots)
    state.meta["tool_completeness"] = known / 5.0
    return "retrieve"


def n_retrieve(state, ctx: TurnContext) -> str:
    order_no = slot_value(state.slots, "order_no")
    result = ctx.deps.retrieval.search(
        state.query,
        top_k=ctx.settings.top_k_final,
        order_no=order_no,
        today=ctx.today,
    )
    state.evidence = result.evidence
    state.meta["retrieval"] = {
        "n": len(result.evidence),
        "latency_ms": result.latency_ms,
        "rewritten": result.rewritten,
        "aliases": result.aliases_hit,
    }
    return "evaluate"


# 同一个订单会同时命中多条条款（退货、发票、会员权益…），光看「哪条更具体」
# 仍会答错题。这里用问题的主题词把候选条款收窄到「用户实际在问的那一类」。
# 主题词是显式规则，好处是可解释、可回归；真实系统里可以换成意图-条款的
# 检索打分，但必须先有这层「不许答跑题」的硬约束。
TOPIC_RULES: list[tuple[str, tuple[str, ...]]] = [
    (r"发票", ("发票",)),
    (r"价保|价格保护|补差价|差价|买贵", ("价保", "价格保护", "差价")),
    (r"会员|金卡|钻石|权益|特权", ("会员", "权益")),
    (r"运费|邮费", ("运费",)),
    (r"质量|坏了|坏的|坏|破损|脱线|瑕疵|故障", ("质量", "保修", "换货")),
    (r"地址", ("地址",)),
    (r"退款|退钱|到账|退到哪", ("退款", "到账")),
    (r"物流|快递|派送|运单|包裹|到哪了|到哪儿|到哪了|什么时候到", ("物流", "发货", "派送")),
]


def _topic_filter(outcomes, text: str, engine) -> list:
    for pattern, kws in TOPIC_RULES:
        if not re.search(pattern, text):
            continue
        scored = []
        for o in outcomes:
            clause = engine.by_id(o.clause_id)
            haystack = o.title + " " + (" ".join(clause.keywords) if clause else "")
            hit = max((len(k) for k in kws if k in haystack), default=0)
            if hit:
                scored.append((hit, o))
        if scored:
            # 命中词越长越具体：「价格保护」比同名的「价保」更能代表用户问的那件事。
            best = max(h for h, _ in scored)
            return [o for h, o in scored if h == best]
        # 主题已经明确、但这一桶里没有对得上的条款：宁可空着，
        # 也不要把不相干的条款（比如「发货时效」）拿上去答非所问。
        return []
    return list(outcomes)


# 用户问「能不能退」时，只有退款/拒绝类条款能给出这个答案；
# 保修条款（action=repair）即使状态是 unknown 也不该抢答。
REFUND_ACTIONS = {"refund_full", "refund_partial", "refund_shipping", "reject", "exchange", "special_handling"}


def _action_filter(outcomes, text: str) -> list:
    if "退" in text and not re.search(r"换|修|保修", text):
        picked = [o for o in outcomes if o.action in REFUND_ACTIONS]
        return picked or list(outcomes)
    if re.search(r"换货|换一个|换个", text):
        picked = [o for o in outcomes if o.action == "exchange"]
        return picked or list(outcomes)
    if re.search(r"保修|维修|修一下", text):
        picked = [o for o in outcomes if o.action in ("repair", "exchange")]
        return picked or list(outcomes)
    return list(outcomes)


def n_evaluate(state, ctx: TurnContext) -> str:
    from ..verify.confidence import ConfidenceScorer

    order = state.meta.get("order")
    user = ctx.repo.get_user(state.user_id) or {}
    facts = build_facts(order, user, state.slots, ctx.today)
    state.meta["facts"] = facts

    # 资格判定不能靠「检索到了哪几条」：查到订单后类目已知，就应该把该类目的
    # 条款全量评估一遍，否则「O10004 能退吗」这种短问题会漏掉定制商品条款。
    # 只有类目未知（还没查到订单）时，才退化为「只评估检索命中的条款」。
    # 类目已知（查到了订单）时全量评估该类目条款，避免短问题漏掉定制/质量条款；
    # 没有订单时只能评估检索命中的条款——但这里不接受「检索没命中就答不出来」，
    # 所以检索层通过别名把「金卡会员」「买贵了」这类口语映射到条款关键词。
    clause_ids = None if facts.get("category") else ([e.clause_id for e in state.evidence] or None)
    bundle = ctx.deps.engine.evaluate(facts, clause_ids=clause_ids)

    # 用主题词收窄候选：问发票就别拿退货条款来抢答。
    elig = _topic_filter(bundle.eligibility, state.normalized_input, ctx.deps.engine)
    infos = _topic_filter(bundle.info, state.normalized_input, ctx.deps.engine)
    elig = _action_filter(elig, state.normalized_input)
    infos = _action_filter(infos, state.normalized_input)
    from ..kb.policy_engine import STATUS_RANK

    # 「我要退」「帮我改地址」是办理请求，必须先拿订单号再下结论；
    # 「买贵了能补差价吗」是规则咨询，没有订单也应该直接讲规则。
    request = bool(re.search(r"帮我|我想|我要|请帮|麻烦|申请|开个|开一张|改成|修改|换绑", state.normalized_input))
    timing = any(
        k in state.normalized_input for k in ("多久", "到账", "几天", "什么时候", "时间", "进度", "运费", "谁出")
    )
    eligibility_asked = bool(re.search(r"能|可以|支持|吗|条件|符合|行不行|可不可以|够不", state.normalized_input))
    info_style = (not request) and (timing or not eligibility_asked)

    if elig and not info_style:
        state.policy = max(elig, key=lambda o: (o.scope_specific, STATUS_RANK[o.status]))
    elif not elig and infos and not info_style:
        usable = [o for o in infos if o.status == PolicyStatus.APPLICABLE]
        state.policy = usable[0] if usable else None
    else:
        # 信息型回答：在「信息条款（APPLICABLE）」与「资格条款但缺信息」里，
        # 选与问题字面最贴的那条，只转述规则、不追问也不下资格结论。
        pool = [o for o in infos if o.status == PolicyStatus.APPLICABLE]
        pool += [o for o in elig if o.status == PolicyStatus.UNKNOWN]
        if pool:
            best = max(pool, key=lambda o: (char_jaccard(state.normalized_input, o.title), o.scope_specific))
            best.info_only = True
            state.policy = best
        elif elig:
            state.policy = max(elig, key=lambda o: (o.scope_specific, STATUS_RANK[o.status]))

    # 没给订单号、也没在说「这个订单」，那就是在问规则本身：
    # 追问「请提供订单号」之前先把规则讲清楚，否则等于什么也没答。
    deictic = bool(re.search(r"这个订单|这单|该订单|我的订单|订单号", state.normalized_input))
    if (
        state.policy is not None
        and state.policy.status == PolicyStatus.UNKNOWN
        and not slot_value(state.slots, "order_no")
        and not request
        and not deictic
    ):
        state.policy.info_only = True
    state.meta["policy_bundle"] = {
        "n_eligibility": len(bundle.eligibility),
        "n_info": len(bundle.info),
        "needs_clarify": bundle.needs_clarify,
    }

    scorer = ConfidenceScorer()
    state.signals = scorer.signals(state)
    raw = scorer.fuse(
        state.signals,
        has_evidence=bool(state.evidence),
        policy_status=state.policy.status if state.policy else None,
    )
    state.confidence = round(raw, 4)

    calibration = ctx.deps.calibration
    if calibration.selective.n >= 5:
        calibrated, accepted = calibration.route(raw)
        state.threshold = calibration.selective.threshold
    else:
        calibrated = raw
        state.threshold = ctx.settings.default_threshold
        accepted = raw >= state.threshold
    state.calibrated = round(calibrated, 4)

    # ---- 决策：answer / clarify / escalate ----
    has_tools = bool(state.tool_results) and all(t.ok for t in state.tool_results)

    if state.meta.get("need_order"):
        allowed = missing_fields(
            list(state.policy.missing) if state.policy else [],
            state.slots,
            True,
        )
        state.clarify_fields = allowed or ["order_no"]
        state.decision = Decision.CLARIFY
        return "__router__"

    if state.meta.get("tool_error") == "order_not_found":
        state.decision = Decision.ESCALATE
        state.escalation_reason = "订单不存在或不属于当前用户"
        return "__router__"

    if state.route == Route.ESCALATE:
        state.decision = Decision.ESCALATE
        state.escalation_reason = state.escalation_reason or "意图需要人工处理"
        return "__router__"

    # 只读事实类问题（我的订单到哪了）：工具返回的就是事实，不需要政策结论
    if state.route == Route.TOOL_READ and has_tools:
        state.decision = Decision.ANSWER
        return "__router__"

    policy = state.policy
    if policy is not None:
        # info_only：用户问的是「规则是什么」，只转述条款，不追问也不下结论。
        if policy.info_only:
            state.decision = Decision.ANSWER
            return "__router__"
        if policy.status in (PolicyStatus.APPLICABLE, PolicyStatus.NOT_APPLICABLE):
            # applicable：按条款办；not_applicable：明确告知不符合条件（也是有效回答）
            state.decision = Decision.ANSWER
            return "__router__"
        if policy.status == PolicyStatus.UNKNOWN and policy.missing:
            fields = missing_fields(policy.missing, state.slots, False)
            if fields:
                state.clarify_fields = fields
                state.decision = Decision.CLARIFY
                return "__router__"

    if not state.evidence and not has_tools:
        state.decision = Decision.ESCALATE
        state.escalation_reason = "no_evidence"
        return "__router__"

    # 没有确定性结论、只剩「检索到的参考信息」时，才用校准阈值兜底
    if not accepted:
        state.decision = Decision.ESCALATE
        state.escalation_reason = f"置信度不足({state.calibrated:.2f} < {state.threshold:.2f})"
        return "__router__"

    state.decision = Decision.ANSWER
    return "__router__"


def decision_router(state) -> str:
    return {
        Decision.CLARIFY: "clarify",
        Decision.ESCALATE: "escalate",
        Decision.ANSWER: "generate",
    }.get(state.decision, "escalate")


def n_smalltalk(state, ctx: TurnContext) -> str:
    state.decision = Decision.ANSWER
    state.tier = "L1"
    state.reply = ctx.deps.answerer.smalltalk(state.intent)
    return "finalize"


def n_clarify(state, ctx: TurnContext) -> str:
    questions = [question_for(f).rstrip("？?") for f in state.clarify_fields if question_for(f)]
    state.clarify_question = "；".join(questions)
    # 追问的引导语要跟问题类型匹配：发票/改手机号这类「办理」问题
    # 不能说成「要判断是否符合退换条件」。
    p = state.policy
    if p is not None and p.status == PolicyStatus.UNKNOWN and (
        p.action in ("special_handling", "request_evidence") or "退" in (p.title or "")
    ):
        lead = "要判断是否符合退换条件，还需要确认："
    elif p is not None and p.status == PolicyStatus.UNKNOWN:
        lead = "要确认能否为您办理，还需要确认："
    else:
        lead = "为了帮您处理，还需要确认："
    state.reply = f"{lead}{state.clarify_question}？"
    state.decision = Decision.CLARIFY
    return "finalize"


def n_escalate(state, ctx: TurnContext) -> str:
    reason = state.escalation_reason or ""
    if state.meta.get("tool_error") == "order_not_found":
        text = (f"没有查到订单 {slot_value(state.slots, 'order_no')}，请确认订单号是否正确；"
                "也可以回复「转人工」让客服帮您查询。")
    elif "注入" in reason:
        text = ESCALATE_TEXT["injection"]
    elif state.intent == Intent.COMPLAINT:
        text = ESCALATE_TEXT["complaint"]
    elif reason == "no_evidence":
        text = ESCALATE_TEXT["not_found"]
    else:
        text = ESCALATE_TEXT["low_confidence"]
    state.reply = text
    state.decision = Decision.ESCALATE
    ctx.repo.add_audit(
        actor=state.user_id, action="escalate", target=state.session_id,
        payload={"reason": reason, "intent": state.intent.value,
                 "confidence": state.calibrated},
        result="ok",
    )
    return "finalize"


def _l0_reply(state, ctx: TurnContext) -> str:
    """L0：工具事实直接拼装，不经模型。正确率最高、成本最低。"""
    parts: list[str] = []
    for tr in state.tool_results:
        if not tr.ok:
            continue
        d = tr.data
        if tr.name == "lookup_order":
            line = f"订单 {d.get('order_no')} 当前状态：{d.get('status_label')}"
            if d.get("signed_at"):
                line += f"，签收时间 {d.get('signed_at')}"
            parts.append(line + "。")
        elif tr.name == "lookup_logistics":
            if d.get("has_logistics"):
                carrier = d.get("carrier") or "承运商"
                status = d.get("status_label") or d.get("status") or ""
                desc = d.get("description") or ""
                when = d.get("last_update") or ""
                line = f"物流：{carrier} 当前{status}"
                if desc:
                    line += f"，最新进展：{desc}"
                if when:
                    line += f"（更新于 {when}）"
                parts.append(line + "。")
            else:
                parts.append("该订单暂无物流信息，可能尚未发货。")
        elif tr.name == "lookup_aftersale":
            if d.get("items"):
                lines = [f"{it.get('type_label')}：{it.get('status_label')}"
                         f"（单号 {it.get('aftersale_no')}）" for it in d["items"]]
                parts.append("售后进度：" + "；".join(lines) + "。")
    return "\n".join(parts) if parts else ""


def n_generate(state, ctx: TurnContext) -> str:
    from ..cost.cascade import TierDecision

    if ctx.settings.enable_cascade:
        tier = ctx.deps.cascade.decide(state)
    else:
        tier = TierDecision("L2", "级联已关闭")
    state.tier = tier.tier
    state.meta["tier_reason"] = tier.reason

    # 只把「与这个问题相关」的工具结果交给渲染层：
    # 问退款时效时把订单金额/地址也念一遍，既啰嗦又容易让用户以为在核对账户信息。
    tools_for_answer = state.tool_results
    if state.intent == Intent.REFUND and any(
        k in state.normalized_input for k in ("多久", "到账", "几天", "什么时候", "时间", "进度")
    ):
        tools_for_answer = [t for t in state.tool_results if t.name == "lookup_aftersale"]

    if tier.tier == "L0":
        reply = _l0_reply(state, ctx)
        if not reply:
            state.tier = "L1"
            reply = ctx.deps.answerer.generate(
                question=state.query, evidence=state.evidence, policy=state.policy,
                tool_results=tools_for_answer, slots=state.slots, summary=state.summary,
            )
    else:
        reply = ctx.deps.answerer.generate(
            question=state.query, evidence=state.evidence, policy=state.policy,
            tool_results=tools_for_answer, slots=state.slots, summary=state.summary,
        )
    state.reply = (reply or "").strip()
    state.citations = [
        {"clause_id": e.clause_id, "title": e.title, "doc_id": e.doc_id,
         "source": e.source, "snippet": (e.text or "")[:120]}
        for e in state.evidence[:5]
    ]
    return "verify"


def n_verify(state, ctx: TurnContext) -> str:
    if not state.reply:
        return "finalize"

    cfg = ctx.settings
    if cfg.enable_safety:
        report = check_output(state.reply)
        state.reply = report.redacted_reply
        state.meta["safety"] = report.to_dict()
        if report.blocked:
            state.decision = Decision.ESCALATE
            state.escalation_reason = "输出触发合规拦截"
            state.reply = ESCALATE_TEXT["verify_failed"]
            return "finalize"

    if cfg.enable_claims:
        verifier: ClaimVerifier = ctx.deps.claims
        extras: list[str] = []
        trusted: list[str] = []
        if state.policy is not None:
            p = state.policy
            extras += [p.title, p.explanation]
            extras += list(p.satisfied) + list(p.violated)
            # 违反项是「商品未拆封，实际为 是」这种带实际值的文案，而句子常被切在
            # 逗号前，所以同时登记不带实际值的前半段，避免正确结论被判无依据。
            trusted += list(p.satisfied) + [v.split("，")[0] for v in p.violated] + list(p.violated)
            label = ACTION_LABEL.get(p.action, p.action)
            if p.status == PolicyStatus.APPLICABLE:
                # 真实模型很少照抄「因此您可以{label}」，它更可能写「可申请全额退款」。
                # 动作短语本身就来自引擎结论，所以单独登记一份，别让核验层把正确结论判成编造。
                trusted += [f"因此您可以{label}", label]
            if p.status == PolicyStatus.NOT_APPLICABLE:
                trusted += ["因此暂时无法办理", "很抱歉"]
                extras.append(HeuristicAnswerer._hint(p.violated))
        for tr in state.tool_results:
            if not tr.ok:
                continue
            extras.append(_flatten_values(tr.data))
            # 工具返回的短标量（状态文案、金额）本身就是库里的确定事实，
            # 当成 trusted 可以避免「订单 O10004 已签收」这类句子被误杀。
            for value in _flat_scalars(tr.data):
                if 2 <= len(value) <= 24:
                    trusted.append(value)
        for field, slot in state.slots.items():
            extras.append(f"{field} {slot.value}")
        result = verifier.verify(state.reply, state.evidence, extras=extras, trusted=trusted)
        state.claims = result.claims
        state.meta["verification"] = result.to_dict()

    if cfg.enable_numeric_guard:
        policy_texts = [e.parent_text or e.text for e in state.evidence]
        violations = check_numbers(state.reply, numeric_context(state) + policy_texts)
        if violations:
            state.numeric_violations = [str(v) for v in violations]
            state.decision = Decision.ESCALATE
            state.escalation_reason = "回复包含无法溯源的数字"
            state.reply = ESCALATE_TEXT["verify_failed"]
            return "finalize"

    if state.claims:
        failed = [c for c in state.claims if not c.supported]
        if len(failed) > max(1, len(state.claims) // 2):
            state.decision = Decision.ESCALATE
            state.escalation_reason = "答案中多条陈述缺乏证据支撑"
            state.reply = ESCALATE_TEXT["verify_failed"]
    return "finalize"


def n_finalize(state, ctx: TurnContext) -> str:
    if not state.reply:
        state.reply = ESCALATE_TEXT["low_confidence"]
        state.decision = Decision.ESCALATE
        state.escalation_reason = state.escalation_reason or "empty_reply"

    state.turns.append(Turn(role="assistant", content=state.reply))
    state.trace_id = state.trace_id or uuid.uuid4().hex[:12]

    # 三层上下文：超出窗口就把旧轮次压成摘要，保留最近 raw_window 条
    try:
        if len(state.turns) > state.raw_window * 2:
            older = state.turns[:-state.raw_window]
            state.summary = ctx.deps.answerer.summarize(older, state.summary)
            state.turns = state.turns[-state.raw_window:]
    except Exception as e:  # 摘要失败不能影响主流程
        state.errors.append(f"summarize: {e}")

    tokens_in = sum(len(e.text) for e in state.evidence) // 2 + len(state.query) // 2
    tokens_out = len(state.reply) // 2
    state.tokens = tokens_in + tokens_out
    state.cost_usd = round(estimate_cost(state.tier, tokens_in, tokens_out), 6)

    from ..cost.ledger import CostEntry

    ctx.deps.ledger.record(CostEntry(
        session_id=state.session_id, tier=state.tier,
        prompt_tokens=tokens_in, completion_tokens=tokens_out,
        usd=state.cost_usd, latency_ms=ctx.elapsed_ms(), cache_hit=state.cache_hit,
    ))

    if state.decision == Decision.ANSWER and not state.cache_hit and state.route == Route.POLICY:
        if not slot_value(state.slots, "order_no"):
            ctx.deps.cache.put(state.query, state.reply, "policy", state.citations,
                               {"tier": state.tier})

    # 知识闭环：没答好/没答上的问题记成缺口，供人工审核后回灌
    if state.decision != Decision.ANSWER:
        try:
            ctx.deps.updater.record_gap(
                query=state.user_input,
                intent=state.intent.value,
                confidence=state.calibrated,
                decision=state.decision.value,
                note=state.escalation_reason[:80],
            )
        except Exception as e:
            state.errors.append(f"record_gap: {e}")

    state.audit.append({
        "decision": state.decision.value,
        "route": state.route.value,
        "intent": state.intent.value,
        "tier": state.tier,
        "confidence": state.calibrated,
        "threshold": state.threshold,
        "cache_hit": state.cache_hit,
        "n_evidence": len(state.evidence),
    })
    state.latency_ms = round(ctx.elapsed_ms(), 2)
    return "__end__"
