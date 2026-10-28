"""槽位抽取：把用户口述变成结构化事实。

规则抽取在这里的定位不是「便宜的模型替代品」，而是**可信来源**：
订单号、金额、拆封与否这类信息一旦抽错，后面的规则判定会全错。
所以这里宁可抽不到（留空 -> 触发追问），也不猜——置信度低于阈值的抽取值
会带 ``confidence`` 标记，供编排层决定要不要二次确认。

否定处理是这类抽取最容易翻车的地方：「没拆封」里包含「拆」，
「不是质量问题」里包含「质量问题」。所以先匹配否定式，命中后直接返回 False。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from ..types import Intent, Slot
from ..utils.text import normalize

ORDER_RE = re.compile(r"[Oo]\s?[-_]?\s?(\d{5,})")
AMOUNT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:元|块钱|块|rmb)", re.IGNORECASE)
ADDRESS_RE = re.compile(
    # 地址正文用「非空白字符」截断，但门牌号常被空格分开（「麓山南路 2 号」），
    # 所以补一个可选的「数字 + 号/栋/室」尾巴，否则会把门牌号吃掉。
    r"(?:改成|改到|改为|换成|换到|寄到|送到|地址是)([^\s，。？！,?！]{4,60}(?:\s*\d+\s*[号栋室])?)"
)

NEG_OPENED = ("未拆封", "没拆封", "还没有拆", "还没拆", "没拆", "未拆", "没打开", "还没有打开",
              "没动过", "没使用", "没用过", "没穿", "没洗", "未激活", "没激活", "全新未拆")
POS_OPENED = ("拆封", "拆了", "拆开", "打开了", "用过了", "用过了", "穿过", "洗过", "激活了", "已拆")

NEG_QUALITY = ("不是质量问题", "没质量问题", "质量没问题", "没有问题", "没有破损", "没坏", "完好")
POS_QUALITY = ("质量问题", "坏了", "破损", "脱线", "开胶", "掉色", "漏水", "不能用", "故障",
               "瑕疵", "有毛病", "裂了", "变质", "腐烂", "缺斤少两", "少了", "有划痕")

POS_EVIDENCE = ("照片", "图片", "拍了", "视频", "截图", "凭证", "证据", "录了")

REASON_TABLE: dict[str, str] = {
    "不喜欢": "不喜欢",
    "不想要": "不想要",
    "不需要": "不想要",
    "尺码": "尺码不合适",
    "大小不合适": "尺码不合适",
    "买贵": "价格保护",
    "降价": "价格保护",
    "质量问题": "质量问题",
    "七天无理由": "七天无理由",
    "发错": "发错货",
    "少发": "少发漏发",
}

AFTERSALE_TYPE_TABLE: dict[str, str] = {
    "仅退款": "refund_only",
    "只退款": "refund_only",
    "退货退款": "return_refund",
    "退货": "return_refund",
    "换货": "exchange",
    "换一个": "exchange",
    "维修": "repair",
}

# 事实字段 -> 追问话术。规则引擎吐出 missing 字段后，这里直接变成用户能懂的问句。
FIELD_QUESTIONS: dict[str, str] = {
    "order_no": "请提供订单号（例如 O10001），我帮您查一下",
    "days_since_sign": "请问这笔订单是哪天签收的？",
    "opened": "商品是否已经拆封或使用过？",
    "is_custom": "这件商品是定制款（刻字/改尺寸）吗？",
    "has_quality_issue": "商品是否存在质量问题（如破损、脱线、无法使用）？",
    "has_evidence": "您是否已经拍摄了商品问题的照片或视频？",
    "shipped": "订单是否已经发货？",
    "signed": "订单是否已经签收？",
    "member_level": "请问您的会员等级是普通 / 金卡 / 钻石？",
    "category": "请问这件商品属于什么类目（服装 / 数码 / 家具 / 生鲜 / 内衣）？",
}

ORDER_REQUIRED_INTENTS = frozenset(
    {Intent.ORDER_QUERY, Intent.LOGISTICS, Intent.AFTERSALE, Intent.REFUND}
)


@dataclass
class SlotHit:
    name: str
    value: Any
    confidence: float = 1.0
    reason: str = ""


class SlotExtractor:
    def extract(self, text: str, turn_index: int = -1) -> dict[str, Slot]:
        norm = normalize(text)
        hits: list[SlotHit] = []

        m = ORDER_RE.search(norm)
        if m:
            hits.append(SlotHit("order_no", f"O{m.group(1)}", 0.99, "regex"))

        m = AMOUNT_RE.search(norm)
        if m:
            try:
                hits.append(SlotHit("amount_claimed", float(m.group(1)), 0.9, "regex"))
            except ValueError:
                pass

        m = ADDRESS_RE.search(norm)
        if m:
            address = m.group(1).strip("，。 ")
            # 去掉正文与门牌号之间多余的空格（「麓山南路 2 号」→「麓山南路 2 号」保留单空格）
            address = re.sub(r"\s+", " ", address)
            hits.append(SlotHit("new_address", address, 0.8, "regex"))

        opened = self._polarity(norm, NEG_OPENED, POS_OPENED)
        if opened is not None:
            hits.append(SlotHit("opened", opened, 0.9, "polarity"))

        quality = self._polarity(norm, NEG_QUALITY, POS_QUALITY)
        if quality is not None:
            hits.append(SlotHit("has_quality_issue", quality, 0.85, "polarity"))

        if any(w in norm for w in POS_EVIDENCE):
            hits.append(SlotHit("has_evidence", True, 0.8, "keyword"))

        for key, value in REASON_TABLE.items():
            if key in norm:
                hits.append(SlotHit("reason", value, 0.75, "keyword"))
                break

        for key, value in AFTERSALE_TYPE_TABLE.items():
            if key in norm:
                hits.append(SlotHit("aftersale_type", value, 0.8, "keyword"))
                break

        if "定制" in norm or "刻字" in norm:
            hits.append(SlotHit("is_custom", True, 0.85, "keyword"))

        if "拒收" in norm:
            hits.append(SlotHit("refused", True, 0.8, "keyword"))

        out: dict[str, Slot] = {}
        for h in hits:
            out[h.name] = Slot(name=h.name, value=h.value, confidence=h.confidence,
                               source="user", turn_index=turn_index)
        return out

    @staticmethod
    def _polarity(text: str, negatives: tuple[str, ...], positives: tuple[str, ...]) -> Optional[bool]:
        # 否定式优先：包含关系导致的误判几乎都出在「没拆封」这类说法上
        if any(w in text for w in negatives):
            return False
        if any(w in text for w in positives):
            return True
        return None


def merge_slots(existing: dict[str, Slot], new: dict[str, Slot]) -> dict[str, Slot]:
    """新槽位覆盖旧槽位，但置信度更低时保留旧的（避免一句口头补充把已验证事实冲掉）。"""
    merged = dict(existing)
    for name, slot in new.items():
        old = merged.get(name)
        if old is None or slot.confidence >= old.confidence:
            merged[name] = slot
    return merged


def slot_value(slots: dict[str, Slot], name: str, default: Any = None) -> Any:
    s = slots.get(name)
    return s.value if s is not None else default


def missing_fields(policy_missing: list[str], slots: dict[str, Slot],
                   need_order: bool) -> list[str]:
    """把规则引擎的 missing 字段和「缺订单号」合并成一次追问的字段列表。"""
    fields: list[str] = []
    if need_order and slot_value(slots, "order_no") is None:
        fields.append("order_no")
    for f in policy_missing:
        if f in {"days_since_sign", "opened", "is_custom", "category"} and slot_value(slots, "order_no") is None:
            # 没有订单号时，这些字段的根因是「没给订单」，问订单号就够了
            if "order_no" not in fields:
                fields.append("order_no")
            continue
        if f not in fields:
            fields.append(f)
    return fields[:2]  # 一次最多问两个，问多了用户会烦


def question_for(field: str) -> str:
    return FIELD_QUESTIONS.get(field, f"请补充：{field}")
