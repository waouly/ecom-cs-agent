"""查询改写：别名归一、指代消解、多意图拆分。

改写的收益在评测里是可见的：客服用户很少说规范术语（「zfb 退的钱到哪了」
「这个还能退不」）。不改写就召回不到「支付宝退款到账时间」「七天无理由退货」
这类条款，Recall@5 会掉一大截。

这里刻意不做「用大模型改写」——大模型改写会引入不可控的语义漂移，且每次
多一次调用。规则改写可解释、可测试、零成本，对高频客服场景足够。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ..utils.text import normalize

# 口语/缩写 -> 标准术语。写完每条都值得配一个单测，这是最容易回归的地方。
ALIAS_TABLE: dict[str, str] = {
    "zfb": "支付宝",
    "支付宝退款": "支付宝 退款 到账",
    "wx": "微信",
    "wxzf": "微信支付",
    "退款到哪了": "退款到账时间 进度",
    "退款到哪": "退款到账时间 进度",
    "退到哪": "退款退回 原支付渠道 到账",
    "退到哪里": "退款退回 原支付渠道 到账",
    "退到哪儿": "退款退回 原支付渠道 到账",
    "退的钱": "退款 原支付渠道",
    "钱什么时候到": "退款到账时间",
    "钱多久到": "退款到账时间",
    "无理由": "七天无理由退货",
    "七天无理由": "七天无理由退货",
    "7天无理由": "七天无理由退货",
    "能退吗": "可以退货吗 七天无理由退货",
    "能退不": "可以退货吗 七天无理由退货",
    "退不了": "退货 不支持",
    "换货": "质量问题换货",
    "运费谁出": "退货运费承担规则",
    "邮费谁出": "退货运费承担规则",
    "改地址": "修改收货地址",
    "地址改成": "修改收货地址",
    "地址改到": "修改收货地址",
    "地址改为": "修改收货地址",
    "改下地址": "修改收货地址",
    "开个发票": "电子发票申请",
    "开发票": "电子发票申请",
    "要发票": "电子发票申请",
    "发票怎么开": "电子发票申请",
    "会员权益": "会员权益 金卡",
    "金卡": "金卡会员 会员权益",
    "会员": "会员权益 金卡",
    "买贵了": "价格保护 补差价",
    "补差价": "价格保护 补差价",
    "地址填错": "修改收货地址",
    "快递": "物流",
    "包裹": "物流",
    "拆了": "已拆封",
    "打开了": "已拆封",
    "开票": "发票 申请",
    "便宜了": "降价 价格保护",
    "买贵了": "价格保护 退差价",
    "人工": "人工客服",
    "转人工": "人工客服 转人工",
    "定金": "预售",
    "秒杀": "活动价",
}

# 纯语气词，去掉能提升 BM25 的信噪比
FILLER_PATTERNS = [
    r"^(那个|就是|我想问一下|请问一下|问一下|你好|您好|麻烦问下)[,，。\s]*",
    r"[,，。\s]*(呀|啊|呢|吧|哦|嘛|哈)+$",
    r"(帮我|给我)(看看|查一下|查查)",
]

_SPLIT_RE = re.compile(r"(?:另外|还有|以及|另外想问|顺便问|再加一个问题)")
_PRONOUN_RE = re.compile(r"(这个订单|该订单|这个单|这单|它|这笔订单|这个商品|这件)")


@dataclass
class RewriteResult:
    original: str
    rewritten: str
    aliases_hit: list[str]
    coref_replaced: bool = False


def expand_aliases(text: str) -> tuple[str, list[str]]:
    lowered = text.lower()
    hits: list[str] = []
    out = text
    # 长键优先，避免「退款到哪了」被「退款到哪」抢先替换
    for key in sorted(ALIAS_TABLE, key=len, reverse=True):
        if key in lowered:
            out = re.sub(re.escape(key), ALIAS_TABLE[key], out, flags=re.IGNORECASE)
            hits.append(key)
    return out, hits


def strip_filler(text: str) -> str:
    out = text
    for pattern in FILLER_PATTERNS:
        out = re.sub(pattern, "", out)
    return out.strip() or text


def resolve_coref(text: str, order_no: Optional[str]) -> tuple[str, bool]:
    """把「这个订单」之类的指代替换成具体订单号，让检索和工具都能对上。"""
    if not order_no or not _PRONOUN_RE.search(text):
        return text, False
    return _PRONOUN_RE.sub(f"订单 {order_no} ", text).strip(), True


def split_questions(text: str) -> list[str]:
    """多问题时拆开分别检索，否则两段语义会被平均成一个都不准的向量。"""
    parts = [p.strip() for p in _SPLIT_RE.split(text) if p and p.strip()]
    if len(parts) <= 1:
        # 也尝试按问号切分
        by_q = [p.strip() for p in re.split(r"[?？]", text) if p.strip()]
        if len(by_q) > 1:
            return by_q
    return parts or [text]


def rewrite_query(text: str, order_no: Optional[str] = None) -> RewriteResult:
    base = normalize(text)
    stripped = strip_filler(base)
    coref_text, coref_hit = resolve_coref(stripped, order_no)
    expanded, hits = expand_aliases(coref_text)
    return RewriteResult(
        original=text,
        rewritten=expanded,
        aliases_hit=hits,
        coref_replaced=coref_hit,
    )
