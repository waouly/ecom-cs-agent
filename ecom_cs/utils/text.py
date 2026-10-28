"""文本工具：归一化、分词、分句、数字抽取、PII 脱敏、相似度。

全部零依赖（不引 jieba），面向中文 + 混合英文的电商口语语料。
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Iterable

# ---------------------------------------------------------------------------
# 口语 / 错别字 / 缩写归一化
# ---------------------------------------------------------------------------

COLLOQUIAL_MAP: dict[str, str] = {
    "退活": "退货",
    "退贷": "退货",
    "退钱": "退款",
    "给钱": "退款",
    "退全款": "全额退款",
    "帐号": "账号",
    "定单": "订单",
    "物流单好": "物流单号",
    "zfb": "支付宝",
    "wx": "微信",
    "vx": "微信",
    "扣扣": "qq",
    "咋": "怎么",
    "咋办": "怎么办",
    "啥": "什么",
    "什么时候": "何时",
    "没收到": "未收到",
    "收不到": "未收到",
    "不想要了": "想退货",
    "拍错了": "下错单",
    "改地址": "修改收货地址",
    "发货没": "是否发货",
    "到哪了": "物流进度",
    "亲": "",
    "客服mm": "客服",
    "客服MM": "客服",
}

FILLER_PATTERNS = [
    r"[~～]{2,}",
    r"[!！?？]{2,}",
    r"\s+",
]


def normalize(text: str) -> str:
    """全角转半角、小写、去填充、口语归一。"""
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text).strip().lower()
    for k, v in COLLOQUIAL_MAP.items():
        if k in t:
            t = t.replace(k, v)
    for pat in FILLER_PATTERNS:
        t = re.sub(pat, " " if pat == r"\s+" else "", t)
    return t.strip()


def normalize_light(text: str) -> str:
    """只做保守归一（保留原始大小写与标点），用于展示与数字核验。"""
    return unicodedata.normalize("NFKC", text).strip()


# ---------------------------------------------------------------------------
# 分词：中文按字符 bigram + 英文/数字按词
# ---------------------------------------------------------------------------

_LATIN_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    t = normalize(text)
    tokens: list[str] = []
    # 英文/数字词
    for m in _LATIN_RE.finditer(t):
        tokens.append(m.group(0))
    # 中文 bigram（去掉英文后再切）
    cjk_only = "".join(ch for ch in t if _CJK_RE.match(ch))
    for i in range(len(cjk_only) - 1):
        tokens.append(cjk_only[i : i + 2])
    if len(cjk_only) == 1:
        tokens.append(cjk_only)
    return tokens


def char_ngrams(text: str, n: int = 2) -> list[str]:
    t = re.sub(r"\s+", "", normalize(text))
    if len(t) < n:
        return [t] if t else []
    return [t[i : i + n] for i in range(len(t) - n + 1)]


# ---------------------------------------------------------------------------
# 分句：中英文句末 + 逗号切分（用于 claim 拆解）
# ---------------------------------------------------------------------------

_SPLIT_RE = re.compile(r"(?<=[。！？!?；;\n])|(?<=\.)\s+")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SPLIT_RE.split(text or "") if p and p.strip()]
    out: list[str] = []
    for p in parts:
        # 过长的句子再按逗号切，便于 claim 级核验
        if len(p) > 60 and ("，" in p or "," in p):
            out.extend([x.strip() for x in re.split(r"[，,]", p) if x.strip()])
        else:
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# 数字抽取：用于回答中的金额/时效强校验
# ---------------------------------------------------------------------------

_UNITS = [
    "个工作日", "个自然日", "个工作日", "工作日", "自然日",
    "小时", "分钟", "天", "日", "个月", "月", "年", "周",
    "元", "%", "折", "倍", "件", "次", "公里", "kg", "克", "米",
]
_UNIT_RE = "|".join(sorted({re.escape(u) for u in _UNITS}, key=len, reverse=True))
_NUM_RE = re.compile(rf"(\d+(?:\.\d+)?)\s*({_UNIT_RE})")


def extract_numbers(text: str) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for m in _NUM_RE.finditer(normalize_light(text)):
        out.append({"value": float(m.group(1)), "unit": m.group(2), "raw": m.group(0)})
    return out


def amount_tokens(text: str) -> set[str]:
    """把数字+单位归一成集合，用于「回答里的数字必须在证据中出现」的判定。"""
    toks: set[str] = set()
    for item in extract_numbers(text):
        toks.add(f"{item['value']:g}{item['unit']}")
    return toks


# ---------------------------------------------------------------------------
# PII 脱敏
# ---------------------------------------------------------------------------

PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
IDCARD_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
BANKCARD_RE = re.compile(r"(?<!\d)\d{16,19}(?!\d)")


def mask_pii(text: str) -> str:
    if not text:
        return text
    t = PHONE_RE.sub(lambda m: m.group(0)[:3] + "****" + m.group(0)[-4:], text)
    t = IDCARD_RE.sub(lambda m: m.group(0)[:4] + "**********" + m.group(0)[-4:], t)
    t = BANKCARD_RE.sub(lambda m: m.group(0)[:4] + "****" + m.group(0)[-4:], t)
    t = EMAIL_RE.sub(lambda m: m.group(0)[:2] + "***@" + m.group(0).split("@")[-1], t)
    return t


def contains_pii(text: str) -> bool:
    return bool(PHONE_RE.search(text) or IDCARD_RE.search(text) or BANKCARD_RE.search(text))


# ---------------------------------------------------------------------------
# 相似度
# ---------------------------------------------------------------------------


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def char_jaccard(a: str, b: str) -> float:
    return jaccard(char_ngrams(a, 2), char_ngrams(b, 2))


def overlap_ratio(query: str, doc: str) -> float:
    """query 的 token 有多大比例出现在 doc 中（召回视角，对长文档友好）。"""
    q = set(tokenize(query))
    d = set(tokenize(doc))
    if not q:
        return 0.0
    return len(q & d) / len(q)


def cosine(a, b) -> float:
    import numpy as np

    va = np.asarray(a, dtype="float32")
    vb = np.asarray(b, dtype="float32")
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def truncate(text: str, limit: int = 200) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"
