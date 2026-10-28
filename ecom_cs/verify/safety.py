"""安全与合规检查：PII、越权承诺、提示注入残留在输出里的痕迹。

三条规则都来自真实客服系统的审计要求：
1. 输出里出现完整手机号/身份证/银行卡必须打码（日志与前端都不得裸露）；
2. 不得给出「保证」「100%」「绝对」这类承诺性表述——这是可被追溯的法律风险；
3. 用户注入的指令如果被模型照做，输出里往往会出现「系统提示」「以下是我的设定」
   之类的痕迹，要在出站前拦掉。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..utils.text import contains_pii, mask_pii

PROMISE_PATTERNS = [
    (r"保证(一定|可以|会)?", "承诺性表述：保证"),
    (r"100\s*[%％]", "承诺性表述：100%"),
    (r"绝对(可以|能|会|没问题)", "承诺性表述：绝对"),
    (r"肯定(可以|能|会)", "承诺性表述：肯定"),
    (r"永久|终身免费", "承诺性表述：永久/终身"),
    (r"包(退|赔|换)", "承诺性表述：包退包赔"),
]

LEAK_PATTERNS = [
    (r"系统(提示词|prompt)", "疑似泄露系统提示"),
    (r"我的(设定|指令)是", "疑似泄露系统提示"),
    (r"(忽略|无视)(之前|以上|上面)的?(指令|要求)", "复述注入指令"),
]

SENSITIVE_REQUEST = re.compile(r"(所有用户|全部订单|他人(的)?订单|别的用户(的)?信息)")


@dataclass
class SafetyIssue:
    kind: str
    detail: str
    severity: str = "warn"     # warn | block

    def to_dict(self) -> dict:
        return {"kind": self.kind, "detail": self.detail, "severity": self.severity}


@dataclass
class SafetyReport:
    issues: list[SafetyIssue] = field(default_factory=list)
    redacted_reply: str = ""

    @property
    def blocked(self) -> bool:
        return any(i.severity == "block" for i in self.issues)

    def to_dict(self) -> dict:
        return {"blocked": self.blocked, "issues": [i.to_dict() for i in self.issues]}


def check_output(reply: str) -> SafetyReport:
    report = SafetyReport(redacted_reply=reply)

    if contains_pii(reply):
        report.issues.append(SafetyIssue("pii_leak", "回复包含未脱敏的个人信息", "block"))
        report.redacted_reply = mask_pii(reply)

    for pattern, detail in PROMISE_PATTERNS:
        if re.search(pattern, reply):
            report.issues.append(SafetyIssue("over_promise", detail, "warn"))

    for pattern, detail in LEAK_PATTERNS:
        if re.search(pattern, reply):
            report.issues.append(SafetyIssue("prompt_leak", detail, "block"))

    return report


def check_input(text: str) -> SafetyReport:
    report = SafetyReport(redacted_reply=text)
    if SENSITIVE_REQUEST.search(text):
        report.issues.append(SafetyIssue("bulk_data_request", "请求批量用户数据", "block"))
    return report
