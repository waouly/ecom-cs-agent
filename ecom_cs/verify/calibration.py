"""校准：Platt scaling（概率校准）+ 保形阈值（覆盖率保证）。

两个问题分开解决：

1. **分数不准**。「0.62 分」到底是不是「62% 答对」？Platt scaling 用逻辑回归
   把原始融合分映射成概率，拟合只用几十条带标签样本，成本极低。
2. **阈值没有保证**。拍一个阈值，你无法说明「低于它就转人工」到底拦住了多少
   错误、又误拦了多少正确。这里用 split conformal 的思路：在验证集上找最小的
   阈值，使得「被接受的样本里错误率 ≤ alpha」。这是**有覆盖率保证**的转人工策略，
   比拍阈值强得多，也是这套系统最值得讲的技术点之一。

注意 alpha 越小越保守：alpha=0.05 意味着「接受的答案里最多 5% 是错的」，
代价是覆盖率下降（更多转人工）。业务上这个旋钮应该和人工成本一起算。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from ..utils.text import clamp, sigmoid


@dataclass
class PlattCalibrator:
    a: float = 1.0
    b: float = 0.0
    n: int = 0

    def transform(self, x: float) -> float:
        return clamp(sigmoid(self.a * x + self.b))

    def fit(self, pairs: Iterable[tuple[float, bool]], lr: float = 0.5,
            epochs: int = 400) -> "PlattCalibrator":
        data = [(float(x), 1.0 if y else 0.0) for x, y in pairs]
        self.n = len(data)
        if self.n < 8:
            return self
        for _ in range(epochs):
            ga = gb = 0.0
            for x, y in data:
                p = sigmoid(self.a * x + self.b)
                err = p - y
                ga += err * x
                gb += err
            self.a -= lr * ga / self.n
            self.b -= lr * gb / self.n
        return self

    def to_dict(self) -> dict:
        return {"a": round(self.a, 6), "b": round(self.b, 6), "n": self.n}

    @classmethod
    def from_dict(cls, raw: dict) -> "PlattCalibrator":
        return cls(a=float(raw.get("a", 1.0)), b=float(raw.get("b", 0.0)), n=int(raw.get("n", 0)))


@dataclass
class SelectiveThreshold:
    threshold: float = 0.62
    alpha: float = 0.05
    coverage: float = 0.0
    risk: float = 0.0
    n: int = 0

    @classmethod
    def fit(cls, records: Iterable[tuple[float, bool]], alpha: float = 0.05,
            min_accept: int = 5, fallback: float = 0.62,
            floor: float = 0.5) -> "SelectiveThreshold":
        """records: (置信度, 是否答对)。找最小的 t 使 accepted 的错误率 <= alpha。

        ``floor`` 是工程下限：样本全对时 risk 恒为 0，公式上「任意阈值」都成立，
        会退化成「0.0 置信度也自动答复」。置信度低于 0.5 本来就该转人工，
        所以把接受区间硬性截在这条线下。
        """
        data = sorted(((float(c), bool(ok)) for c, ok in records), key=lambda x: -x[0])
        n = len(data)
        if n < min_accept:
            return cls(threshold=fallback, alpha=alpha, coverage=0.0, risk=0.0, n=n)

        best: Optional[SelectiveThreshold] = None
        errors = 0
        for i, (conf, ok) in enumerate(data, start=1):
            if not ok:
                errors += 1
            risk = errors / i
            if i >= min_accept and risk <= alpha and conf >= floor:
                best = cls(threshold=conf, alpha=alpha, coverage=i / n, risk=risk, n=n)
            elif best is not None and (risk > alpha or conf < floor):
                break
        if best is None:
            return cls(threshold=0.99, alpha=alpha, coverage=0.0, risk=0.0, n=n)
        return best

    def to_dict(self) -> dict:
        return {
            "threshold": round(self.threshold, 6), "alpha": self.alpha,
            "coverage": round(self.coverage, 4), "risk": round(self.risk, 4), "n": self.n,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "SelectiveThreshold":
        return cls(
            threshold=float(raw.get("threshold", 0.62)),
            alpha=float(raw.get("alpha", 0.05)),
            coverage=float(raw.get("coverage", 0.0)),
            risk=float(raw.get("risk", 0.0)),
            n=int(raw.get("n", 0)),
        )


@dataclass
class CalibrationBundle:
    platt: PlattCalibrator = field(default_factory=PlattCalibrator)
    selective: SelectiveThreshold = field(default_factory=SelectiveThreshold)

    def probability(self, raw: float) -> float:
        """原始融合分 -> 校准后的答对概率。"""
        if self.platt.n >= 8:
            return self.platt.transform(raw)
        return clamp(raw)

    def route(self, raw: float) -> tuple[float, bool]:
        """返回 (校准概率, 是否接受为自动答复)。"""
        p = self.probability(raw)
        return p, p >= self.selective.threshold

    def to_dict(self) -> dict:
        return {"platt": self.platt.to_dict(), "selective": self.selective.to_dict()}

    @classmethod
    def from_dict(cls, raw: dict) -> "CalibrationBundle":
        return cls(
            platt=PlattCalibrator.from_dict(raw.get("platt", {})),
            selective=SelectiveThreshold.from_dict(raw.get("selective", {})),
        )

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "CalibrationBundle":
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, KeyError, ValueError):
            return cls()


def expected_calibration_error(records: Iterable[tuple[float, bool]], bins: int = 10) -> float:
    """ECE：把预测概率分桶，比较「平均置信」与「实际正确率」。"""
    data = [(clamp(float(c)), 1.0 if ok else 0.0) for c, ok in records]
    if not data:
        return 0.0
    buckets: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
    for conf, hit in data:
        idx = min(bins - 1, int(conf * bins))
        buckets[idx].append((conf, hit))
    ece = 0.0
    total = len(data)
    for bucket in buckets:
        if not bucket:
            continue
        avg_conf = sum(c for c, _ in bucket) / len(bucket)
        avg_hit = sum(h for _, h in bucket) / len(bucket)
        ece += abs(avg_conf - avg_hit) * len(bucket) / total
    return ece


def brier_score(records: Iterable[tuple[float, bool]]) -> float:
    data = [(clamp(float(c)), 1.0 if ok else 0.0) for c, ok in records]
    if not data:
        return 0.0
    return sum((c - h) ** 2 for c, h in data) / len(data)
