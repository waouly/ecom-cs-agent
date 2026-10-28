"""仓储层：所有数据库访问都从这里走，方便替换存储后端与打审计。"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

from .labels import (
    AFTERSALE_STATUS_LABEL,
    AFTERSALE_TYPE_LABEL,
    CATEGORY_LABEL,
    LOGISTICS_STATUS_LABEL,
    ORDER_STATUS_LABEL,
)
from .schema import DDL

NOW_FMT = "%Y-%m-%d %H:%M:%S"


def _now() -> str:
    return datetime.now().strftime(NOW_FMT)


class Repository:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # 连接与初始化
    # ------------------------------------------------------------------
    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def init_schema(self) -> None:
        self.conn.executescript(DDL)
        self.conn.commit()

    def reset(self) -> None:
        self.conn.executescript(
            """
            DROP TABLE IF EXISTS users;
            DROP TABLE IF EXISTS orders;
            DROP TABLE IF EXISTS logistics;
            DROP TABLE IF EXISTS aftersales;
            DROP TABLE IF EXISTS audit_log;
            DROP TABLE IF EXISTS feedback;
            DROP TABLE IF EXISTS kb_gaps;
            """
        )
        self.conn.commit()
        self.init_schema()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # 用户 / 订单 / 物流
    # ------------------------------------------------------------------
    def get_user(self, user_id: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        return dict(row) if row else None

    def upsert_user(self, user: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO users(user_id,name,phone,level) VALUES(?,?,?,?)",
            (user["user_id"], user.get("name"), user.get("phone"), user.get("level", "normal")),
        )

    def get_order(self, order_no: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute("SELECT * FROM orders WHERE order_no = ?", (order_no,)).fetchone()
        if not row:
            return None
        return self._decorate_order(dict(row))

    def list_orders(self, user_id: str, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM orders WHERE user_id = ? ORDER BY paid_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [self._decorate_order(dict(r)) for r in rows]

    @staticmethod
    def _decorate_order(d: dict[str, Any]) -> dict[str, Any]:
        d["status_label"] = ORDER_STATUS_LABEL.get(d.get("status", ""), d.get("status"))
        d["category_label"] = CATEGORY_LABEL.get(d.get("category", ""), d.get("category"))
        d["opened"] = bool(d.get("opened"))
        d["is_custom"] = bool(d.get("is_custom"))
        return d

    def upsert_order(self, order: dict[str, Any]) -> None:
        cols = ["order_no", "user_id", "sku", "title", "category", "amount", "status",
                "paid_at", "shipped_at", "signed_at", "opened", "is_custom", "address"]
        vals = [order.get(c) for c in cols]
        vals[cols.index("opened")] = 1 if order.get("opened") else 0
        vals[cols.index("is_custom")] = 1 if order.get("is_custom") else 0
        q = f"INSERT OR REPLACE INTO orders({','.join(cols)}) VALUES({','.join('?' * len(cols))})"
        self.conn.execute(q, vals)

    def get_logistics(self, order_no: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute("SELECT * FROM logistics WHERE order_no = ?", (order_no,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["status_label"] = LOGISTICS_STATUS_LABEL.get(d.get("status", ""), d.get("status"))
        return d

    def upsert_logistics(self, item: dict[str, Any]) -> None:
        cols = ["order_no", "carrier", "tracking_no", "status", "latest_event", "updated_at"]
        q = f"INSERT OR REPLACE INTO logistics({','.join(cols)}) VALUES({','.join('?' * len(cols))})"
        self.conn.execute(q, [item.get(c) for c in cols])

    # ------------------------------------------------------------------
    # 售后单（写操作）
    # ------------------------------------------------------------------
    def get_aftersale(self, aftersale_no: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM aftersales WHERE aftersale_no = ?", (aftersale_no,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["status_label"] = AFTERSALE_STATUS_LABEL.get(d.get("status", ""), d.get("status"))
        d["type_label"] = AFTERSALE_TYPE_LABEL.get(d.get("type", ""), d.get("type"))
        return d

    def find_aftersale_by_idem(self, key: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM aftersales WHERE idempotency_key = ?", (key,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["status_label"] = AFTERSALE_STATUS_LABEL.get(d.get("status", ""), d.get("status"))
        d["type_label"] = AFTERSALE_TYPE_LABEL.get(d.get("type", ""), d.get("type"))
        return d

    def list_aftersales(self, user_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM aftersales WHERE user_id = ? ORDER BY created_at DESC", (user_id,)
        ).fetchall()
        return [self.get_aftersale(r["aftersale_no"]) or dict(r) for r in rows]

    def create_aftersale(
        self,
        *,
        aftersale_no: str,
        order_no: str,
        user_id: str,
        type_: str,
        amount: float,
        idempotency_key: str,
        note: str = "",
        eta: str = "1-3 个工作日",
    ) -> dict[str, Any]:
        self.conn.execute(
            "INSERT INTO aftersales(aftersale_no,order_no,user_id,type,status,amount,created_at,eta,idempotency_key,note)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            (aftersale_no, order_no, user_id, type_, "pending_review", amount, _now(), eta, idempotency_key, note),
        )
        return self.get_aftersale(aftersale_no) or {}

    def set_order_status(self, order_no: str, status: str) -> None:
        self.conn.execute("UPDATE orders SET status = ? WHERE order_no = ?", (status, order_no))

    def update_order_address(self, order_no: str, address: str) -> None:
        self.conn.execute("UPDATE orders SET address = ? WHERE order_no = ?", (address, order_no))

    # ------------------------------------------------------------------
    # 审计 / 反馈 / 知识缺口
    # ------------------------------------------------------------------
    def add_audit(
        self, actor: str, action: str, target: str, payload: dict[str, Any], result: str
    ) -> None:
        self.conn.execute(
            "INSERT INTO audit_log(ts,actor,action,target,payload,result) VALUES(?,?,?,?,?,?)",
            (_now(), actor, action, target, json.dumps(payload, ensure_ascii=False), result),
        )

    def list_audit(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def add_feedback(self, session_id: str, rating: int, comment: str = "", intent: str = "") -> None:
        self.conn.execute(
            "INSERT INTO feedback(ts,session_id,rating,comment,intent) VALUES(?,?,?,?,?)",
            (_now(), session_id, rating, comment, intent),
        )

    def feedback_stats(self) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n, AVG(rating) AS avg_rating,"
            " SUM(CASE WHEN rating >= 1 THEN 1 ELSE 0 END) AS positive FROM feedback"
        ).fetchone()
        return dict(row) if row else {"n": 0, "avg_rating": 0, "positive": 0}

    def add_gap(
        self, query: str, intent: str, confidence: float, decision: str, cluster: str = ""
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO kb_gaps(ts,query,intent,confidence,decision,cluster) VALUES(?,?,?,?,?,?)",
            (_now(), query, intent, confidence, decision, cluster),
        )
        self.conn.commit()
        return int(cur.lastrowid or 0)

    def resolve_gap(self, gap_id: int) -> None:
        self.conn.execute("UPDATE kb_gaps SET resolved = 1 WHERE id = ?", (gap_id,))

    def list_gaps(self, limit: int = 50, unresolved_only: bool = False) -> list[dict[str, Any]]:
        sql = "SELECT * FROM kb_gaps"
        params: list[Any] = []
        if unresolved_only:
            sql += " WHERE resolved = 0"
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def gap_clusters(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT cluster, COUNT(*) AS n FROM kb_gaps WHERE cluster <> ''"
            " GROUP BY cluster ORDER BY n DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict[str, Any]:
        def one(sql: str) -> int:
            row = self.conn.execute(sql).fetchone()
            return int(row[0]) if row else 0

        return {
            "users": one("SELECT COUNT(*) FROM users"),
            "orders": one("SELECT COUNT(*) FROM orders"),
            "aftersales": one("SELECT COUNT(*) FROM aftersales"),
            "audit": one("SELECT COUNT(*) FROM audit_log"),
            "feedback": one("SELECT COUNT(*) FROM feedback"),
            "gaps": one("SELECT COUNT(*) FROM kb_gaps"),
        }
