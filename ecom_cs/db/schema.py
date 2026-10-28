"""SQLite 建表 DDL。

字段与 MySQL 基本兼容（TEXT/INTEGER/REAL），切换 MySQL 只需替换驱动与
少量类型（REAL -> DECIMAL），SQL 主体不用改。
"""

from __future__ import annotations

DDL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS users (
    user_id   TEXT PRIMARY KEY,
    name      TEXT NOT NULL,
    phone     TEXT,
    level     TEXT DEFAULT 'normal'
);

CREATE TABLE IF NOT EXISTS orders (
    order_no   TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    sku        TEXT,
    title      TEXT,
    category   TEXT,
    amount     REAL DEFAULT 0,
    status     TEXT,
    paid_at    TEXT,
    shipped_at TEXT,
    signed_at  TEXT,
    opened     INTEGER DEFAULT 0,
    is_custom  INTEGER DEFAULT 0,
    address    TEXT
);

CREATE TABLE IF NOT EXISTS logistics (
    order_no     TEXT PRIMARY KEY,
    carrier      TEXT,
    tracking_no  TEXT,
    status       TEXT,
    latest_event TEXT,
    updated_at   TEXT
);

CREATE TABLE IF NOT EXISTS aftersales (
    aftersale_no    TEXT PRIMARY KEY,
    order_no        TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    type            TEXT,
    status          TEXT,
    amount          REAL DEFAULT 0,
    created_at      TEXT,
    eta             TEXT,
    idempotency_key TEXT UNIQUE,
    note            TEXT
);

CREATE TABLE IF NOT EXISTS audit_log (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TEXT NOT NULL,
    actor   TEXT,
    action  TEXT,
    target  TEXT,
    payload TEXT,
    result  TEXT
);

CREATE TABLE IF NOT EXISTS feedback (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT NOT NULL,
    session_id TEXT,
    rating     INTEGER,
    comment    TEXT,
    intent     TEXT
);

CREATE TABLE IF NOT EXISTS kb_gaps (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT NOT NULL,
    query      TEXT,
    intent     TEXT,
    confidence REAL,
    decision   TEXT,
    cluster    TEXT,
    resolved   INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_orders_user ON orders(user_id);
CREATE INDEX IF NOT EXISTS idx_aftersales_order ON aftersales(order_no);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts);
"""
