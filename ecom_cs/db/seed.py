"""演示数据。所有日期相对「今天」生成，保证 days_since_sign 这类判定长期有效。"""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta
from typing import Any, Optional

from .repo import NOW_FMT, Repository


def _d(day: date, hh: int = 10, mm: int = 30) -> str:
    return datetime(day.year, day.month, day.day, hh, mm).strftime(NOW_FMT)


USERS: list[dict[str, Any]] = [
    {"user_id": "u_demo", "name": "张伟", "phone": "138****5678", "level": "gold"},
    {"user_id": "u_other", "name": "李娜", "phone": "139****1234", "level": "normal"},
]


def build_orders(today: Optional[date] = None) -> list[dict[str, Any]]:
    t = today or date.today()

    def signed(n: int) -> str:
        return _d(t - timedelta(days=n))

    def shipped(n: int) -> str:
        return _d(t - timedelta(days=n))

    def paid(n: int) -> str:
        return _d(t - timedelta(days=n))

    return [
        {
            # 7 天无理由：满足
            "order_no": "O10001", "user_id": "u_demo", "sku": "SKU-APP-001",
            "title": "纯棉圆领短袖T恤", "category": "apparel", "amount": 299.0,
            "status": "signed", "paid_at": paid(6), "shipped_at": shipped(5),
            "signed_at": signed(3), "opened": False, "is_custom": False,
            "address": "湖南省长沙市芙蓉区人民路 88 号 1 栋 302",
        },
        {
            # 超出 7 天
            "order_no": "O10002", "user_id": "u_demo", "sku": "SKU-APP-002",
            "title": "加厚连帽卫衣", "category": "apparel", "amount": 459.0,
            "status": "signed", "paid_at": paid(15), "shipped_at": shipped(14),
            "signed_at": signed(12), "opened": False, "is_custom": False,
            "address": "湖南省长沙市芙蓉区人民路 88 号 1 栋 302",
        },
        {
            # 数码已拆封
            "order_no": "O10003", "user_id": "u_demo", "sku": "SKU-DIG-001",
            "title": "无线蓝牙降噪耳机", "category": "digital", "amount": 899.0,
            "status": "signed", "paid_at": paid(4), "shipped_at": shipped(3),
            "signed_at": signed(2), "opened": True, "is_custom": False,
            "address": "湖南省长沙市芙蓉区人民路 88 号 1 栋 302",
        },
        {
            # 定制商品
            "order_no": "O10004", "user_id": "u_demo", "sku": "SKU-CUS-001",
            "title": "定制刻字实木书桌", "category": "custom", "amount": 2380.0,
            "status": "signed", "paid_at": paid(9), "shipped_at": shipped(5),
            "signed_at": signed(1), "opened": True, "is_custom": True,
            "address": "湖南省长沙市芙蓉区人民路 88 号 1 栋 302",
        },
        {
            # 内衣类目不支持无理由
            "order_no": "O10005", "user_id": "u_demo", "sku": "SKU-UND-001",
            "title": "纯棉无痕内衣", "category": "underwear", "amount": 129.0,
            "status": "signed", "paid_at": paid(3), "shipped_at": shipped(2),
            "signed_at": signed(1), "opened": False, "is_custom": False,
            "address": "湖南省长沙市芙蓉区人民路 88 号 1 栋 302",
        },
        {
            # 质量问题：已拆封但可退
            "order_no": "O10006", "user_id": "u_demo", "sku": "SKU-APP-003",
            "title": "儿童纯棉家居服", "category": "apparel", "amount": 189.0,
            "status": "signed", "paid_at": paid(8), "shipped_at": shipped(7),
            "signed_at": signed(5), "opened": True, "is_custom": False,
            "address": "湖南省长沙市芙蓉区人民路 88 号 1 栋 302",
        },
        {
            # 在途，未签收
            "order_no": "O10007", "user_id": "u_demo", "sku": "SKU-FRE-001",
            "title": "云南高山蓝莓 2 斤装", "category": "fresh", "amount": 79.9,
            "status": "shipped", "paid_at": paid(2), "shipped_at": shipped(1),
            "signed_at": None, "opened": False, "is_custom": False,
            "address": "湖南省长沙市芙蓉区人民路 88 号 1 栋 302",
        },
        {
            # 已在退款中
            "order_no": "O10008", "user_id": "u_demo", "sku": "SKU-BEA-001",
            "title": "氨基酸洁面乳", "category": "beauty", "amount": 168.0,
            "status": "refunding", "paid_at": paid(22), "shipped_at": shipped(21),
            "signed_at": signed(18), "opened": True, "is_custom": False,
            "address": "湖南省长沙市芙蓉区人民路 88 号 1 栋 302",
        },
        {
            # 他人订单：越权测试用
            "order_no": "O10009", "user_id": "u_other", "sku": "SKU-APP-004",
            "title": "女士针织开衫", "category": "apparel", "amount": 399.0,
            "status": "signed", "paid_at": paid(5), "shipped_at": shipped(4),
            "signed_at": signed(2), "opened": False, "is_custom": False,
            "address": "湖南省株洲市天元区黄河北路 12 号",
        },
        {
            # 待发货：唯一可以走「改地址」写操作链路的订单
            # （其余订单都已发货，改地址应当被拒绝——这正是幂等/前置条件校验的用例）
            "order_no": "O10010", "user_id": "u_demo", "sku": "SKU-APP-005",
            "title": "男士休闲夹克", "category": "apparel", "amount": 529.0,
            "status": "paid", "paid_at": paid(0), "shipped_at": None,
            "signed_at": None, "opened": False, "is_custom": False,
            "address": "湖南省长沙市芙蓉区人民路 88 号 1 栋 302",
        },
    ]


def build_logistics(today: Optional[date] = None) -> list[dict[str, Any]]:
    t = today or date.today()

    def ts(n: int, hh: int = 14) -> str:
        return _d(t - timedelta(days=n), hh, 0)

    return [
        {"order_no": "O10001", "carrier": "顺丰速运", "tracking_no": "SF1234567890123",
         "status": "signed", "latest_event": "已签收，签收人：本人", "updated_at": ts(3)},
        {"order_no": "O10003", "carrier": "京东物流", "tracking_no": "JD9988776655443",
         "status": "signed", "latest_event": "已签收，签收人：门卫代收", "updated_at": ts(2)},
        {"order_no": "O10007", "carrier": "中通快递", "tracking_no": "ZT5566778899001",
         "status": "out_for_delivery", "latest_event": "快件已到达【长沙芙蓉区营业点】，正在派送",
         "updated_at": ts(0, 9)},
        {"order_no": "O10008", "carrier": "圆通速递", "tracking_no": "YT2233445566778",
         "status": "signed", "latest_event": "已签收", "updated_at": ts(18)},
    ]


def seed(repo: Repository, today: Optional[date] = None, force: bool = False) -> dict[str, int]:
    """写入演示数据。已存在且 force=False 时直接返回现状。"""
    repo.init_schema()
    existing = repo.stats()
    if existing["orders"] > 0 and not force:
        return existing
    if force:
        repo.reset()

    for u in USERS:
        repo.upsert_user(u)
    for o in build_orders(today):
        repo.upsert_order(o)
    for lg in build_logistics(today):
        repo.upsert_logistics(lg)

    # 历史售后单：O10008 已在退款流程
    if not repo.get_aftersale("AS20260001"):
        repo.create_aftersale(
            aftersale_no="AS20260001", order_no="O10008", user_id="u_demo",
            type_="refund_only", amount=168.0,
            idempotency_key="seed-AS20260001", note="商品与描述不符",
        )
    repo.conn.commit()
    return repo.stats()


def random_feedback(repo: Repository, n: int = 20, seed_value: int = 7) -> None:
    """造一批反馈，供指标看板演示。"""
    rnd = random.Random(seed_value)
    intents = ["policy_qa", "order_query", "logistics", "aftersale", "refund"]
    for _ in range(n):
        repo.add_feedback(
            session_id=f"s-{rnd.randint(1000, 9999)}",
            rating=rnd.choice([1, 1, 1, 0, -1]),
            comment="",
            intent=rnd.choice(intents),
        )
    repo.conn.commit()
