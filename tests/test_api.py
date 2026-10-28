"""API / 演示台冒烟：静态页可访问、/chat 返回可复盘的字段、政策接口可用。

用 create_app(service=...) 注入临时依赖，避免污染工作区的 data/app.db。
"""

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from ecom_cs.api.app import create_app
from ecom_cs.orchestrator import ChatService, build_deps

from conftest_helpers import make_settings


class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        deps = build_deps(make_settings(Path(cls._tmp.name)))
        cls.client = TestClient(create_app(service=ChatService(deps)))
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        cls.client.app.state.svc.deps.repo.conn.close()
        cls._tmp.cleanup()

    def test_index_serves_ui(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("电商智能客服", r.text)
        # 演示台的 UI 不依赖任何 CDN，离线可用
        self.assertNotIn("http://", r.text)

    def test_chat_and_decision_payload(self):
        r = self.client.post(
            "/chat",
            json={"session_id": "t1", "user_id": "u_demo", "message": "O10001 签收三天还没拆，能退吗？"},
        )
        self.assertEqual(r.status_code, 200)
        out = r.json()
        self.assertEqual(out["decision"], "answer")
        self.assertIn("RET-7D-001", [c["clause_id"] for c in out["citations"]])
        for key in ("confidence", "threshold", "tier", "route", "nodes"):
            self.assertIn(key, out)

    def test_policy_detail(self):
        r = self.client.get("/policies/RET-7D-001")
        self.assertEqual(r.status_code, 200)
        self.assertIn("七天无理由", r.json()["title"])

    def test_orders_feed_returns_demo_orders(self):
        r = self.client.get("/orders", params={"user_id": "u_demo", "limit": 5})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertLessEqual(body["count"], 5)
        order = next(o for o in body["orders"] if o["order_no"] == "O10001")
        # 商品卡片要能显示名称与价格，所以列表接口必须带 title/amount/status_label
        self.assertTrue(order["title"])
        self.assertGreater(order["amount"], 0)
        self.assertTrue(order["status_label"])

    def test_order_detail_hides_other_users_order(self):
        # O10009 属于 u_other：查询者不是所有者时，与「不存在」返回同一个 404，
        # 避免用订单号撞库探测他人订单（越权侧信道）。
        self.assertEqual(self.client.get("/orders/O10009", params={"user_id": "u_demo"}).status_code, 404)
        self.assertEqual(self.client.get("/orders/O10009", params={"user_id": "u_other"}).status_code, 200)
        self.assertEqual(self.client.get("/orders/O99999", params={"user_id": "u_demo"}).status_code, 404)

    def test_chat_attaches_product_card(self):
        r = self.client.post(
            "/chat",
            json={"session_id": "t2", "user_id": "u_demo", "message": "O10001 签收三天还没拆，能退吗？"},
        )
        self.assertEqual(r.status_code, 200)
        product = r.json().get("product")
        self.assertIsNotNone(product)
        self.assertEqual(product["order_no"], "O10001")
        self.assertIn("title", product)
        self.assertIn("status_label", product)


if __name__ == "__main__":
    unittest.main()
