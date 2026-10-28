"""工具层测试：二次确认、幂等、越权、失败补偿。"""

import unittest
from types import SimpleNamespace

from ecom_cs.tools.base import ToolContext, ToolError
from ecom_cs.tools.write_tools import ChangeAddressTool, CreateReturnTool, HandoffTool
from tests.conftest_helpers import TempWorld

ADDR = "湖南省长沙市岳麓区麓山南路 2 号"


class WriteToolTest(TempWorld):
    def ctx(self, user: str = "u_demo", session: str = "s1") -> ToolContext:
        sess = SimpleNamespace(user_id=user, session_id=session)
        return ToolContext(session=sess, repo=self.deps.repo, today=self.deps.today, settings=self.settings)

    def setUp(self):
        # 每个用例回到种子状态：写操作会真的落库
        self.deps.repo.reset()
        self.deps.repo.init_schema()
        from ecom_cs.db.seed import seed

        seed(self.deps.repo, self.deps.today)

    def test_preview_requires_confirm(self):
        res = ChangeAddressTool().run(
            {"order_no": "O10010", "new_address": ADDR, "confirm": False}, self.ctx()
        )
        self.assertTrue(res.ok)
        self.assertTrue(res.data["needs_confirm"])
        self.assertIn("确认", res.data["message"])
        # 预览不应该改库
        self.assertNotEqual(self.deps.repo.get_order("O10010")["address"], ADDR)

    def test_confirm_applies_and_audits(self):
        tool = ChangeAddressTool()
        tool.run({"order_no": "O10010", "new_address": ADDR, "confirm": False}, self.ctx())
        res = tool.run({"order_no": "O10010", "new_address": ADDR, "confirm": True}, self.ctx())
        self.assertTrue(res.ok)
        self.assertTrue(res.data["changed"])
        self.assertEqual(self.deps.repo.get_order("O10010")["address"], ADDR)
        audits = self.deps.repo.list_audit(limit=10)
        self.assertTrue(any(a["action"] == "change_address" for a in audits))

    def test_idempotent_replay(self):
        tool = ChangeAddressTool()
        first = tool.run({"order_no": "O10010", "new_address": ADDR, "confirm": True}, self.ctx())
        self.assertTrue(first.data["changed"])
        # 第二次同样的请求：地址已经等于目标值，工具直接短路，不重复写、不重复审计
        again = tool.run({"order_no": "O10010", "new_address": ADDR, "confirm": True}, self.ctx())
        self.assertTrue(again.ok)
        self.assertFalse(again.data["changed"])
        self.assertEqual(self.deps.repo.get_order("O10010")["address"], ADDR)

    def test_shipped_order_rejected(self):
        with self.assertRaises(ToolError) as cm:
            ChangeAddressTool().run(
                {"order_no": "O10001", "new_address": ADDR, "confirm": False}, self.ctx()
            )
        self.assertEqual(cm.exception.code, "already_shipped")

    def test_cross_user_order_is_not_found(self):
        # 越权与不存在必须返回同一个错误码，否则就是订单号探测侧信道
        with self.assertRaises(ToolError) as cm:
            ChangeAddressTool().run(
                {"order_no": "O10009", "new_address": ADDR, "confirm": False}, self.ctx()
            )
        self.assertEqual(cm.exception.code, "order_not_found")

    def test_create_return_idempotent(self):
        tool = CreateReturnTool()
        args = {"order_no": "O10001", "reason": "不想要了", "aftersale_type": "return_refund",
                "confirm": True}
        first = tool.run(args, self.ctx())
        second = tool.run(args, self.ctx())
        self.assertTrue(first.ok and second.ok)
        self.assertEqual(second.data["aftersale_no"], first.data["aftersale_no"])
        # 同一用户下该订单只应留下一条售后单
        mine = [a for a in self.deps.repo.list_aftersales("u_demo") if a["order_no"] == "O10001"]
        self.assertEqual(len(mine), 1)

    def test_handoff_creates_ticket(self):
        res = HandoffTool().run({"reason": "用户要求人工"}, self.ctx())
        self.assertTrue(res.ok)
        self.assertTrue(res.data["ticket"].startswith("CS"))


if __name__ == "__main__":
    unittest.main()
