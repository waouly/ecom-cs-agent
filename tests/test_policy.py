"""政策层测试：DSL 三值求值、规则引擎判定、信息型条款与生效期。"""

import unittest
from datetime import date

from ecom_cs.config import PROJECT_ROOT
from ecom_cs.kb.policy_dsl import eval_all, eval_condition, parse_condition
from ecom_cs.kb.policy_engine import PolicyEngine, load_docs
from ecom_cs.types import PolicyStatus

TODAY = date(2026, 10, 28)
POLICIES = PROJECT_ROOT / "configs" / "policies"


def facts(**over) -> dict:
    base = {
        "order_no": "O10001", "category": "apparel", "amount": 299.0,
        "order_status": "signed", "opened": False, "is_custom": False,
        "shipped": True, "signed": True, "days_since_paid": 6,
        "days_since_sign": 3, "member_level": "gold",
        "has_quality_issue": False, "has_evidence": False,
    }
    base.update(over)
    return base


class DslTest(unittest.TestCase):
    def test_three_valued_logic(self):
        conds = [parse_condition({"field": "days_since_sign", "op": "<=", "value": 7})]
        self.assertTrue(eval_all(conds, facts()))
        self.assertFalse(eval_all(conds, facts(days_since_sign=30)))
        # 字段缺失必须是「未知」而不是 False，否则会误判为不满足条件
        self.assertIsNone(eval_all(conds, {}))

    def test_unknown_op_rejected_at_parse(self):
        with self.assertRaises(Exception):
            parse_condition({"field": "amount", "op": "~=", "value": 1})

    def test_in_and_exists_ops(self):
        c = parse_condition({"field": "category", "op": "in", "value": ["apparel", "digital"]})
        self.assertTrue(eval_condition(c, facts()))
        self.assertFalse(eval_condition(c, facts(category="fresh")))


class EngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs, cls.errors = load_docs(POLICIES)
        cls.engine = PolicyEngine(cls.docs, TODAY)

    def test_no_parse_errors(self):
        self.assertEqual(self.errors, [])
        self.assertGreaterEqual(len(self.engine.clauses), 20)

    def test_info_clauses_are_typed(self):
        # 「退款 1-3 个工作日」这类条款没有条件、没有类目，只能是信息型，
        # 否则会被当成资格依据去下「可以退」的结论。
        types = {c.policy_type for c in self.engine.clauses}
        self.assertEqual(types, {"eligibility", "info"})
        info = self.engine.by_id("REF-001")
        self.assertEqual(info.policy_type, "info")

    def test_applicable_refund(self):
        bundle = self.engine.evaluate(facts())
        self.assertEqual(bundle.primary.clause_id, "RET-7D-001")
        self.assertEqual(bundle.primary.status, PolicyStatus.APPLICABLE)
        self.assertEqual(bundle.primary.action, "refund_full")
        self.assertEqual(len(bundle.primary.satisfied), 3)

    def test_opened_digital_rejected_definitively(self):
        # 数码已拆封是「确定性拒绝」：该条款本身判 not_applicable/reject，
        # 不能再因为缺「质量问题」事实而变成追问。
        bundle = self.engine.evaluate(facts(category="digital", opened=True))
        outcome = next(o for o in bundle.eligibility if o.clause_id == "RET-7D-003")
        self.assertEqual(outcome.status, PolicyStatus.NOT_APPLICABLE)
        self.assertEqual(outcome.action, "reject")

    def test_overdue_asks_for_quality_issue(self):
        # 超期但可能走质量问题退货 —— 缺事实时是 unknown 而不是拒绝
        f = facts(days_since_sign=12)
        f.pop("has_quality_issue")
        bundle = self.engine.evaluate(f)
        outcome = next(o for o in bundle.eligibility if o.clause_id == "RET-7D-001")
        self.assertEqual(outcome.status, PolicyStatus.UNKNOWN)
        self.assertIn("has_quality_issue", outcome.missing)

    def test_unknown_category_does_not_short_circuit(self):
        # 没查到订单时，类目未知的条款既不能判「不适用」，也不能把整个问题判死：
        # 引擎应当给出 unknown（缺 category），由编排层决定是追问还是只讲规则。
        f = facts(category="", days_since_sign=None)
        bundle = self.engine.evaluate(f)
        self.assertTrue(bundle.eligibility, "类目未知时应当仍能产出候选结论")
        self.assertTrue(any(o.status == PolicyStatus.UNKNOWN for o in bundle.eligibility))

    def test_effective_window(self):
        clause = self.engine.by_id("RET-7D-001")
        self.assertTrue(clause.effective_from <= TODAY.isoformat())


if __name__ == "__main__":
    unittest.main()
