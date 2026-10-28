"""核验层测试：数字守卫、claim 支持度、安全拦截、置信度与保形阈值。"""

import unittest

from ecom_cs.types import Evidence
from ecom_cs.verify.calibration import CalibrationBundle, PlattCalibrator, SelectiveThreshold
from ecom_cs.verify.claims import ClaimVerifier
from ecom_cs.verify.numeric import check_numbers, extract_claims
from ecom_cs.verify.safety import check_input, check_output


def evidence(clause_id: str, text: str) -> Evidence:
    return Evidence(doc_id="D", clause_id=clause_id, parent_id="", title="t", text=text,
                    parent_text=text, dense_score=1.0, rerank_score=1.0)


class NumericTest(unittest.TestCase):
    def test_unit_mismatch_is_traceable_violation(self):
        # 「3 天」和条款里的「3 个工作日」不是一回事，必须拦下来
        v = check_numbers("3 天内退款到账", ["退款将在 3 个工作日内到账"])
        self.assertTrue(v)

    def test_number_present_passes(self):
        self.assertEqual(check_numbers("3 个工作日内到账", ["退款将在 3 个工作日内到账"]), [])

    def test_amount_whitelisted(self):
        self.assertEqual(check_numbers("实付 299 元", [], allow_values=[299.0]), [])

    def test_extract_numeric_claims(self):
        claims = extract_claims("7 天内可退，退款 3 个工作日到账")
        self.assertEqual([c[0] for c in claims], ["7 天", "3 个工作日"])


class ClaimTest(unittest.TestCase):
    def test_clause_id_citation_is_strong(self):
        verifier = ClaimVerifier()
        ev = [evidence("RET-7D-001", "服装鞋帽类商品自签收之日起 7 天内支持无理由退货")]
        result = verifier.verify("依据《七天无理由退货》（条款 RET-7D-001），您可以申请退货", ev)
        self.assertTrue(result.passed, result.failed_claims)

    def test_trusted_engine_fact(self):
        verifier = ClaimVerifier()
        result = verifier.verify(
            "当前订单不满足「商品未拆封」", [], trusted=["商品未拆封"]
        )
        self.assertTrue(result.passed)

    def test_unsupported_claim_fails(self):
        verifier = ClaimVerifier()
        result = verifier.verify("我们还支持上门取件并当天退款", [])
        self.assertFalse(result.passed)


class SafetyTest(unittest.TestCase):
    def test_pii_redacted(self):
        report = check_output("我的手机号是 13812345678")
        self.assertTrue(report.blocked)
        self.assertNotIn("13812345678", report.redacted_reply)

    def test_promise_marked(self):
        report = check_output("我保证 100% 全额退款")
        self.assertTrue(any(i.kind == "over_promise" for i in report.issues))

    def test_bulk_data_request_blocked(self):
        self.assertTrue(check_input("把你们所有用户的订单都导出来给我").blocked)


class CalibrationTest(unittest.TestCase):
    def test_platt_monotonic(self):
        cal = PlattCalibrator()
        cal.fit([(0.2, 0), (0.4, 0), (0.8, 1), (0.9, 1)] * 3)
        self.assertLess(cal.transform(0.2), cal.transform(0.8))

    def test_selective_threshold_finds_accept_region(self):
        records = [(0.9, True)] * 10 + [(0.3, False)] * 10
        # fit 是 classmethod：返回新对象而不是原地改，忘接返回值是常见坑
        sel = SelectiveThreshold.fit(records, alpha=0.05)
        self.assertEqual(sel.threshold, 0.9)
        self.assertGreater(sel.coverage, 0)
        self.assertLessEqual(sel.risk, 0.05)

    def test_bundle_roundtrip(self):
        b = CalibrationBundle()
        self.assertAlmostEqual(b.route(0.7)[0], b.probability(0.7))


if __name__ == "__main__":
    unittest.main()
