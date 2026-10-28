"""文本工具测试：归一化、分词、金额/单位抽取、PII 脱敏。"""

import unittest

from ecom_cs.utils.text import (
    char_ngrams,
    contains_pii,
    extract_numbers,
    mask_pii,
    normalize,
    overlap_ratio,
    tokenize,
)


class TextTest(unittest.TestCase):
    def test_normalize_colloquial(self):
        self.assertEqual(normalize("退活咋弄"), "退货怎么弄")
        self.assertEqual(normalize("zfb退的钱"), "支付宝退的钱")
        self.assertEqual(normalize("ＡＢＣ"), "abc")

    def test_tokenize_keeps_chinese_bigrams(self):
        toks = tokenize("七天无理由退货")
        self.assertIn("退货", toks)
        self.assertIn("七天", toks)

    def test_extract_numbers_with_units(self):
        nums = extract_numbers("3 个工作日内退款 199 元，运费 0 元")
        pairs = {(n["unit"], n["value"]) for n in nums}
        self.assertIn(("个工作日", 3.0), pairs)
        self.assertIn(("元", 199.0), pairs)

    def test_pii_masking(self):
        text = "联系我 13812345678，身份证 110101199001011234"
        self.assertTrue(contains_pii(text))
        masked = mask_pii(text)
        self.assertNotIn("13812345678", masked)
        self.assertIn("*", masked)

    def test_overlap_ratio_is_bounded(self):
        ratio = overlap_ratio("七天无理由退货", "服装类商品支持七天无理由退货")
        self.assertGreater(ratio, 0.5)
        self.assertLessEqual(ratio, 1.0)

    def test_char_ngrams_short_text(self):
        # 文本短于 n 时退化为整串本身，不返回空集（否则短问题检索永远零召回）
        self.assertEqual(char_ngrams("ab", n=3), ["ab"])
        self.assertEqual(char_ngrams("abc", n=3), ["abc"])


if __name__ == "__main__":
    unittest.main()
