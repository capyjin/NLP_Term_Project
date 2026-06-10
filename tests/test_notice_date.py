"""Regression tests for notice date extraction."""

import unittest

from src.handlers.notice_handler import _extract_date


class NoticeDateExtractionTest(unittest.TestCase):
    def test_registration_date_rejects_future_and_falls_back_to_past(self):
        content = "등록일 2999-12-31 본문 수정일 2020-05-20"
        self.assertEqual(_extract_date(content), "2020-05-20")

    def test_reception_date_rejects_future(self):
        self.assertEqual(_extract_date("접수기간 2999-12-31"), "")

    def test_invalid_date_is_ignored(self):
        self.assertEqual(_extract_date("등록일 2026-99-99"), "")

    def test_korean_dot_date_is_normalized(self):
        self.assertEqual(_extract_date("게시일 2020. 5. 7."), "2020-05-07")


if __name__ == "__main__":
    unittest.main()
