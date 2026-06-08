"""
학사일정 핸들러
──────────────────
역할: data/raw/academic_calendar.json에서 학사일정 조회 및 응답 생성.

처리 대상 질문:
  "6월 학사일정 알려줘", "이번달 학사일정", "다음 학사일정 뭐야?"
  "계절학기 일정", "개강 언제야?" 등

answer(question) → (answer_text, source)
source: "academic_calendar_handler"
"""

import json
import re
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

_TTL_SECONDS = 24 * 3600
_PORTAL_URL = (
    "https://plus.cnu.ac.kr/_prog/academic_calendar/"
    "?site_dvs_cd=kr&menu_dvs_cd=05020101"
)

# 키워드 → title 검색어
_KEYWORD_MAP = {
    "수강신청": ["수강신청", "예비수강신청"],
    "수강정정": ["수강신청 확인 및 변경", "수강정정"],
    "수강정정일정": ["수강신청 확인 및 변경"],
    "계절학기일정": ["계절학기"],
    "계절학기": ["계절학기"],
    "개강": ["개강일"],
    "종강": ["종강", "방학"],
    "시험": ["시험"],
    "성적": ["성적"],
    "성적공시": ["성적발표"],
    "등록금": ["등록금"],
    "등록금납부일정": ["등록금"],
    "방학": ["방학"],
}


class AcademicCalendarHandler:
    """
    학사일정 핸들러.
    answer(question) → (answer_text, source)
    source: "academic_calendar_handler"
    """

    def __init__(self, base_dir: Path):
        self._path = base_dir / "data" / "raw" / "academic_calendar.json"
        self._cache: Optional[list] = None
        self._cache_time: float = 0.0

    # ── 캐시 ──────────────────────────────────────────────────────────

    def _load(self) -> list[dict]:
        """TTL 24h 캐시 적용 후 이벤트 목록 반환."""
        now = time.time()
        if self._cache is not None and (now - self._cache_time) < _TTL_SECONDS:
            return self._cache

        if not self._path.exists():
            return []

        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            self._cache = data.get("events", [])
            self._cache_time = now
            return self._cache
        except Exception:
            return []

    # ── 포맷팅 ────────────────────────────────────────────────────────

    def _fmt_event(self, e: dict) -> str:
        """단일 이벤트 포맷: '2026-06-03 ~ 2026-06-12  정기휴업일 수업결손 보충강의'"""
        s, en, title = e["start_date"], e["end_date"], e["title"]
        if s == en:
            return f"  • {s}  {title}"
        return f"  • {s} ~ {en}  {title}"

    def _no_data_msg(self) -> str:
        return (
            "저장된 학사일정이 없습니다.\n"
            f"포털에서 직접 확인하세요: {_PORTAL_URL}"
        )

    # ── 응답 유형별 메서드 ───────────────────────────────────────────

    def _answer_month_after(self, month: int) -> tuple[str, str]:
        """'N월 이후' 필터 (start_date >= 2026-N-01), 최대 8건."""
        events = self._load()
        if not events:
            return self._no_data_msg(), "academic_calendar_handler"

        threshold = f"2026-{month:02d}-01"
        filtered = [e for e in events if e["start_date"] >= threshold]
        filtered = sorted(filtered, key=lambda e: e["start_date"])[:8]

        if not filtered:
            return (
                f"{month}월 이후 학사일정을 찾을 수 없습니다.\n"
                f"포털에서 확인하세요: {_PORTAL_URL}"
            ), "academic_calendar_handler"

        lines = [f"충남대학교 {month}월 이후 학사일정 (최대 8건)\n"]
        for e in filtered:
            lines.append(self._fmt_event(e))
        lines.append(f"\n전체 학사일정: {_PORTAL_URL}")
        return "\n".join(lines), "academic_calendar_handler"

    def _answer_this_month(self) -> tuple[str, str]:
        """이번달 일정 필터."""
        events = self._load()
        if not events:
            return self._no_data_msg(), "academic_calendar_handler"

        today = date.today()
        month = today.month
        filtered = [e for e in events if e.get("month") == month]
        filtered = sorted(filtered, key=lambda e: e["start_date"])

        if not filtered:
            return (
                f"이번달({month}월) 학사일정이 없습니다.\n"
                f"포털에서 확인하세요: {_PORTAL_URL}"
            ), "academic_calendar_handler"

        lines = [f"충남대학교 {month}월 학사일정\n"]
        for e in filtered:
            lines.append(self._fmt_event(e))
        lines.append(f"\n전체 학사일정: {_PORTAL_URL}")
        return "\n".join(lines), "academic_calendar_handler"

    def _answer_next_upcoming(self) -> tuple[str, str]:
        """오늘 이후 가장 가까운 3건."""
        events = self._load()
        if not events:
            return self._no_data_msg(), "academic_calendar_handler"

        today_str = date.today().isoformat()
        upcoming = [e for e in events if e["start_date"] >= today_str]
        upcoming = sorted(upcoming, key=lambda e: e["start_date"])[:3]

        if not upcoming:
            return (
                "앞으로 예정된 학사일정을 찾을 수 없습니다.\n"
                f"포털에서 확인하세요: {_PORTAL_URL}"
            ), "academic_calendar_handler"

        lines = ["다가오는 학사일정 (최근 3건)\n"]
        for e in upcoming:
            lines.append(self._fmt_event(e))
        lines.append(f"\n전체 학사일정: {_PORTAL_URL}")
        return "\n".join(lines), "academic_calendar_handler"

    def _answer_keyword(self, keywords: list[str], label: str) -> tuple[str, str]:
        """키워드가 title에 포함된 일정 검색."""
        events = self._load()
        if not events:
            return self._no_data_msg(), "academic_calendar_handler"

        matched = [
            e for e in events
            if any(kw in e["title"] for kw in keywords)
        ]
        matched = sorted(matched, key=lambda e: e["start_date"])

        if not matched:
            return (
                f"'{label}' 관련 학사일정을 찾을 수 없습니다.\n"
                f"포털에서 확인하세요: {_PORTAL_URL}"
            ), "academic_calendar_handler"

        lines = [f"'{label}' 관련 학사일정\n"]
        for e in matched:
            lines.append(self._fmt_event(e))
        lines.append(f"\n전체 학사일정: {_PORTAL_URL}")
        return "\n".join(lines), "academic_calendar_handler"

    def _answer_specific_month(self, month: int) -> tuple[str, str]:
        """특정 N월 일정."""
        events = self._load()
        if not events:
            return self._no_data_msg(), "academic_calendar_handler"

        filtered = [e for e in events if e.get("month") == month]
        filtered = sorted(filtered, key=lambda e: e["start_date"])

        if not filtered:
            return (
                f"{month}월 학사일정이 없습니다.\n"
                f"포털에서 확인하세요: {_PORTAL_URL}"
            ), "academic_calendar_handler"

        lines = [f"충남대학교 {month}월 학사일정\n"]
        for e in filtered:
            lines.append(self._fmt_event(e))
        lines.append(f"\n전체 학사일정: {_PORTAL_URL}")
        return "\n".join(lines), "academic_calendar_handler"

    def _answer_default(self) -> tuple[str, str]:
        """기본: 이번달 + 다음달 일정."""
        events = self._load()
        if not events:
            return self._no_data_msg(), "academic_calendar_handler"

        today = date.today()
        this_m = today.month
        next_m = this_m % 12 + 1

        filtered = [e for e in events if e.get("month") in (this_m, next_m)]
        filtered = sorted(filtered, key=lambda e: e["start_date"])

        if not filtered:
            return (
                f"{this_m}월~{next_m}월 학사일정이 없습니다.\n"
                f"포털에서 확인하세요: {_PORTAL_URL}"
            ), "academic_calendar_handler"

        lines = [f"충남대학교 {this_m}월~{next_m}월 학사일정\n"]
        for e in filtered:
            lines.append(self._fmt_event(e))
        lines.append(f"\n전체 학사일정: {_PORTAL_URL}")
        return "\n".join(lines), "academic_calendar_handler"

    # ── 메인 진입점 ───────────────────────────────────────────────────

    def answer(self, question: str) -> tuple[str, str]:
        """
        질문 분석 → 응답 유형 감지.

        유형 1: "N월 이후" + 일정 → _answer_month_after(N)
        유형 2: "이번달" → _answer_this_month()
        유형 3: "다음 학사일정" / "다음에" → _answer_next_upcoming()
        유형 4: 키워드 검색 → _answer_keyword()
        유형 5: "N월 학사일정" → _answer_specific_month(N)
        유형 6: 기본 → _answer_default()
        """
        nq = question.replace(" ", "")

        # 유형 1: "N월이후" + 일정 키워드
        m = re.search(r"(\d+)월이후", nq)
        if m:
            return self._answer_month_after(int(m.group(1)))

        # 유형 2: 이번달
        if any(k in nq for k in ("이번달", "이번월", "금월")):
            return self._answer_this_month()

        # 유형 3: 다음 학사일정 (가장 가까운 것)
        if any(k in nq for k in ("다음학사일정", "다음일정", "다음에뭐", "앞으로일정", "가장가까운일정")):
            return self._answer_next_upcoming()

        # 유형 4: 키워드 검색
        for kw, search_terms in _KEYWORD_MAP.items():
            if kw in nq:
                return self._answer_keyword(search_terms, kw)

        # 유형 5: "N월 학사일정"
        m2 = re.search(r"(\d+)월", nq)
        if m2 and any(k in nq for k in ("학사일정", "일정", "학사")):
            return self._answer_specific_month(int(m2.group(1)))

        # 유형 6: 기본 (이번달 + 다음달)
        return self._answer_default()
