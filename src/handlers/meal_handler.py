"""
식단 안내 핸들러
─────────────────
우선순위:
  1. 크롤링 결과 (meal_crawler.py 실행 → data/raw/meal_menu.json)
  2. 수동 입력 JSON (data/raw/meal_menu.json)
  3. 공식 사이트 안내 fallback

중요:
  - 허위 메뉴 절대 생성 금지
  - 데이터 없을 때는 반드시 공식 URL 안내
  - 식단 질문을 RAG로 넘기지 않음 (공지사항 노출 방지)
"""

import json
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

MEAL_URL    = "https://mobileadmin.cnu.ac.kr/food/index.jsp"
_OFFICIAL   = f"🔗 {MEAL_URL}"
_NO_DATA_MSG = (
    "현재 식단 정보를 불러올 수 없습니다.\n"
    "충남대학교 학생식당 메뉴는 공식 페이지에서 확인하세요:\n"
    + _OFFICIAL
)

_DAY_NAMES = ["월", "화", "수", "목", "금", "토", "일"]
_DAY_MAP   = {n: i for i, n in enumerate(_DAY_NAMES)}

_MEAL_KW_TODAY    = {"오늘", "지금", "현재"}
_MEAL_KW_TOMORROW = {"내일"}
_MEAL_KW_DAYAFTER = {"모레"}
_MEAL_KW_WEEK     = {"이번주", "이번 주", "주간"}
_MEAL_KW_NEXTWEEK = {"다음주", "다음 주", "다음주"}

_MEAL_TYPE_MAP = {
    "아침": {"조식", "아침"},
    "점심": {"중식", "점심"},
    "저녁": {"석식", "저녁"},
}

_RESTAURANTS = [
    "제1학생회관", "제2학생회관", "제3학생회관", "제4학생회관",
    "생활과학대학", "교직원식당",
]


class MealHandler:
    """
    식단 안내 핸들러.

    answer(question) → (answer_text, source)
    source: "meal_handler" | "meal_official"
    """

    def __init__(self, base_dir: Path):
        self._path  = base_dir / "data" / "raw" / "meal_menu.json"
        self._cache: Optional[dict] = None

    # ── 데이터 로딩 ──────────────────────────────────────────────────

    def _load(self) -> Optional[list[dict]]:
        if self._cache:
            return self._cache.get("menus")
        if not self._path.exists():
            return None
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            self._cache = data
            return data.get("menus", [])
        except Exception as e:
            print(f"[meal_handler] 로드 오류: {e}")
            return None

    # ── 질문 분석 ────────────────────────────────────────────────────

    def _target_date(self, q: str) -> Optional[date]:
        """질문에서 목표 날짜 추출. None = 다음 주 (데이터 없을 가능성)."""
        nq = q.replace(" ", "")
        today = date.today()

        if any(k in nq for k in _MEAL_KW_NEXTWEEK):
            return None
        if any(k in nq for k in _MEAL_KW_DAYAFTER):
            return today + timedelta(days=2)
        if any(k in nq for k in _MEAL_KW_TOMORROW):
            return today + timedelta(days=1)
        if any(k in nq for k in _MEAL_KW_TODAY):
            return today

        # 요일 지정: "월요일 학식"
        for day_name, day_num in _DAY_MAP.items():
            if day_name + "요일" in q or day_name + "날" in q:
                diff = (day_num - today.weekday()) % 7
                return today + timedelta(days=diff)

        return today  # 기본: 오늘

    def _meal_type(self, q: str) -> Optional[str]:
        nq = q.replace(" ", "")
        for mtype, kws in _MEAL_TYPE_MAP.items():
            if any(k in nq for k in kws):
                return mtype
        return None  # 미지정 → 점심 우선

    def _restaurant(self, q: str) -> Optional[str]:
        for r in _RESTAURANTS:
            if r in q:
                return r
            # "제1", "1관" 약칭 처리
            m = re.search(r"제(\d)(?:학생회관)?", q)
            if m:
                return f"제{m.group(1)}학생회관"
        return None

    def _question_intent(self, q: str) -> str:
        nq = q.replace(" ", "")
        if any(k in nq for k in ("어디서", "어디서봐", "어디", "어딜", "홈페이지", "링크", "사이트")):
            return "location"
        if any(k in nq for k in ("몇시", "영업시간", "운영시간", "몇 시")):
            return "hours"
        return "menu"

    # ── 응답 포매팅 ──────────────────────────────────────────────────

    def _format(
        self,
        menus: list[dict],
        target: Optional[date],
        mtype: Optional[str],
        rest:   Optional[str],
    ) -> str:
        today = date.today()

        # 날짜 필터
        if target:
            filtered = [m for m in menus if m.get("date") == target.isoformat()]
        else:
            # 다음 주 → 데이터 없음 안내
            return (
                "다음 주 식단 정보는 아직 게시되지 않았습니다.\n"
                "최신 식단은 공식 페이지에서 확인하세요:\n"
                + _OFFICIAL
            )

        if not filtered and target:
            # 데이터에 해당 날짜가 없음
            date_diff = (target - today).days
            if date_diff < 0:
                return f"{target.isoformat()} 식단 데이터는 제공되지 않습니다.\n" + _OFFICIAL
            return (
                f"{target.isoformat()} 식단 정보를 아직 불러오지 못했습니다.\n"
                "최신 식단은 공식 페이지에서 확인하세요:\n"
                + _OFFICIAL
            )

        # 식사 종류 필터
        if mtype:
            typed = [m for m in filtered if m.get("meal_type") == mtype]
            if typed:
                filtered = typed
        else:
            lunch = [m for m in filtered if m.get("meal_type") == "점심"]
            if lunch:
                filtered = lunch

        # 식당 필터
        if rest:
            rf = [m for m in filtered if m.get("restaurant") == rest]
            if rf:
                filtered = rf

        if not filtered:
            return _NO_DATA_MSG

        # 날짜 헤더
        date_str = filtered[0].get("date", "")
        lines = []
        try:
            d = date.fromisoformat(date_str)
            day_str = _DAY_NAMES[d.weekday()]
            lines.append(f"📅 {date_str} ({day_str}) 식단 안내\n")
        except ValueError:
            pass

        # 식당별 메뉴
        for m in filtered[:4]:
            r     = m.get("restaurant", "식당")
            mt    = m.get("meal_type", "")
            price = m.get("price", 0)
            items = m.get("menu", [])
            header = f"🍱 {r} {mt}"
            if price:
                header += f"  ({price:,}원)"
            lines.append(header)
            if items:
                lines.append("  " + " / ".join(items[:7]))
            lines.append("")

        lines.append(f"📌 메뉴 전체 보기: {MEAL_URL}")
        return "\n".join(lines).strip()

    # ── 공개 API ─────────────────────────────────────────────────────

    def answer(self, question: str) -> tuple[str, str]:
        """
        Returns: (answer_text, source)
        source: "meal_handler" | "meal_official"
        """
        intent = self._question_intent(question)

        if intent == "location":
            return (
                "충남대학교 학생식당 식단은 아래 공식 페이지에서 확인하세요:\n"
                f"{_OFFICIAL}\n\n"
                "또는 Google Play에서 '충남대학교 식당메뉴' 앱을 이용하실 수 있습니다.",
                "meal_handler",
            )

        if intent == "hours":
            return (
                "학생식당 운영 시간은 식당마다 다릅니다. "
                "정확한 운영 시간은 아래 공식 페이지를 확인해주세요:\n"
                + _OFFICIAL,
                "meal_handler",
            )

        # 메뉴 조회
        menus = self._load()
        if not menus:
            return _NO_DATA_MSG, "meal_official"

        target = self._target_date(question)
        mtype  = self._meal_type(question)
        rest   = self._restaurant(question)
        text   = self._format(menus, target, mtype, rest)

        source = "meal_handler" if text != _NO_DATA_MSG else "meal_official"
        return text, source
