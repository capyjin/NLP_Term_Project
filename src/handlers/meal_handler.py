"""
식단 안내 핸들러
─────────────────
우선순위:
  1. 크롤링 결과 (meal_crawler.py 실행 → data/raw/meal_menu.json)
  2. 수동 입력 JSON (data/raw/meal_menu.json)
  3. 공식 사이트 안내 fallback

중요:
  - 허위 메뉴 절대 생성 금지
  - 데이터 없을 때는 저장된 날짜 안내 + 최신 메뉴 표시
  - 식단 질문을 RAG로 넘기지 않음 (공지사항 노출 방지)
"""

import json
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

MEAL_URL  = "https://mobileadmin.cnu.ac.kr/food/index.jsp"
_REF_LINK = f"참고: {MEAL_URL}"

_DAY_NAMES = ["월", "화", "수", "목", "금", "토", "일"]
_DAY_MAP   = {n: i for i, n in enumerate(_DAY_NAMES)}

_MEAL_KW_TODAY    = {"오늘", "지금", "현재"}
_MEAL_KW_TOMORROW = {"내일"}
_MEAL_KW_DAYAFTER = {"모레"}
_MEAL_KW_WEEK     = {"이번주", "이번 주", "주간"}
_MEAL_KW_NEXTWEEK = {"다음주", "다음 주"}

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

    def _available_dates(self, menus: list[dict]) -> list[str]:
        return sorted(set(m.get("date", "") for m in menus if m.get("date")))

    # ── 질문 분석 ────────────────────────────────────────────────────

    def _parse_specific_date(self, q: str) -> Optional[date]:
        """
        "6월 5일", "6/5", "06-05", "6.5" 형태의 특정 날짜 파싱.
        연도 없으면 올해 기준.
        """
        today = date.today()
        patterns = [
            r"(\d{1,2})월\s*(\d{1,2})일",   # 6월 5일
            r"(\d{1,2})[/.](\d{1,2})(?!\d)", # 6/5, 6.5
            r"(\d{2})-(\d{2})(?!\d)",         # 06-05
        ]
        for pat in patterns:
            m = re.search(pat, q)
            if m:
                try:
                    month, day = int(m.group(1)), int(m.group(2))
                    return date(today.year, month, day)
                except ValueError:
                    continue
        return None

    def _target_date(self, q: str) -> Optional[date]:
        """질문에서 목표 날짜 추출. None = 이번 주 전체 / 다음 주."""
        nq = q.replace(" ", "")
        today = date.today()

        if any(k in nq for k in _MEAL_KW_NEXTWEEK):
            return None  # 다음 주
        if any(k in nq for k in _MEAL_KW_WEEK):
            return None  # 이번 주 전체 → answer()에서 별도 처리

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

        # 특정 날짜: "6월 5일"
        specific = self._parse_specific_date(q)
        if specific:
            return specific

        return today  # 기본: 오늘

    def _is_week_query(self, q: str) -> bool:
        nq = q.replace(" ", "")
        return any(k in nq for k in _MEAL_KW_WEEK)

    def _is_nextweek_query(self, q: str) -> bool:
        nq = q.replace(" ", "")
        return any(k in nq for k in _MEAL_KW_NEXTWEEK)

    def _meal_type(self, q: str) -> Optional[str]:
        nq = q.replace(" ", "")
        for mtype, kws in _MEAL_TYPE_MAP.items():
            if any(k in nq for k in kws):
                return mtype
        return None

    def _restaurant(self, q: str) -> Optional[str]:
        for r in _RESTAURANTS:
            if r in q:
                return r
        m = re.search(r"제(\d)(?:학생회관)?", q)
        if m:
            return f"제{m.group(1)}학생회관"
        return None

    def _question_intent(self, q: str) -> str:
        nq = q.replace(" ", "")
        if any(k in nq for k in ("어디서", "어딜", "홈페이지", "링크", "사이트", "앱")):
            return "location"
        if any(k in nq for k in ("몇시", "영업시간", "운영시간")):
            return "hours"
        return "menu"

    # ── 응답 포매팅 ──────────────────────────────────────────────────

    def _format_day(self, menus: list[dict], date_str: str,
                    mtype: Optional[str], rest: Optional[str]) -> str:
        """특정 날짜 메뉴 포매팅."""
        try:
            d = date.fromisoformat(date_str)
            day_str = _DAY_NAMES[d.weekday()]
            header = f"{date_str} ({day_str}) 식단\n"
        except ValueError:
            header = f"{date_str} 식단\n"

        filtered = menus
        if mtype:
            typed = [m for m in filtered if m.get("meal_type") == mtype]
            if typed:
                filtered = typed
        else:
            lunch = [m for m in filtered if m.get("meal_type") == "점심"]
            if lunch:
                filtered = lunch

        if rest:
            rf = [m for m in filtered if m.get("restaurant") == rest]
            if rf:
                filtered = rf

        if not filtered:
            return header + "해당 조건의 메뉴 정보가 없습니다."

        lines = [header]
        for m in filtered[:4]:
            r     = m.get("restaurant", "식당")
            mt    = m.get("meal_type", "")
            price = m.get("price", 0)
            items = m.get("menu", [])
            h = f"  [{r}] {mt}"
            if price:
                h += f" ({price:,}원)"
            lines.append(h)
            if items:
                lines.append("    " + " / ".join(items[:6]))
        return "\n".join(lines)

    def _format_week(self, menus: list[dict], available: list[str]) -> str:
        """이번 주 저장된 전체 날짜 요약."""
        lines = ["저장된 식단 전체 요약\n"]
        for d_str in available:
            day_menus = [m for m in menus if m.get("date") == d_str]
            lunch = [m for m in day_menus if m.get("meal_type") == "점심"] or day_menus[:1]
            try:
                d = date.fromisoformat(d_str)
                day_name = _DAY_NAMES[d.weekday()]
            except ValueError:
                day_name = ""
            lines.append(f"{d_str} ({day_name})")
            for m in lunch[:2]:
                r     = m.get("restaurant", "식당")
                items = m.get("menu", [])
                lines.append(f"  [{r}] " + " / ".join(items[:4]))
            lines.append("")
        lines.append(_REF_LINK)
        return "\n".join(lines).strip()

    def _format_no_data(self, target_str: str, menus: list[dict]) -> str:
        """해당 날짜 데이터 없을 때 응답."""
        available = self._available_dates(menus)
        date_list = ", ".join(available) if available else "없음"

        lines = [
            f"오늘({target_str}) 식단은 저장된 데이터에 없습니다.",
            f"일요일·공휴일은 식당이 운영되지 않거나 식단이 게시되지 않을 수 있습니다.",
            f"",
            f"현재 확인 가능한 식단 날짜: {date_list}",
        ]

        if available:
            latest = available[-1]
            latest_menus = [m for m in menus if m.get("date") == latest]
            lunch = [m for m in latest_menus if m.get("meal_type") == "점심"] or latest_menus[:3]
            try:
                d = date.fromisoformat(latest)
                day_name = _DAY_NAMES[d.weekday()]
            except ValueError:
                day_name = ""
            lines.append(f"")
            lines.append(f"가장 최근 식단인 {latest} ({day_name}) 메뉴를 안내드립니다.")
            for m in lunch[:3]:
                r     = m.get("restaurant", "식당")
                items = m.get("menu", [])
                lines.append(f"  [{r}] " + " / ".join(items[:5]))

        lines.append("")
        lines.append(_REF_LINK)
        return "\n".join(lines)

    # ── 공개 API ─────────────────────────────────────────────────────

    def answer(self, question: str) -> tuple[str, str]:
        intent = self._question_intent(question)

        if intent == "location":
            return (
                "충남대학교 학생식당 식단은 아래 공식 페이지에서 확인하세요.\n"
                f"{_REF_LINK}",
                "meal_handler",
            )

        if intent == "hours":
            return (
                f"학생식당 운영 시간은 식당마다 다릅니다.\n{_REF_LINK}",
                "meal_handler",
            )

        menus = self._load()
        if not menus:
            return (
                "현재 식단 정보를 불러올 수 없습니다.\n"
                f"{_REF_LINK}",
                "meal_official",
            )

        available = self._available_dates(menus)

        # 다음 주 질문
        if self._is_nextweek_query(question):
            return (
                "다음 주 식단 정보는 아직 게시되지 않았습니다.\n"
                f"현재 확인 가능한 날짜: {', '.join(available)}\n"
                f"{_REF_LINK}",
                "meal_handler",
            )

        # 이번 주 전체 질문
        if self._is_week_query(question):
            if not available:
                return f"이번 주 식단 데이터가 없습니다.\n{_REF_LINK}", "meal_official"
            return self._format_week(menus, available), "meal_handler"

        # 특정 날짜 / 오늘 / 내일 등
        target = self._target_date(question)
        mtype  = self._meal_type(question)
        rest   = self._restaurant(question)

        target_str = target.isoformat() if target else date.today().isoformat()
        day_menus  = [m for m in menus if m.get("date") == target_str]

        if not day_menus:
            return self._format_no_data(target_str, menus), "meal_handler"

        text = self._format_day(day_menus, target_str, mtype, rest)
        text += f"\n\n{_REF_LINK}"
        return text, "meal_handler"
