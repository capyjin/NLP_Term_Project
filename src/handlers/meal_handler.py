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
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

MEAL_URL  = "https://mobileadmin.cnu.ac.kr/food/index.jsp"
_REF_LINK = f"참고: {MEAL_URL}"

# TTL: 식단은 6시간마다 갱신 (주간 단위로 올라오므로 충분)
_MEAL_TTL_SECONDS = 6 * 3600

_DAY_NAMES = ["월", "화", "수", "목", "금", "토", "일"]
_DAY_MAP   = {n: i for i, n in enumerate(_DAY_NAMES)}

_MEAL_TYPE_MAP = {
    "아침": {"조식", "아침"},
    "점심": {"중식", "점심"},
    "저녁": {"석식", "저녁"},
}

_RESTAURANTS = [
    "제1학생회관", "제2학생회관", "제3학생회관", "제4학생회관",
    "생활과학대학", "교직원식당",
]

# 주간 의도 키워드 — today보다 먼저 검사해야 함
_KW_LASTWEEK = {"지난주", "지난 주", "저번주", "저번 주"}
_KW_THISWEEK = {"이번주", "이번 주", "주간", "이번주학식", "이번 주 학식"}
_KW_NEXTWEEK = {"다음주", "다음 주", "다음주학식", "다음 주 학식"}

# 단일 날짜 의도
_KW_TODAY    = {"오늘", "지금", "현재"}
_KW_TOMORROW = {"내일"}
_KW_DAYAFTER = {"모레"}


def _week_range(base: date) -> tuple[date, date]:
    """base 날짜가 속한 주의 월요일~일요일 반환."""
    monday = base - timedelta(days=base.weekday())
    sunday = monday + timedelta(days=6)
    return monday, sunday


class MealHandler:
    """
    식단 안내 핸들러.
    answer(question) → (answer_text, source)
    source: "meal_handler" | "meal_official"
    """

    def __init__(self, base_dir: Path):
        self._path     = base_dir / "data" / "raw" / "meal_menu.json"
        self._crawler  = base_dir / "src" / "crawling" / "meal_crawler.py"
        self._cache: Optional[dict] = None

    # ── TTL 캐시 ─────────────────────────────────────────────────────

    def _is_stale(self) -> bool:
        """파일이 없거나 TTL 초과 시 True."""
        if not self._path.exists():
            return True
        age = time.time() - self._path.stat().st_mtime
        return age > _MEAL_TTL_SECONDS

    def _try_refresh(self) -> None:
        """크롤러 실행으로 파일 갱신 시도. 실패해도 기존 파일 유지."""
        if not self._crawler.exists():
            print("[TTL] meal_crawler.py 없음 — 갱신 건너뜀")
            return
        print("[TTL] meal_menu.json stale — refreshing...")
        result = subprocess.run(
            [sys.executable, str(self._crawler)],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print("[TTL] meal refresh success")
            self._cache = None  # 인메모리 캐시 초기화 → 다음 _load에서 재로드
        else:
            err = (result.stderr or "").strip().splitlines()
            print(f"[TTL] meal refresh failed — using cached file")
            if err:
                print(f"      {err[-1][:100]}")

    # ── 데이터 로딩 ──────────────────────────────────────────────────

    def _load(self) -> Optional[list[dict]]:
        # TTL 체크: 캐시가 없거나 파일이 stale이면 크롤러 실행
        if self._cache is None and self._is_stale():
            self._try_refresh()
        elif self._cache is not None:
            # 인메모리 캐시 유효 (프로세스 재시작 전까지 유지)
            print("[TTL] meal using cached (in-memory)")
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

    # ── 의도 감지 ────────────────────────────────────────────────────

    def _detect_week_intent(self, q: str) -> Optional[str]:
        """
        주간 의도 우선 검사. today보다 먼저 호출해야 함.
        반환: "last_week" | "this_week" | "next_week" | None
        """
        nq = q.replace(" ", "")
        # 지난주 — "이번주" 보다 먼저 (부분 포함 방지)
        if any(k.replace(" ", "") in nq for k in _KW_LASTWEEK):
            return "last_week"
        if any(k.replace(" ", "") in nq for k in _KW_NEXTWEEK):
            return "next_week"
        if any(k.replace(" ", "") in nq for k in _KW_THISWEEK):
            return "this_week"
        return None

    def _parse_specific_date(self, q: str) -> Optional[date]:
        """6월 5일 / 6/5 / 06-05 형태 파싱."""
        today = date.today()
        patterns = [
            r"(\d{1,2})월\s*(\d{1,2})일",
            r"(\d{1,2})[/.](\d{1,2})(?!\d)",
            r"(\d{2})-(\d{2})(?!\d)",
        ]
        for pat in patterns:
            m = re.search(pat, q)
            if m:
                try:
                    return date(today.year, int(m.group(1)), int(m.group(2)))
                except ValueError:
                    continue
        return None

    def _target_date(self, q: str) -> date:
        """단일 날짜 추출. 주간 의도는 이미 제외된 상태에서 호출."""
        nq = q.replace(" ", "")
        today = date.today()

        if any(k in nq for k in _KW_DAYAFTER):
            return today + timedelta(days=2)
        if any(k in nq for k in _KW_TOMORROW):
            return today + timedelta(days=1)
        if any(k in nq for k in _KW_TODAY):
            return today

        for day_name, day_num in _DAY_MAP.items():
            if day_name + "요일" in q or day_name + "날" in q:
                diff = (day_num - today.weekday()) % 7
                return today + timedelta(days=diff)

        specific = self._parse_specific_date(q)
        if specific:
            return specific

        return today

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

    def _format_day(self, day_menus: list[dict], date_str: str,
                    mtype: Optional[str], rest: Optional[str]) -> str:
        """특정 날짜 메뉴 포매팅."""
        try:
            d = date.fromisoformat(date_str)
            day_name = _DAY_NAMES[d.weekday()]
            header = f"{date_str} ({day_name}) 식단\n"
        except ValueError:
            header = f"{date_str} 식단\n"

        filtered = day_menus
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

    def _format_week_range(self, menus: list[dict],
                           monday: date, sunday: date,
                           label: str) -> str:
        """주간 범위 내 날짜별 식단 요약."""
        week_dates = [
            (monday + timedelta(days=i)).isoformat()
            for i in range(7)
        ]
        # 해당 주에 속하는 저장 날짜만
        in_range = [d for d in self._available_dates(menus) if d in week_dates]

        if not in_range:
            all_dates = self._available_dates(menus)
            date_list = ", ".join(all_dates) if all_dates else "없음"
            return (
                f"{label} ({monday.isoformat()} ~ {sunday.isoformat()}) 식단 데이터가 없습니다.\n"
                f"현재 저장된 날짜: {date_list}\n"
                f"{_REF_LINK}"
            )

        lines = [f"{label} 식단 ({monday.isoformat()} ~ {sunday.isoformat()})\n"]
        for d_str in in_range:
            day_menus = [m for m in menus if m.get("date") == d_str]
            try:
                d = date.fromisoformat(d_str)
                day_name = _DAY_NAMES[d.weekday()]
            except ValueError:
                day_name = ""

            lines.append(f"[{d_str} {day_name}]")
            # 점심 우선, 없으면 다른 식사
            lunch = [m for m in day_menus if m.get("meal_type") == "점심"] or day_menus
            for m in lunch[:3]:
                r     = m.get("restaurant", "식당")
                items = m.get("menu", [])
                lines.append(f"  - {r}: " + " / ".join(items[:5]))
            lines.append("")

        lines.append(_REF_LINK)
        return "\n".join(lines).strip()

    def _format_no_data(self, target_str: str, menus: list[dict]) -> str:
        """해당 날짜 데이터 없을 때 응답."""
        available = self._available_dates(menus)
        date_list = ", ".join(available) if available else "없음"

        lines = [
            f"오늘({target_str}) 식단은 저장된 데이터에 없습니다.",
            "일요일·공휴일은 식당이 운영되지 않거나 식단이 게시되지 않을 수 있습니다.",
            "",
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
            lines += [
                "",
                f"가장 최근 식단인 {latest} ({day_name}) 메뉴를 안내드립니다.",
            ]
            for m in lunch[:3]:
                r     = m.get("restaurant", "식당")
                items = m.get("menu", [])
                lines.append(f"  [{r}] " + " / ".join(items[:5]))

        lines += ["", _REF_LINK]
        return "\n".join(lines)

    # ── 공개 API ─────────────────────────────────────────────────────

    def answer(self, question: str) -> tuple[str, str]:
        intent = self._question_intent(question)

        if intent == "location":
            return (
                f"충남대학교 학생식당 식단은 아래 공식 페이지에서 확인하세요.\n{_REF_LINK}",
                "meal_handler",
            )
        if intent == "hours":
            return (
                f"학생식당 운영 시간은 식당마다 다릅니다.\n{_REF_LINK}",
                "meal_handler",
            )

        menus = self._load()
        if not menus:
            return f"현재 식단 정보를 불러올 수 없습니다.\n{_REF_LINK}", "meal_official"

        today = date.today()

        # ── 주간 의도 우선 검사 (today보다 먼저) ──────────────────────
        week_intent = self._detect_week_intent(question)

        if week_intent == "last_week":
            last_monday = today - timedelta(days=today.weekday() + 7)
            last_sunday = last_monday + timedelta(days=6)
            text = self._format_week_range(menus, last_monday, last_sunday, "지난주")
            return text, "meal_handler"

        if week_intent == "next_week":
            next_monday = today - timedelta(days=today.weekday()) + timedelta(days=7)
            next_sunday = next_monday + timedelta(days=6)
            text = self._format_week_range(menus, next_monday, next_sunday, "다음주")
            return text, "meal_handler"

        if week_intent == "this_week":
            this_monday, this_sunday = _week_range(today)
            text = self._format_week_range(menus, this_monday, this_sunday, "이번주")
            return text, "meal_handler"

        # ── 단일 날짜 처리 ────────────────────────────────────────────
        target     = self._target_date(question)
        target_str = target.isoformat()
        day_menus  = [m for m in menus if m.get("date") == target_str]

        if not day_menus:
            return self._format_no_data(target_str, menus), "meal_handler"

        mtype = self._meal_type(question)
        rest  = self._restaurant(question)
        text  = self._format_day(day_menus, target_str, mtype, rest)
        text += f"\n\n{_REF_LINK}"
        return text, "meal_handler"
