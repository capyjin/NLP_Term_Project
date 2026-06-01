"""
통학/셔틀버스 안내 핸들러
───────────────────────────
우선순위:
  1. 크롤링 결과 (shuttle_crawler.py → data/raw/shuttle_bus.json)
  2. 수동 입력 JSON (data/raw/shuttle_bus.json)
  3. 사전 조사된 static 정보 (known_data fallback)
  4. 공식 사이트 안내

중요:
  - 허위 시간표/정류장 절대 생성 금지
  - known_data 사용 시 "※ 변경될 수 있음" 명시
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

SHUTTLE_URL = "https://plus.cnu.ac.kr/html/kr/sub05/sub05_050403.html"
_OFFICIAL   = f"🔗 {SHUTTLE_URL}"

# 2026년 6월 조사 기준 알려진 시간표 (크롤링/파일 모두 없을 때 사용)
_KNOWN_ROUTES = [
    {
        "route":     "교내순환",
        "direction": "등교/하교",
        "stops": [
            "정심화국제문화회관", "사회과학대학입구", "서문(공동실험실습관앞)",
            "음악2호관앞", "공동동물실험센터", "체육관입구", "예술대학앞",
            "도서관앞", "학생생활관3거리", "농업생명과학대학앞", "동문주차장",
        ],
        "times": [
            "08:30", "09:30", "09:40", "10:30", "11:30",
            "13:30", "14:30", "15:30", "16:30", "17:30",
        ],
        "frequency": "1일 10회",
        "note":      "학기중 평일 운행 | 야간·주말·공휴일·방학 미운행",
    },
    {
        "route":     "캠퍼스순환",
        "direction": "대덕↔보운",
        "stops":     ["대덕캠퍼스 출발", "보운캠퍼스 도착 후 회차"],
        "times":     ["08:10"],
        "frequency": "1일 1회 왕복 (대덕 08:10 → 보운 08:50)",
        "note":      "학기중 평일 운행",
    },
]


class ShuttleHandler:
    """
    통학/셔틀버스 안내 핸들러.

    answer(question) → (answer_text, source)
    source: "shuttle_handler" | "shuttle_known" | "shuttle_official"
    """

    def __init__(self, base_dir: Path):
        self._path  = base_dir / "data" / "raw" / "shuttle_bus.json"
        self._cache: Optional[dict] = None

    # ── 데이터 로딩 ──────────────────────────────────────────────────

    def _load(self) -> tuple[list[dict], str]:
        """(routes, source)"""
        if self._cache:
            return self._cache.get("routes", []), "file"
        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as f:
                    data = json.load(f)
                self._cache = data
                return data.get("routes", []), "file"
            except Exception as e:
                print(f"[shuttle_handler] 로드 오류: {e}")
        return _KNOWN_ROUTES, "known"

    # ── 질문 분석 ────────────────────────────────────────────────────

    def _intent(self, q: str) -> str:
        nq = q.replace(" ", "")
        if any(k in nq for k in ("정류장", "어디서타", "어디타", "정거장", "어디서 타", "어디 타")):
            return "stops"
        if any(k in nq for k in ("시간표", "몇시", "언제", "첫차", "막차", "배차간격", "마지막")):
            return "schedule"
        if any(k in nq for k in ("운행하나", "다니나", "있나요", "정상운행", "운행여부", "운행하나요")):
            return "operation"
        if any(k in nq for k in ("노선", "어디까지", "경유", "정차", "거쳐")):
            return "route"
        return "general"

    def _route_filter(self, q: str) -> Optional[str]:
        nq = q.replace(" ", "")
        if "교내" in nq:
            return "교내순환"
        if "캠퍼스" in nq or "보운" in nq or "대덕" in nq:
            return "캠퍼스순환"
        return None  # 전체

    # ── 응답 포매팅 ──────────────────────────────────────────────────

    @staticmethod
    def _fmt_route(r: dict) -> str:
        name   = r.get("route", "")
        direct = r.get("direction", "")
        times  = r.get("times", [])
        stops  = r.get("stops", [])
        note   = r.get("note", "")
        freq   = r.get("frequency", "")

        lines = [f"🚌 {name} ({direct})"]
        if times:
            lines.append(f"  ⏰ 운행 시간: {' / '.join(times)}")
        if freq:
            lines.append(f"  🔄 운행 횟수: {freq}")
        if stops:
            stop_str = " → ".join(stops[:9])
            if len(stops) > 9:
                stop_str += " → ..."
            lines.append(f"  🗺️ 정류장: {stop_str}")
        if note:
            lines.append(f"  📋 {note}")
        return "\n".join(lines)

    @staticmethod
    def _is_weekend() -> bool:
        return datetime.now().weekday() >= 5

    # ── 공개 API ─────────────────────────────────────────────────────

    def answer(self, question: str) -> tuple[str, str]:
        """
        Returns: (answer_text, source)
        source: "shuttle_handler" | "shuttle_known" | "shuttle_official"
        """
        intent      = self._intent(question)
        route_name  = self._route_filter(question)
        routes, src = self._load()

        known_note = (
            "\n\n※ 2026년 1학기 기준 정보입니다. 변경될 수 있으니 공식 페이지를 확인하세요."
            if src == "known" else ""
        )
        source_tag  = "shuttle_known" if src == "known" else "shuttle_handler"

        # 노선 필터
        filtered = [r for r in routes if not route_name or route_name in r.get("route", "")]
        if not filtered:
            filtered = routes

        # ── 운행 여부 ────────────────────────────────────────────
        if intent == "operation":
            if self._is_weekend():
                return (
                    "🚫 오늘은 주말이므로 셔틀버스가 **운행하지 않습니다**.\n"
                    "셔틀버스는 학기중 평일에만 운행합니다.\n"
                    f"📌 상세 정보: {SHUTTLE_URL}",
                    source_tag,
                )
            first = filtered[0].get("times", ["?"])[0] if filtered else "?"
            return (
                f"✅ 평일에는 셔틀버스가 정상 운행합니다.\n\n"
                f"첫차: {first}\n"
                f"운행: 학기중 평일 (야간·주말·공휴일·방학 미운행)\n"
                f"📌 시간표 전체: {SHUTTLE_URL}"
                + known_note,
                source_tag,
            )

        # ── 정류장 ───────────────────────────────────────────────
        if intent == "stops":
            stop_lines = []
            for r in filtered:
                stops = r.get("stops", [])
                if stops:
                    stop_lines.append(f"• {r['route']}: {' → '.join(stops)}")
            if stop_lines:
                return (
                    "🗺️ 셔틀버스 정류장\n\n"
                    + "\n".join(stop_lines)
                    + f"\n\n📌 공식 지도: {SHUTTLE_URL}"
                    + known_note,
                    source_tag,
                )

        # ── 시간표 / 노선 / 일반 ────────────────────────────────
        lines = []
        if intent == "schedule":
            lines.append("⏰ 셔틀버스 운행 시간표\n")
        elif intent == "route":
            lines.append("🗺️ 셔틀버스 노선 안내\n")
        else:
            lines.append("🚌 충남대학교 셔틀버스 안내\n")

        for r in filtered[:3]:
            lines.append(self._fmt_route(r))
            lines.append("")

        lines.append(f"📌 공식 페이지: {SHUTTLE_URL}")
        return "\n".join(lines).strip() + known_note, source_tag
