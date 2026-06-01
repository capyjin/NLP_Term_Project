"""
충남대학교 셔틀버스 시간표 크롤러
────────────────────────────────────
대상: https://plus.cnu.ac.kr/html/kr/sub05/sub05_050403.html
출력: data/raw/shuttle_bus.json

실행:
  python src/crawling/shuttle_crawler.py

특성:
  - 순수 정적 HTML <table> 구조 → Selenium 불필요
  - 학기당 1~2회만 변경되는 반고정 데이터
  - robots.txt: /html/ 경로는 주요 봇에 허용

파싱 전략:
  1. <table> 내 운행 시간표 파싱 (th/td 구조)
  2. 정류장 텍스트 → 구분자(→ ▶ ⟶) 기반 분리
  3. 파싱 실패 시 WebFetch로 확인된 known data fallback
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

BASE_DIR   = Path(__file__).parent.parent.parent
OUT_PATH   = BASE_DIR / "data" / "raw" / "shuttle_bus.json"
SHUTTLE_URL = "https://plus.cnu.ac.kr/html/kr/sub05/sub05_050403.html"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9",
    "Referer": "https://plus.cnu.ac.kr/",
}

# 조사 결과 확인된 알려진 데이터 (크롤링 실패 시 fallback)
# 출처: WebFetch 조사 (2026-06-01), 실제 페이지와 다를 수 있음
_KNOWN_ROUTES = [
    {
        "route":     "교내순환",
        "direction": "등교/하교",
        "stops": [
            "정심화국제문화회관", "사회과학대학입구", "서문(공동실험실습관앞)",
            "음악2호관앞", "공동동물실험센터", "체육관입구", "예술대학앞",
            "도서관앞", "학생생활관3거리", "농업생명과학대학앞", "동문주차장"
        ],
        "times": [
            "08:30", "09:30", "09:40", "10:30", "11:30",
            "13:30", "14:30", "15:30", "16:30", "17:30"
        ],
        "frequency": "1일 10회",
        "note":      "학기중 평일 운행 | 야간·주말·공휴일·방학 미운행",
        "source":    "known_data",
    },
    {
        "route":     "캠퍼스순환",
        "direction": "대덕↔보운",
        "stops":     ["대덕캠퍼스 출발", "보운캠퍼스 도착 후 회차"],
        "times":     ["08:10"],
        "frequency": "1일 1회 왕복 (대덕 08:10 출발, 보운 08:50 도착)",
        "note":      "학기중 평일 운행",
        "source":    "known_data",
    },
]


# ── 네트워크 ─────────────────────────────────────────────────────────────────

def _fetch(timeout: int = 15) -> Optional[str]:
    try:
        resp = requests.get(SHUTTLE_URL, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        if resp.encoding and resp.encoding.lower() in ("euc-kr", "cp949"):
            return resp.content.decode("euc-kr", errors="replace")
        resp.encoding = "utf-8"
        return resp.text
    except requests.RequestException as e:
        print(f"[shuttle_crawler] 접속 실패: {e}")
        return None


# ── 파싱 유틸 ─────────────────────────────────────────────────────────────────

def _extract_times(text: str) -> list[str]:
    """HH:MM 또는 H:MM 형식 시간 추출"""
    return re.findall(r"\b\d{1,2}:\d{2}\b", text)


def _extract_stops(text: str) -> list[str]:
    """
    정류장 목록 추출.
    구분자: →, ▶, ⟶, ▷, -, >
    """
    # 구분자로 분리
    parts = re.split(r"[\s]*[→▶⟶▷>]\s*", text)
    if len(parts) < 2:
        # 대시/줄바꿈 기반 시도
        parts = re.split(r"\s*[-–—]\s*|\n", text)

    stops = []
    for p in parts:
        p = p.strip()
        # 시간, 숫자, 짧은 기호 제거
        p = re.sub(r"\d{1,2}:\d{2}", "", p).strip()
        p = re.sub(r"[①②③④⑤⑥⑦⑧⑨⑩]", "", p).strip()
        if p and len(p) >= 2 and not re.match(r"^\d+$", p):
            stops.append(p)
    return stops


def _identify_route(text: str) -> Optional[str]:
    if "교내" in text or ("순환" in text and "캠퍼스" not in text):
        return "교내순환"
    if "캠퍼스" in text or "보운" in text or "대덕" in text:
        return "캠퍼스순환"
    return None


def _identify_direction(text: str) -> str:
    if "등교" in text:
        return "등교"
    if "하교" in text:
        return "하교"
    if "왕복" in text:
        return "왕복"
    return "운행"


# ── 메인 파싱 ─────────────────────────────────────────────────────────────────

def parse_shuttle(html: str) -> list[dict]:
    """HTML → 셔틀버스 노선 리스트"""
    soup = BeautifulSoup(html, "html.parser")
    routes_found: dict[str, dict] = {}

    for table in soup.find_all("table"):
        headers = [th.get_text(" ", strip=True) for th in table.find_all("th")]
        full_table_text = table.get_text(" ", strip=True)

        # 셔틀 관련 테이블인지 확인
        if not any(kw in full_table_text for kw in ["교내", "캠퍼스", "셔틀", "순환", "보운"]):
            continue

        rows = table.find_all("tr")
        for row in rows:
            row_text = row.get_text(" ", strip=True)
            route_name = _identify_route(row_text)
            if not route_name:
                continue

            cells   = row.find_all(["td", "th"])
            all_txt = " ".join(c.get_text(" ", strip=True) for c in cells)

            times   = _extract_times(all_txt)
            direction = _identify_direction(row_text)

            # 정류장: →가 포함된 셀 탐색
            stops = []
            for cell in cells:
                ct = cell.get_text(" ", strip=True)
                if "→" in ct or "▶" in ct or len(re.findall(r"[가-힣]+\s+[가-힣]+", ct)) > 2:
                    stops = _extract_stops(ct)
                    if stops:
                        break

            key = f"{route_name}_{direction}"
            if key not in routes_found:
                routes_found[key] = {
                    "route":     route_name,
                    "direction": direction,
                    "stops":     stops,
                    "times":     times,
                    "frequency": f"1일 {len(times)}회" if times else "미확인",
                    "note":      "학기중 평일 운행 | 야간·주말·공휴일·방학 미운행",
                    "source":    "crawled",
                }
            else:
                # 시간/정류장 병합
                existing = routes_found[key]
                if times and not existing["times"]:
                    existing["times"] = times
                if stops and not existing["stops"]:
                    existing["stops"] = stops

    return list(routes_found.values())


def crawl_and_save() -> bool:
    """크롤링 실행 및 저장. 성공 시 True."""
    print(f"[shuttle_crawler] 크롤링 시작: {SHUTTLE_URL}")
    html = _fetch()

    routes = []
    if html:
        routes = parse_shuttle(html)
        if routes:
            print(f"  [parser] 크롤링 성공: {len(routes)}노선")
        else:
            print("  [parser] 파싱 실패 — known_data fallback 사용")

    if not routes:
        # 크롤링/파싱 실패 시 조사된 known data 사용
        routes = _KNOWN_ROUTES
        print(f"  [known_data] 사전 조사된 데이터 사용: {len(routes)}노선")
        print("  ※ 실제 시간표와 다를 수 있음 — 공식 페이지 확인 필요")

    data = {
        "fetched_at": datetime.now().isoformat(),
        "source_url": SHUTTLE_URL,
        "routes":     routes,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[shuttle_crawler] 저장 완료: {OUT_PATH} ({len(routes)}노선)")
    return True


if __name__ == "__main__":
    crawl_and_save()
