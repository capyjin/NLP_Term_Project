"""
충남대학교 학사일정 크롤러
──────────────────────────
URL: https://plus.cnu.ac.kr/_prog/academic_calendar/?site_dvs_cd=kr&menu_dvs_cd=05020101&year=2026
month 파라미터 없이 연도 전체 조회 → calen_box 구조 파싱

저장: data/raw/academic_calendar.json
실행: python src/crawling/academic_calendar_crawler.py
"""

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_PATH = BASE_DIR / "data" / "raw" / "academic_calendar.json"

_BASE_URL = "https://plus.cnu.ac.kr/_prog/academic_calendar/"
_YEAR = 2026
_TTL_SECONDS = 24 * 3600

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# 월 이름 → 번호
_MONTH_KR = {
    "01월": 1, "02월": 2, "03월": 3, "04월": 4,
    "05월": 5, "06월": 6, "07월": 7, "08월": 8,
    "09월": 9, "10월": 10, "11월": 11, "12월": 12,
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}


def _parse_date(date_str: str, year: int) -> tuple[str, str]:
    """
    날짜 문자열 파싱.
    "06.03(수)" → ("2026-06-03", "2026-06-03")
    "06.09(화) ~ 06.12(금)" → ("2026-06-09", "2026-06-12")
    단일 날짜는 start_date == end_date.
    """
    date_str = date_str.strip()
    pattern = r"(\d{2})\.(\d{2})(?:\([^)]*\))?"
    matches = re.findall(pattern, date_str)
    if not matches:
        return "", ""

    def to_iso(m):
        mo, d = m
        return f"{year}-{mo}-{d}"

    start = to_iso(matches[0])
    end = to_iso(matches[-1])
    return start, end


def crawl_year(year: int = _YEAR) -> list[dict]:
    """
    연도 전체 학사일정 크롤링 (month 파라미터 없이).
    HTML 구조:
      <div class="calen_box">
        <div class="fl_month"><strong>06월</strong><span>June</span></div>
        <div class="fr_list">
          <ul>
            <li>
              <strong>06.03(수)</strong>
              <span class="list">지방선거일</span>
            </li>
          </ul>
        </div>
      </div>
    """
    params = {
        "site_dvs_cd": "kr",
        "menu_dvs_cd": "05020101",
        "year": str(year),
    }
    try:
        resp = requests.get(_BASE_URL, params=params, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        text = resp.content.decode("utf-8")
    except requests.RequestException as e:
        print(f"  [오류] 크롤링 요청 실패: {e}", file=sys.stderr)
        return []

    soup = BeautifulSoup(text, "html.parser")
    events = []

    for box in soup.select(".calen_box"):
        # 현재 박스의 월 번호 추출
        fl_month = box.select_one(".fl_month strong")
        if not fl_month:
            continue
        month_text = fl_month.get_text(strip=True)  # "06월"
        month_num = _MONTH_KR.get(month_text)
        if month_num is None:
            # "06" 숫자만 있는 경우 처리
            m = re.match(r"(\d{1,2})월?", month_text)
            month_num = int(m.group(1)) if m else 0

        # 일정 항목 파싱
        for li in box.select(".fr_list li"):
            date_tag = li.find("strong")
            list_tag = li.select_one(".list")
            if not date_tag or not list_tag:
                continue

            date_text = date_tag.get_text(strip=True)
            title = list_tag.get_text(strip=True)

            if not date_text or not title:
                continue

            start, end = _parse_date(date_text, year)
            if not start:
                continue

            events.append({
                "start_date": start,
                "end_date": end,
                "title": title,
                "month": month_num,
            })

    return events


def _dedup(events: list[dict]) -> list[dict]:
    """start_date + title 기준 중복 제거."""
    seen = set()
    result = []
    for e in events:
        key = (e["start_date"], e["title"])
        if key not in seen:
            seen.add(key)
            result.append(e)
    return result


def save(events: list[dict], year: int = _YEAR) -> None:
    """academic_calendar.json 저장."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "year": year,
        "crawled_at": datetime.now().isoformat(),
        "source_url": _BASE_URL,
        "events": events,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  저장 완료: {OUTPUT_PATH} ({len(events)}건)")


def is_stale() -> bool:
    """파일이 없거나 TTL(24h) 초과 시 True."""
    if not OUTPUT_PATH.exists():
        return True
    age = time.time() - OUTPUT_PATH.stat().st_mtime
    return age > _TTL_SECONDS


def main():
    print("=" * 50)
    print("  충남대학교 학사일정 크롤러")
    print("=" * 50)

    if not is_stale():
        age_h = (time.time() - OUTPUT_PATH.stat().st_mtime) / 3600
        print(f"  기존 파일 유효 (갱신된 지 {age_h:.1f}시간) — 스킵")
        return

    print(f"  {_YEAR}년 학사일정 크롤링 중...")
    try:
        events = crawl_year(_YEAR)
        events = _dedup(events)
        print(f"  크롤링 완료: {len(events)}건 (중복 제거 후)")

        if not events:
            print("  [경고] 크롤링 결과 없음 — 기존 파일 유지")
            return

        save(events, _YEAR)

        # 샘플 출력
        print("\n  [샘플 — 6월 일정]")
        for e in events:
            if e["month"] == 6:
                print(f"    {e['start_date']} ~ {e['end_date']}  {e['title']}")

    except Exception as e:
        print(f"  [오류] 크롤링 실패 — 기존 파일 유지: {e}", file=sys.stderr)
        return


if __name__ == "__main__":
    main()
