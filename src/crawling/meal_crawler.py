"""
충남대학교 학생식당 식단 크롤러 (v2)
────────────────────────────────────
대상: https://mobileadmin.cnu.ac.kr/food/index.jsp?searchYmd=YYYY.MM.DD
출력: data/raw/meal_menu.json

실행:
  python src/crawling/meal_crawler.py

실제 확인된 HTML 구조 (2026-06-01 기준):
  <table class="menu-tbl type-cap">
    <tr>  (헤더)
      <th colspan=2>구분</th>
      <th>제1학생회관</th>  ... (제2~4, 생활과학대학)
    </tr>
    <tr>  (조식-직원)
      <td rowspan=2>조식</td>
      <td>직원</td>
      <td rowspan=100>메뉴운영내역</td>  ← 제1학생회관 전체 스팬
      <td>운영안함</td> ...
    </tr>
    <tr>  (조식-학생)
      <td>학생</td>
      ← 제1학생회관은 rowspan=100으로 채워짐 → 이 행에 없음
      <td><ul><li><h3 class="menu-tit03">정식(4500)</h3>
                  <p>냉우동<br/>참치마요주먹밥<br/></p></li></ul></td>
      ...
    </tr>
    <tr> (중식-직원) ... </tr>
    <tr> (중식-학생) ... </tr>
    <tr> (석식-직원) ... </tr>
    <tr> (석식-학생) ... </tr>
  </table>

핵심:
  1. 날짜별 URL: ?searchYmd=YYYY.MM.DD
  2. colspan/rowspan 추적으로 정확한 컬럼-식당 매핑
  3. 메뉴 아이템: <p>...</p> 내 <br/> 구분 (기존 파서는 <li> 탐색으로 실패)
  4. 학생 행만 추출 (직원 행 제외)

주의:
  - 허위 메뉴 절대 생성 금지
  - 파싱 실패 시 빈 리스트 반환
"""

import json
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).parent.parent.parent
OUT_PATH = BASE_DIR / "data" / "raw" / "meal_menu.json"
MEAL_URL = "https://mobileadmin.cnu.ac.kr/food/index.jsp"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9",
    "Referer": "https://mobileadmin.cnu.ac.kr/",
}

_DAY_NAMES = ["월", "화", "수", "목", "금", "토", "일"]


# ── 네트워크 ─────────────────────────────────────────────────────────────────

def _fetch(url: str, timeout: int = 15) -> Optional[str]:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        resp.encoding = "utf-8"
        return resp.text
    except requests.RequestException as e:
        print(f"[meal_crawler] 접속 실패 ({url}): {e}")
        return None


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def _date_to_day(date_str: str) -> str:
    try:
        return _DAY_NAMES[date.fromisoformat(date_str).weekday()]
    except ValueError:
        return ""


def _parse_price(text: str) -> int:
    m = re.search(r"\((\d[\d,]+)\)", text)
    if m:
        try:
            return int(m.group(1).replace(",", ""))
        except ValueError:
            pass
    return 0


def _clean_items(raw_lines: list[str]) -> list[str]:
    """
    메뉴 아이템 정제.
    - 알레르기 표시 제거: (pork included), (chicken included) 등
    - 가격 포함 텍스트 제거: "정식(4500)" 같은 줄
    - 너무 짧거나 빈 항목 제거
    """
    cleaned = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        # 가격 줄 제거: "정식(4500)" 패턴
        if re.match(r"^정식\s*\(\d+\)$", line):
            continue
        # 알레르기 괄호 제거
        line = re.sub(r"\([^)]*(?:pork|beef|chicken|shrimp|milk|egg|included)[^)]*\)", "", line, flags=re.I)
        # 원산지 괄호/대괄호 제거
        line = re.sub(r"\[[^\]]*\]", "", line)
        line = re.sub(r"\([^)]*(?:국내산|수입산|원산지)[^)]*\)", "", line)
        line = line.strip(" ·,./").strip()
        if line and len(line) >= 2:
            cleaned.append(line)
    return cleaned


# ── 그리드 빌더 (colspan/rowspan 추적) ────────────────────────────────────────

def _build_grid(tbl) -> dict[tuple[int, int], object]:
    """
    table의 모든 tr/td/th를 colspan/rowspan을 반영한 2D 딕셔너리로 변환.
    grid[(row_idx, col_idx)] = BeautifulSoup tag
    """
    grid: dict[tuple[int, int], object] = {}
    occupied: set[tuple[int, int]] = set()
    rows = tbl.find_all("tr")

    for r_idx, row in enumerate(rows):
        cells = row.find_all(["td", "th"])
        col_pos = 0

        for cell in cells:
            # 이미 점유된 컬럼 건너뛰기
            while (r_idx, col_pos) in occupied:
                col_pos += 1

            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))

            grid[(r_idx, col_pos)] = cell

            # 점유 영역 등록 (rowspan/colspan 범위 — 최대 20행으로 제한)
            for dr in range(min(rowspan, 20)):
                for dc in range(colspan):
                    if dr > 0 or dc > 0:
                        occupied.add((r_idx + dr, col_pos + dc))

            col_pos += colspan

    return grid


# ── 식단 파싱 ─────────────────────────────────────────────────────────────────

def _parse_menu_cell(cell) -> Optional[dict]:
    """
    td에서 메뉴 정보 추출.

    실제 구조:
    <td>
      <ul><li>
        <h3 class="menu-tit03">정식(4500)</h3>
        <p>냉우동<br/>참치마요주먹밥<br/>...</p>
      </li></ul>
    </td>

    핵심: 메뉴 아이템은 <li>가 아닌 <p> 내 <br/> 구분자로 저장됨.
    """
    h3 = cell.find("h3", class_="menu-tit03")
    if not h3:
        return None

    price     = _parse_price(h3.get_text(strip=True))
    p_tag     = h3.find_next_sibling("p")
    if not p_tag:
        # <li> 하위에 있을 수 있음
        li = h3.find_parent("li")
        if li:
            p_tag = li.find("p")

    if not p_tag:
        return None

    # <br/> → 줄바꿈 치환 후 split
    for br in p_tag.find_all("br"):
        br.replace_with("\n")
    items = _clean_items(p_tag.get_text().split("\n"))

    if not items:
        return None
    return {"price": price, "items": items}


def parse_one_day(html: str, date_str: str) -> list[dict]:
    """
    특정 날짜 HTML → 학생 식단 리스트.

    알고리즘:
    1. table.menu-tbl 찾기
    2. colspan/rowspan 추적 그리드 빌드
    3. 헤더 행(row 0)에서 컬럼번호 → 식당명 매핑
    4. 각 행에서 조식/중식/석식 타입 추적
    5. "학생" 행만 처리: 각 컬럼 셀에서 메뉴 추출
    """
    soup = BeautifulSoup(html, "html.parser")
    tbl  = soup.find("table", class_="menu-tbl")
    if not tbl:
        return []

    grid = _build_grid(tbl)
    rows = tbl.find_all("tr")
    if not rows:
        return []

    # 헤더(row 0): 컬럼 번호 → 식당명
    col_to_restaurant: dict[int, str] = {}
    col = 0
    for cell in rows[0].find_all(["th", "td"]):
        txt = cell.get_text(strip=True)
        cs  = int(cell.get("colspan", 1))
        if txt not in ("구분",):
            col_to_restaurant[col] = txt
        col += cs

    results          = []
    current_mealtype = None

    for r_idx in range(1, len(rows)):
        # 컬럼 0: 조식/중식/석식 (rowspan 으로 없을 수 있음)
        type_cell = grid.get((r_idx, 0))
        if type_cell:
            t = type_cell.get_text(strip=True)
            if "조식" in t:
                current_mealtype = "아침"
            elif "중식" in t:
                current_mealtype = "점심"
            elif "석식" in t:
                current_mealtype = "저녁"

        # 컬럼 1: 직원/학생 구분
        persona_cell = grid.get((r_idx, 1))
        if not persona_cell:
            continue
        if "학생" not in persona_cell.get_text(strip=True):
            continue  # 직원 행은 건너뜀
        if not current_mealtype:
            continue

        # 식당별 컬럼 처리 (col 2 이상)
        for col_idx, restaurant in col_to_restaurant.items():
            cell = grid.get((r_idx, col_idx))
            if not cell:
                continue  # rowspan 으로 채워진 셀 (예: 메뉴운영내역)
            menu = _parse_menu_cell(cell)
            if not menu:
                continue  # 운영안함

            results.append({
                "date":       date_str,
                "day":        _date_to_day(date_str),
                "restaurant": restaurant,
                "meal_type":  current_mealtype,
                "menu":       menu["items"],
                "price":      menu["price"],
                "url":        f"{MEAL_URL}?searchYmd={date_str.replace('-', '.')}",
            })

    return results


# ── 주간 크롤링 ───────────────────────────────────────────────────────────────

def crawl_week() -> list[dict]:
    """이번 주 월~토 식단 전체 수집. 빈 날짜는 건너뜀."""
    today      = date.today()
    monday     = today - timedelta(days=today.weekday())
    all_menus  = []

    for offset in range(6):  # 월~토 (일요일 제외)
        target = monday + timedelta(days=offset)
        ymd    = target.strftime("%Y.%m.%d")
        url    = f"{MEAL_URL}?searchYmd={ymd}"

        html = _fetch(url)
        if not html:
            print(f"  {ymd}: 접속 실패")
            continue

        date_str  = target.isoformat()
        day_menus = parse_one_day(html, date_str)

        if day_menus:
            all_menus.extend(day_menus)
            print(f"  {date_str} ({_date_to_day(date_str)}): {len(day_menus)}건")
        else:
            print(f"  {date_str}: 메뉴 없음 (운영 없는 날)")

    return all_menus


def crawl_and_save() -> bool:
    """주간 크롤링 실행 및 저장. 성공 시 True."""
    print(f"[meal_crawler] 주간 식단 크롤링 시작")
    menus = crawl_week()

    if not menus:
        print("[meal_crawler] 파싱된 메뉴 없음")
        return False

    data = {
        "fetched_at": datetime.now().isoformat(),
        "source_url": MEAL_URL,
        "menus":      menus,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[meal_crawler] 저장 완료: {OUT_PATH} (총 {len(menus)}건)")
    return True


if __name__ == "__main__":
    ok = crawl_and_save()
    if not ok:
        print("[meal_crawler] 크롤링 실패 - data/raw/meal_menu.json 수동 작성 권장")
        sys.exit(1)
