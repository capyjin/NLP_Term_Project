"""
충남대 웹사이트 크롤러
대상: 학사공지, 학사일정, 졸업요건, 장학금, 셔틀버스, 식당메뉴
"""

import requests
from bs4 import BeautifulSoup
import json
import time
from pathlib import Path

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

BASE_URL = "https://www.cnu.ac.kr"
RAW_DIR = Path("../../data/raw")

CRAWL_TARGETS = {
    "academic_notice": "/main/kr/sub05_01_01.do",   # 학사공지
    "scholarship": "/main/kr/sub05_01_02.do",        # 장학공지
    "general_notice": "/main/kr/sub05_01_03.do",     # 일반공지
}


def crawl_notice_list(category: str, path: str, max_pages: int = 5) -> list[dict]:
    items = []
    for page in range(1, max_pages + 1):
        url = f"{BASE_URL}{path}?pageIndex={page}"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            break
        soup = BeautifulSoup(resp.text, "lxml")
        rows = soup.select("table tbody tr")
        if not rows:
            break
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 3:
                continue
            title_tag = row.select_one("td.subject a, td a")
            if not title_tag:
                continue
            items.append({
                "category": category,
                "title": title_tag.get_text(strip=True),
                "href": title_tag.get("href", ""),
            })
        time.sleep(0.5)
    return items


def crawl_detail(href: str) -> str:
    if not href.startswith("http"):
        href = BASE_URL + href
    try:
        resp = requests.get(href, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(resp.text, "lxml")
        content = soup.select_one(".board-view-content, .view-content, .board_content")
        return content.get_text(strip=True) if content else ""
    except Exception:
        return ""


def run_crawl():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    all_docs = []
    for category, path in CRAWL_TARGETS.items():
        print(f"크롤링 중: {category}")
        items = crawl_notice_list(category, path)
        for item in items:
            detail = crawl_detail(item["href"])
            item["content"] = detail
            all_docs.append(item)
            time.sleep(0.3)
    out_path = RAW_DIR / "notices.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, ensure_ascii=False, indent=2)
    print(f"저장 완료: {out_path} ({len(all_docs)}건)")


if __name__ == "__main__":
    run_crawl()
