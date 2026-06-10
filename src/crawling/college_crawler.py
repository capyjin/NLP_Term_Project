"""
충남대학교 단과대/학과 구조 크롤러
────────────────────────────────
출력: data/raw/college_departments.json
TTL : 7일 (학과 구조는 잘 바뀌지 않음)

전략:
  1. CNU 포털 내비게이션에서 단과대 URL 목록 수집 시도
  2. 각 단과대 페이지에서 소속 학과 파싱 시도
  3. 실패 시 기존 JSON 유지 (fallback 보장)
"""

import json
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT   = BASE_DIR / "data" / "raw" / "college_departments.json"

try:
    import requests
    from bs4 import BeautifulSoup
    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False

TIMEOUT = 10  # seconds

# 단과대 목록 페이지 (포털 내비게이션 기반)
_PORTAL_NAV_URL = "https://plus.cnu.ac.kr/html/kr/"

# 단과대별 홈페이지 URL 힌트 (fallback 탐색용)
_KNOWN_COLLEGE_URLS = {
    "공과대학":        "https://eng.cnu.ac.kr/",
    "인문대학":        "https://humane.cnu.ac.kr/",
    "사회과학대학":    "https://social.cnu.ac.kr/",
    "자연과학대학":    "https://natural.cnu.ac.kr/",
    "경상대학":        "https://business.cnu.ac.kr/",
    "농업생명과학대학":"https://agri.cnu.ac.kr/",
    "사범대학":        "https://education.cnu.ac.kr/",
    "생활과학대학":    "https://livingsci.cnu.ac.kr/",
    "예술대학":        "https://art.cnu.ac.kr/",
    "생명시스템과학대학":"https://biosys.cnu.ac.kr/",
    "약학대학":        "https://pharm.cnu.ac.kr/",
    "의과대학":        "https://medicine.cnu.ac.kr/",
    "간호대학":        "https://nursing.cnu.ac.kr/",
    "수의과대학":      "https://vet.cnu.ac.kr/",
}


def _get_session() -> "requests.Session":
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "ko-KR,ko;q=0.9",
    })
    return s


def _parse_departments_from_page(html: str) -> list[str]:
    """단과대 페이지 HTML에서 학과명 목록 추출 시도."""
    soup = BeautifulSoup(html, "html.parser")
    depts = []

    # 공통 패턴 1: <a> 태그에 학과/학부/전공 포함
    for a in soup.find_all("a"):
        text = a.get_text(strip=True)
        if any(suffix in text for suffix in ("학과", "학부", "전공", "대학원")):
            if len(text) <= 20 and text not in depts:
                depts.append(text)

    # 공통 패턴 2: li 태그
    if not depts:
        for li in soup.find_all("li"):
            text = li.get_text(strip=True)
            if any(suffix in text for suffix in ("학과", "학부", "전공")):
                if len(text) <= 20 and text not in depts:
                    depts.append(text)

    return depts


def _try_crawl_college(session: "requests.Session", college: str, url: str) -> list[str] | None:
    """단과대 URL에서 학과 목록 파싱. 실패 시 None 반환."""
    try:
        resp = session.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"
        depts = _parse_departments_from_page(resp.text)
        if depts:
            return depts
    except Exception:
        pass
    return None


def crawl() -> bool:
    """
    메인 크롤링 함수.
    성공 시 college_departments.json 갱신 후 True 반환.
    실패 시 기존 파일 유지하고 False 반환.
    """
    if not _HAS_DEPS:
        print("[college_crawler] requests/BeautifulSoup4 미설치 — 기존 파일 유지")
        return False

    session = _get_session()

    # 기존 파일에서 colleges 구조 로드 (fallback 기준)
    existing_colleges: list[dict] = []
    if OUTPUT.exists():
        try:
            existing_data = json.loads(OUTPUT.read_text(encoding="utf-8"))
            existing_colleges = existing_data.get("colleges", [])
        except Exception:
            pass

    if not existing_colleges:
        print("[college_crawler] 기존 college_departments.json 없음 — 크롤링만으로 진행")

    # 단과대별 크롤링 시도
    updated_colleges = []
    any_success = False

    for entry in existing_colleges:
        college_name = entry["college"]
        url = _KNOWN_COLLEGE_URLS.get(college_name)
        if url:
            depts = _try_crawl_college(session, college_name, url)
            if depts and len(depts) >= len(entry["departments"]) // 2:
                # 크롤링 결과가 기존 데이터의 절반 이상이면 신뢰
                updated_colleges.append({
                    "college": college_name,
                    "departments": depts,
                })
                any_success = True
                print(f"  ✓ {college_name}: {len(depts)}개 학과 크롤링")
                continue
        # fallback: 기존 데이터 유지
        updated_colleges.append(entry)

    if not any_success:
        print("[college_crawler] 모든 크롤링 실패 — 기존 파일 유지")
        return False

    # JSON 저장
    result = {
        "crawled_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "source_url": _PORTAL_NAV_URL,
        "colleges": updated_colleges,
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[college_crawler] {OUTPUT.name} 갱신 완료 ({len(updated_colleges)}개 단과대)")
    return True


if __name__ == "__main__":
    success = crawl()
    sys.exit(0 if success else 1)
