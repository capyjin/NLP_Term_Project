"""
충남대학교 공지사항 크롤러 (Selenium)
──────────────────────────────────────
대상: plus.cnu.ac.kr — 학사/장학/일반/취업/행사 5개 게시판
출력: data/raw/all_docs.json  (기존 데이터와 병합 저장)
      data/raw/crawled.json   (이번 실행분만 저장)

실행:
  python src/crawling/cnu_crawler.py

크롤링 범위:
  - 게시판당 최대 3페이지 × 게시글 20개 = 최대 60건/게시판
  - 총 최대 300건 (5개 게시판 × 60건)

흐름:
  make_driver() → 각 게시판 URL로 이동 → 게시글 URL 수집
    → 각 게시글 접속 → 제목/본문 추출 → all_docs.json 병합 저장

⚠️ all_docs.json에 기존 데이터와 append 병합
   중복 URL 제거는 preprocess.py에서 처리
"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import json, time, os

def make_driver():
    """헤드리스 Chrome 드라이버 생성 (Colab 호환 설정 포함)."""
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")            # Colab/Linux 루트 실행 시 필요
    opts.add_argument("--disable-dev-shm-usage") # Colab 메모리 제한 우회
    opts.add_argument("--window-size=1920,1080")
    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=opts
    )

# 크롤링 대상 게시판 (5개) — plus.cnu.ac.kr 공지사항
BOARDS = {
    "학사공지": "https://plus.cnu.ac.kr/_prog/_board/?code=sub07_0701&site_dvs_cd=kr&menu_dvs_cd=0701",
    "장학공지": "https://plus.cnu.ac.kr/_prog/_board/?code=sub07_0702&site_dvs_cd=kr&menu_dvs_cd=0702",
    "일반공지": "https://plus.cnu.ac.kr/_prog/_board/?code=sub07_0703&site_dvs_cd=kr&menu_dvs_cd=0703",
    "취업공지": "https://plus.cnu.ac.kr/_prog/_board/?code=sub07_0704&site_dvs_cd=kr&menu_dvs_cd=0704",
    "행사안내": "https://plus.cnu.ac.kr/_prog/_board/?code=sub07_0705&site_dvs_cd=kr&menu_dvs_cd=0705",
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_PATH = os.path.join(BASE_DIR, "data", "raw", "crawled.json")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)


def crawl():
    """5개 게시판 순회 크롤링 → all_docs.json 병합 저장."""
    driver = make_driver()
    all_docs = []

    for category, board_url in BOARDS.items():
        print(f"\n[{category}] 크롤링 시작...")
        article_urls = []

        # 게시판 3페이지까지 URL 수집
        for page in range(1, 4):
            url = board_url + f"&GotoPage={page}"
            driver.get(url)
            time.sleep(2)
            # "mode=V": 게시글 상세보기 링크
            links = driver.find_elements(By.CSS_SELECTOR, "a[href*='mode=V']")
            for link in links:
                href = link.get_attribute("href")
                if href and href not in article_urls:
                    article_urls.append(href)

        print(f"  게시글 URL 수집: {len(article_urls)}개")

        collected = 0
        for url in article_urls[:20]:   # 게시판당 최대 20건
            try:
                driver.get(url)
                time.sleep(1.5)

                # 제목 추출 — CSS 선택자 다중 fallback
                try:
                    title = driver.find_element(By.CSS_SELECTOR,
                        ".board-view-title, .view-title, h3, h4, .title").text.strip()
                except:
                    title = driver.title.strip()

                # 본문 추출 — CSS 선택자 fallback → body 텍스트 일부
                try:
                    content = driver.find_element(By.CSS_SELECTOR,
                        ".board-view-content, .view-content, .content").text.strip()
                except:
                    body  = driver.find_element(By.TAG_NAME, "body").text
                    lines = [l.strip() for l in body.split('\n') if l.strip()]
                    content = '\n'.join(lines[10:50])   # 헤더/네비 제외

                if len(content) > 50:
                    all_docs.append({
                        "category": category,
                        "title":    title[:100],
                        "content":  content[:5000],   # 너무 긴 본문 잘라냄
                        "url":      url
                    })
                    collected += 1
                    print(f"  [{collected}] {title[:40]}")

            except Exception as e:
                print(f"  오류: {e}")
            time.sleep(0.5)

        print(f"  {category} 완료: {collected}건")

    driver.quit()

    # 기존 all_docs.json과 병합 (중복 제거는 preprocess.py에서 처리)
    existing_path = os.path.join(BASE_DIR, "data", "raw", "all_docs.json")
    try:
        with open(existing_path, encoding="utf-8") as f:
            existing = json.load(f)
    except:
        existing = []

    merged = existing + all_docs
    with open(existing_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    # 이번 실행분만 별도 저장
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*40}")
    print(f"크롤링 완료: {len(all_docs)}건")
    print(f"전체 데이터: {len(merged)}건 → {existing_path}")
    print(f"다음 단계: python src/crawling/pdf_crawler.py  (PDF 첨부파일 추출)")
    print(f"그 다음  : python src/preprocessing/preprocess.py  (청킹)")

if __name__ == "__main__":
    crawl()
