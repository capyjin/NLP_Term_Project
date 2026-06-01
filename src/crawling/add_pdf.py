"""
PDF 텍스트 보강 스크립트 — all_docs.json 인플레이스 업데이트
────────────────────────────────────────────────────────────
pdf_crawler.py와의 차이:
  - pdf_crawler.py: PDF 텍스트를 별도 pdf_docs.json으로 저장
  - add_pdf.py    : 기존 all_docs.json 게시글의 content에 PDF 텍스트를 직접 추가
    → 전체 재크롤링 없이 PDF 내용만 보강할 때 사용

실행:
  python src/crawling/add_pdf.py

입출력:
  입력:  data/raw/all_docs.json (기존)
  출력:  data/raw/all_docs.json (PDF 텍스트 추가된 버전으로 덮어쓰기)

처리 범위:
  - content에 ".pdf" 언급된 게시글만 처리 (효율화)
  - 문서당 최대 1개 PDF, 최대 4페이지
  - 최대 60건 처리 (MAX_DOCS)

⚠️ all_docs.json을 덮어쓰므로 실행 전 백업 권장
   처리 후 preprocess.py 재실행 필요
"""

import json, time, io, os
from pathlib import Path
import requests
import pdfplumber
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

BASE_DIR  = Path(__file__).parent.parent.parent
DOCS_PATH = BASE_DIR / "data" / "raw" / "all_docs.json"

MAX_PDF_PER_DOC = 1    # 문서당 처리할 PDF 최대 수
MAX_PDF_PAGES   = 4    # 추출할 최대 페이지 수
MAX_DOCS        = 60   # 처리할 문서 수 상한 (속도 조절)


def make_driver():
    """헤드리스 Chrome 드라이버 (쿠키 수집용)."""
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)


def extract_pdf(url: str, cookies: dict) -> str:
    """
    PDF URL 다운로드 → pdfplumber로 텍스트 추출.
    실패하거나 PDF가 아니면 빈 문자열 반환.
    """
    try:
        r = requests.get(url, cookies=cookies, timeout=10, stream=True)
        if r.status_code != 200 or not r.content[:4] == b"%PDF":
            return ""
        with pdfplumber.open(io.BytesIO(r.content)) as pdf:
            texts = [p.extract_text() for p in pdf.pages[:MAX_PDF_PAGES] if p.extract_text()]
        return "\n".join(texts)[:2000]   # 너무 긴 내용 잘라냄
    except:
        return ""


def run():
    """all_docs.json에서 PDF 언급 게시글을 찾아 PDF 텍스트를 content에 추가."""
    with open(DOCS_PATH, encoding="utf-8") as f:
        docs = json.load(f)

    # content에 ".pdf" 언급된 게시글만 우선 처리 (효율화)
    targets = [d for d in docs if ".pdf" in d.get("content", "").lower()][:MAX_DOCS]
    print(f"PDF 첨부 문서: {len(targets)}건 처리 예정")

    driver = make_driver()
    updated = 0

    for i, doc in enumerate(targets):
        try:
            driver.get(doc["url"])
            time.sleep(1.2)
            # 쿠키: 인증이 필요한 PDF 다운로드에 사용
            cookies   = {c["name"]: c["value"] for c in driver.get_cookies()}
            pdf_links = driver.find_elements(
                By.CSS_SELECTOR, "a[href*='.pdf'], a[href*='.PDF']"
            )
            pdf_texts = []
            for link in pdf_links[:MAX_PDF_PER_DOC]:
                href = link.get_attribute("href")
                if href:
                    text = extract_pdf(href, cookies)
                    if text:
                        pdf_texts.append(text)

            if pdf_texts:
                # content 끝에 "[첨부파일 내용]" 섹션으로 추가
                doc["content"] = doc["content"].rstrip() + \
                    "\n\n[첨부파일 내용]\n" + "\n\n".join(pdf_texts)
                # 원본 docs 리스트에도 반영 (URL 기준 매칭)
                for orig in docs:
                    if orig["url"] == doc["url"]:
                        orig["content"] = doc["content"]
                        break
                updated += 1
                print(f"  [{i+1}/{len(targets)}] PDF 추가: {doc['content'][:40]}")
            else:
                print(f"  [{i+1}/{len(targets)}] PDF 없음: {doc['url'][-40:]}")

        except Exception as e:
            print(f"  오류: {e}")
        time.sleep(0.3)

    driver.quit()

    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"\nPDF 추가 완료: {updated}건 업데이트 → {DOCS_PATH}")
    print(f"다음 단계: python src/preprocessing/preprocess.py  (재청킹)")


if __name__ == "__main__":
    run()
