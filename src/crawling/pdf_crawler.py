"""
CNU 공지사항 PDF 첨부파일 크롤러
======================================
전략 (리서치 기반):
  - 1순위: pymupdf4llm.to_markdown()
      → RAG/LLM 최적화 Markdown 변환, 표·헤더 구조 보존, 처리속도 0.12s/건
      → 출처: https://medium.com/@danushidk507/using-pymupdf4llm-a-practical-guide-for-pdf-extraction-in-llm-rag-environments-63649915abbf
  - 2순위(fallback): pdfplumber
      → 표 데이터 추출 강점, pymupdf4llm 실패 시 사용
      → 출처: https://velog.io/@judy_choi/RAG-시리즈-pdf
  - OCR 생략: 충남대 공지 PDF는 디지털 문서 (스캔본 아님)

입력 : data/raw/all_docs.json  (기존 크롤링 결과 — PDF URL 추적에 사용)
출력 : data/raw/pdf_docs.json  (PDF 텍스트만 별도 저장)

사용법:
  python src/crawling/pdf_crawler.py
  python src/crawling/pdf_crawler.py --max 30   # 최대 30건만 처리
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

import pdfplumber
import pymupdf4llm
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

BASE_DIR = Path(__file__).parent.parent.parent
ALL_DOCS_PATH = BASE_DIR / "data" / "raw" / "all_docs.json"
PDF_DOCS_PATH = BASE_DIR / "data" / "raw" / "pdf_docs.json"

MAX_PDF_SIZE_MB = 15      # 이보다 큰 PDF는 건너뜀
MAX_PAGES = 8             # 최대 추출 페이지 수
MIN_TEXT_LEN = 80         # 이보다 짧은 추출 결과는 버림


# ── 드라이버 ────────────────────────────────────────────────

def make_driver() -> webdriver.Chrome:
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1920,1080")
    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=opts,
    )


def get_cookies(driver) -> dict:
    return {c["name"]: c["value"] for c in driver.get_cookies()}


# ── PDF 텍스트 추출 ─────────────────────────────────────────

def extract_with_pymupdf4llm(raw: bytes) -> str:
    """
    pymupdf4llm — RAG 최적화 Markdown 변환 (1순위)
    표·헤더를 Markdown 형식으로 보존하여 임베딩 품질 향상.
    """
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        md = pymupdf4llm.to_markdown(tmp_path, pages=list(range(MAX_PAGES)))
        return md.strip()
    except Exception:
        return ""
    finally:
        os.unlink(tmp_path)


def extract_with_pdfplumber(raw: bytes) -> str:
    """
    pdfplumber — 표 추출 강점, fallback (2순위)
    한글 띄어쓰기가 깔끔하게 추출되는 경우가 많음.
    """
    try:
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            parts = []
            for page in pdf.pages[:MAX_PAGES]:
                # 표가 있으면 표 텍스트 먼저 추출
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        for row in table:
                            if row:
                                parts.append(" | ".join(str(c or "") for c in row))
                text = page.extract_text()
                if text:
                    parts.append(text.strip())
        return "\n".join(parts).strip()
    except Exception:
        return ""


def extract_text(raw: bytes) -> str:
    """pymupdf4llm 시도 → 결과 빈약하면 pdfplumber fallback."""
    text = extract_with_pymupdf4llm(raw)
    if len(text) >= MIN_TEXT_LEN:
        return text[:5000]          # 너무 긴 경우 앞부분만 사용

    # fallback
    text = extract_with_pdfplumber(raw)
    return text[:5000] if len(text) >= MIN_TEXT_LEN else ""


# ── PDF 다운로드 ────────────────────────────────────────────

def download_pdf(url: str, cookies: dict) -> bytes | None:
    """PDF URL에서 바이트 다운로드. 실패하거나 크기 초과 시 None."""
    try:
        resp = requests.get(url, cookies=cookies, timeout=15, stream=True)
        if resp.status_code != 200:
            return None

        # Content-Length 확인 (너무 큰 파일 스킵)
        content_length = int(resp.headers.get("Content-Length", 0))
        if content_length > MAX_PDF_SIZE_MB * 1024 * 1024:
            print(f"      파일 너무 큼({content_length // 1024 // 1024}MB), 스킵")
            return None

        raw = resp.content
        if not raw[:4] == b"%PDF":  # PDF 시그니처 확인
            return None
        return raw
    except Exception:
        return None


# ── 메인 크롤링 로직 ────────────────────────────────────────

def find_pdf_links(driver) -> list[str]:
    """현재 페이지에서 PDF 링크 수집 (대소문자 무시)."""
    elems = driver.find_elements(
        By.XPATH,
        "//a[contains(translate(@href,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'.pdf')]",
    )
    links = []
    for e in elems:
        href = e.get_attribute("href")
        if href and href not in links:
            links.append(href)
    return links


def load_existing_pdf_docs() -> dict[str, dict]:
    """이미 처리한 PDF URL 목록 로드 (재실행 시 중복 방지)."""
    if PDF_DOCS_PATH.exists():
        with open(PDF_DOCS_PATH, encoding="utf-8") as f:
            docs = json.load(f)
        return {d["pdf_url"]: d for d in docs}
    return {}


def crawl_pdfs(max_docs: int = 60):
    # 기존 공지 데이터 로드
    if not ALL_DOCS_PATH.exists():
        print(f"[오류] {ALL_DOCS_PATH} 없음. 먼저 cnu_crawler.py 실행하세요.")
        sys.exit(1)

    with open(ALL_DOCS_PATH, encoding="utf-8") as f:
        all_docs = json.load(f)

    # PDF 언급된 문서만 처리 (효율화)
    targets = [d for d in all_docs if ".pdf" in d.get("content", "").lower()]
    targets = targets[:max_docs]
    print(f"PDF 첨부 문서: {len(targets)}건 처리 예정")

    # 이미 처리한 항목 로드
    done_map = load_existing_pdf_docs()
    pdf_docs = list(done_map.values())
    new_count = 0

    driver = make_driver()

    try:
        for i, doc in enumerate(targets):
            print(f"\n[{i+1}/{len(targets)}] {doc.get('content','')[:40]}")
            try:
                driver.get(doc["url"])
                time.sleep(1.5)
                cookies = get_cookies(driver)
                pdf_links = find_pdf_links(driver)

                if not pdf_links:
                    print("  PDF 링크 없음")
                    continue

                for pdf_url in pdf_links[:2]:   # 문서당 최대 2개 PDF
                    if pdf_url in done_map:
                        print(f"  이미 처리됨: {pdf_url[-40:]}")
                        continue

                    print(f"  다운로드 중: {pdf_url[-50:]}")
                    raw = download_pdf(pdf_url, cookies)
                    if not raw:
                        print("  다운로드 실패 또는 스킵")
                        continue

                    text = extract_text(raw)
                    if not text:
                        print("  텍스트 추출 실패")
                        continue

                    entry = {
                        "pdf_url": pdf_url,
                        "page_url": doc["url"],
                        "title": doc.get("content", "")[:60].split("\n")[0],
                        "category": doc.get("category", ""),
                        "content": text,
                        "source": "pdf",
                    }
                    pdf_docs.append(entry)
                    done_map[pdf_url] = entry
                    new_count += 1
                    print(f"  추출 완료 ({len(text)}자)")

            except Exception as e:
                print(f"  오류: {e}")
            time.sleep(0.5)

    finally:
        driver.quit()

    # 저장
    PDF_DOCS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PDF_DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(pdf_docs, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*40}")
    print(f"신규 PDF 추출: {new_count}건")
    print(f"전체 PDF 문서: {len(pdf_docs)}건 → {PDF_DOCS_PATH}")
    print(f"\n다음 단계: python src/preprocessing/preprocess.py  (재전처리)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=60, help="처리할 최대 문서 수")
    args = parser.parse_args()
    crawl_pdfs(max_docs=args.max)
