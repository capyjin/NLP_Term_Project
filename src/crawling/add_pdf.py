"""
기존 all_docs.json의 URL을 재방문해서 PDF 텍스트만 추가.
전체 재크롤링 없이 PDF 내용만 보강.
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

BASE_DIR = Path(__file__).parent.parent.parent
DOCS_PATH = BASE_DIR / "data" / "raw" / "all_docs.json"

MAX_PDF_PER_DOC = 1      # 문서당 최대 PDF 수
MAX_PDF_PAGES = 4        # PDF 페이지 최대
MAX_DOCS = 60            # 처리할 문서 수 제한 (속도 조절)


def make_driver():
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)


def extract_pdf(url: str, cookies: dict) -> str:
    try:
        r = requests.get(url, cookies=cookies, timeout=10, stream=True)
        if r.status_code != 200 or not r.content[:4] == b"%PDF":
            return ""
        with pdfplumber.open(io.BytesIO(r.content)) as pdf:
            texts = [p.extract_text() for p in pdf.pages[:MAX_PDF_PAGES] if p.extract_text()]
        return "\n".join(texts)[:2000]
    except:
        return ""


def run():
    with open(DOCS_PATH, encoding="utf-8") as f:
        docs = json.load(f)

    # PDF 언급된 문서만 우선 처리
    targets = [d for d in docs if ".pdf" in d.get("content", "").lower()][:MAX_DOCS]
    print(f"PDF 첨부 문서: {len(targets)}건 처리 예정")

    driver = make_driver()
    updated = 0

    for i, doc in enumerate(targets):
        try:
            driver.get(doc["url"])
            time.sleep(1.2)
            cookies = {c["name"]: c["value"] for c in driver.get_cookies()}
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
                doc["content"] = doc["content"].rstrip() + \
                    "\n\n[첨부파일 내용]\n" + "\n\n".join(pdf_texts)
                # 원본 docs 리스트에도 반영
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


if __name__ == "__main__":
    run()
