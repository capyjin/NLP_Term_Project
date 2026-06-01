"""
전처리 + 청킹 스크립트
──────────────────────
입력: data/raw/all_docs.json  (cnu_crawler.py 출력)
      data/raw/pdf_docs.json  (pdf_crawler.py 출력 — 있으면 자동 병합)
출력: data/processed/chunks.json

실행:
  python src/preprocessing/preprocess.py

청크 형식:
  {"id": "...", "content": "...", "title": "...", "category": "...", "url": "..."}

청킹 전략:
  - 문자 단위 슬라이딩 윈도우 (CHUNK_SIZE=400자, CHUNK_OVERLAP=50자)
  - 문장 경계 우선: 마침표·개행·!? 기준 ±30자 탐색 후 자름
  - 30자 미만 청크 제거

다음 단계:
  python src/vectordb/build_db.py  (ChromaDB 구축)
"""

import json
import re
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "all_docs.json"
PDF_PATH = BASE_DIR / "data" / "raw" / "pdf_docs.json"
OUT_DIR = BASE_DIR / "data" / "processed"
OUT_PATH = OUT_DIR / "chunks.json"

CHUNK_SIZE = 400      # 목표 청크 크기 (자)
CHUNK_OVERLAP = 50    # 청크 간 겹침


# 제거할 노이즈 패턴
_NOISE_PATTERNS = [
    r"작성자[^\n]*",
    r"등록일[^\n]*",
    r"조회수[^\n]*",
    r"파일\n?",
    r"첨부파일[^\n]*",
    r"목록\s*",
    r"페이지 관리자[^\n]*",
    r"관리자메일[^\n]*",
    r"\s*\n\s*\n\s*",   # 연속 빈 줄 → 한 줄
]
_NOISE_RE = re.compile("|".join(_NOISE_PATTERNS))


def clean_content(text: str) -> str:
    text = _NOISE_RE.sub("\n", text)
    # 파일명처럼 보이는 줄(.hwp .pdf .png) 제거
    lines = [l for l in text.splitlines() if not re.search(r"\.(hwp|pdf|png|jpg|xlsx|docx|zip)$", l.strip(), re.I)]
    text = "\n".join(lines)
    # 앞뒤 공백 정리
    return text.strip()


def extract_title(raw_title: str, content: str) -> str:
    if raw_title and raw_title.strip():
        return raw_title.strip()
    # content 첫 줄이 제목
    first_line = content.splitlines()[0].strip() if content else ""
    return first_line[:80]


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """문자 단위 슬라이딩 윈도우 청킹. 문장 경계 우선."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        # 마침표/개행 기준으로 자연스러운 경계 탐색 (±30자)
        boundary = -1
        for i in range(end, max(start + size // 2, end - 30), -1):
            if text[i] in (".", "。", "\n", "!", "?"):
                boundary = i + 1
                break
        if boundary == -1:
            boundary = end
        chunk = text[start:boundary].strip()
        if chunk:
            chunks.append(chunk)
        start = boundary - overlap
    return [c for c in chunks if len(c) >= 30]


def preprocess():
    with open(RAW_PATH, encoding="utf-8") as f:
        docs = json.load(f)
    print(f"공지 문서 수: {len(docs)}")

    # PDF 문서 병합 (있으면)
    if PDF_PATH.exists():
        with open(PDF_PATH, encoding="utf-8") as f:
            pdf_docs = json.load(f)
        # pdf_docs는 page_url 기준으로 url 필드 통일
        for d in pdf_docs:
            d.setdefault("url", d.get("page_url", ""))
            d.setdefault("title", d.get("title", ""))
        docs = docs + pdf_docs
        print(f"PDF 문서 병합: +{len(pdf_docs)}건 → 합계 {len(docs)}건")
    else:
        print("pdf_docs.json 없음 (PDF 크롤링 전 상태)")

    print(f"원본 문서 수: {len(docs)}")

    # 중복 URL 제거
    seen_urls = set()
    deduped = []
    for doc in docs:
        url = doc.get("url", "")
        if url not in seen_urls:
            seen_urls.add(url)
            deduped.append(doc)
    print(f"중복 제거 후: {len(deduped)}")

    chunks = []
    skipped = 0
    for doc in deduped:
        content = clean_content(doc.get("content", ""))
        title = extract_title(doc.get("title", ""), content)

        # 제목이 content 첫 줄인 경우 본문에서 제거
        if content.startswith(title):
            content = content[len(title):].strip()

        if len(content) < 50:
            skipped += 1
            continue

        for i, chunk in enumerate(chunk_text(content)):
            chunks.append({
                "id": f"{doc.get('category','')}_url{len(seen_urls)}_chunk{i}_{len(chunks)}",
                "content": chunk,
                "title": title,
                "category": doc.get("category", ""),
                "url": doc.get("url", ""),
            })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"짧은 문서 제외: {skipped}건")
    print(f"최종 청크 수: {len(chunks)}")
    print(f"저장 완료 → {OUT_PATH}")
    return chunks


if __name__ == "__main__":
    preprocess()
