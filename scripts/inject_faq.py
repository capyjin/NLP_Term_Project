"""
FAQ 데이터 삽입 스크립트
────────────────────────
입력: data/faq/faq_manual.json  (수동 관리 FAQ 소스)
출력: data/processed/chunks.json (FAQ 청크 추가 반영)

실행:
  python scripts/inject_faq.py

흐름:
  faq_manual.json 로드
    → search_text = category + subcategory + title + keywords + content 생성
    → 기존 chunks.json에 없는 ID만 추가 (중복 방지)
    → 원본 백업: chunks_original.json (최초 1회만)
    → chunks.json 덮어쓰기

search_text 역할:
  - BM25: retriever.py의 _build_bm25()가 search_text로 인덱싱
           → category/subcategory/keywords 키워드가 BM25 매칭에 반영
  - Dense: build_db.py가 embed_text=search_text로 KURE-v1 임베딩
           → 의미 벡터에 카테고리·키워드 정보 포함

재실행 안전:
  - 이미 추가된 faq_ ID는 건너뜀 → 중복 없음
  - chunks_original.json이 있으면 백업 재생성 안 함

FAQ 내용 수정 후 재삽입:
  1. faq_manual.json 편집
  2. chunks_original.json 기준으로 재merge: --reset 플래그 사용
     python scripts/inject_faq.py --reset
"""

import json
import shutil
import sys
from pathlib import Path

BASE_DIR    = Path(__file__).parent.parent
FAQ_PATH    = BASE_DIR / "data" / "faq"     / "faq_manual.json"
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.json"
BACKUP_PATH = BASE_DIR / "data" / "processed" / "chunks_original.json"


def build_search_text(item: dict) -> str:
    """
    search_text = category + subcategory + title + keywords + content
    BM25 인덱싱과 Dense 임베딩 모두에 사용되는 풍부한 검색용 텍스트.
    """
    keywords_str = " ".join(item.get("keywords", []))
    parts = [
        item.get("category", ""),
        item.get("subcategory", ""),
        item.get("title", ""),
        keywords_str,
        item.get("content", ""),
    ]
    return " ".join(p for p in parts if p).strip()


def faq_to_chunk(item: dict) -> dict:
    """FAQ 항목 → chunks.json 호환 형식 변환 (search_text 포함)."""
    return {
        "id":          item["id"],
        "title":       item["title"],
        "category":    item.get("category", ""),
        "subcategory": item.get("subcategory", ""),
        "keywords":    item.get("keywords", []),
        "content":     item["content"],
        "url":         item.get("url", ""),
        "source_type": item.get("source_type", "faq_manual"),
        "search_text": build_search_text(item),
    }


def inject(reset: bool = False):
    """
    FAQ를 chunks.json에 삽입.
    reset=True: chunks_original.json 기준으로 재merge (FAQ 내용 변경 시 사용).
    """
    # ── 원본 백업 ──────────────────────────────────────────────────
    if not BACKUP_PATH.exists():
        shutil.copy(CHUNKS_PATH, BACKUP_PATH)
        print(f"원본 백업 생성: {BACKUP_PATH.name}")

    # ── 로드 ───────────────────────────────────────────────────────
    with open(FAQ_PATH,    encoding="utf-8") as f:
        faq_items = json.load(f)

    source_path = BACKUP_PATH if reset else CHUNKS_PATH
    with open(source_path, encoding="utf-8") as f:
        chunks = json.load(f)

    if reset:
        # 기존 faq_ 청크 제거 후 원본 기준 재삽입
        chunks = [c for c in chunks if not c["id"].startswith("faq_")]
        print("기존 FAQ 청크 제거 후 재삽입 모드")

    # ── 중복 확인 및 추가 ──────────────────────────────────────────
    existing_ids = {c["id"] for c in chunks}
    new_chunks   = [
        faq_to_chunk(item)
        for item in faq_items
        if item["id"] not in existing_ids
    ]

    if not new_chunks:
        print("추가할 새 FAQ 청크 없음 (이미 모두 포함됨)")
        print(f"현재 총 청크 수: {len(chunks)}개")
        return

    chunks.extend(new_chunks)

    # ── 저장 ───────────────────────────────────────────────────────
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    faq_count_total = sum(1 for c in chunks if c.get("source_type") == "faq_manual")
    print(f"FAQ {len(new_chunks)}건 추가 완료")
    print(f"총 청크: {len(chunks)}개 (원본 크롤링: {len(chunks)-faq_count_total}개 + FAQ: {faq_count_total}개)")
    print(f"저장 경로: {CHUNKS_PATH}")


if __name__ == "__main__":
    reset_flag = "--reset" in sys.argv
    inject(reset=reset_flag)
