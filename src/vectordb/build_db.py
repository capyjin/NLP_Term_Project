"""
벡터 DB 구축 스크립트
──────────────────────
입력: data/processed/chunks.json  (preprocess.py + inject_faq.py 출력)
출력: chroma_db/                  (ChromaDB 영구 저장소)

실행:
  python src/vectordb/build_db.py --fresh

흐름:
  chunks.json 로드
    → CNUVectorStore(persist_dir=DB_PATH) 생성  # DB 없으면 새로 생성
    → store.add_documents(docs)                  # KURE-v1 임베딩 + upsert
    → chroma_db/ 폴더에 영구 저장

embed_text 우선순위 (Phase2 수정):
  1순위: chunk["search_text"]  — FAQ 청크가 inject_faq.py에서 생성
         category + subcategory + title + keywords + content
         → 카테고리·키워드 정보가 Dense 임베딩 벡터에 반영
  2순위: category + title + content  — 크롤링 청크 fallback
         (기존 Phase1: title+content에서 category 추가)

metadata에 subcategory 추가 (Phase2 수정):
  retriever.py의 boost/penalty 로직이 subcategory로 쿼리 의도를 판단
  → ChromaDB 검색 결과에 subcategory가 포함되어야 _embed_search가 전달 가능

⚠️ inject_faq.py 실행 후 반드시 --fresh로 chroma_db 재구축
   (FAQ 청크가 embed_text 없이 color된 기존 DB와 혼용 불가)
⚠️ 임베딩 모델(config.py) 변경 시 반드시 --fresh로 재실행
   (차원 불일치 시 InvalidDimensionException 발생)
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.vectordb.chroma_store import CNUVectorStore
from src.vectordb.integrity import check_integrity, write_manifest

CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.json"
DB_PATH     = BASE_DIR / "chroma_db"


def _get_embed_text(chunk: dict) -> str:
    """
    embed_text 생성 규칙 (우선순위 순):
      1. chunk["search_text"] — FAQ 청크: category+subcategory+title+keywords+content
      2. category + title + content — 크롤링 청크 fallback (Phase2: category 추가)

    FAQ 청크는 inject_faq.py가 search_text를 미리 생성해둠.
    크롤링 청크는 search_text 없음 → category+title+content로 구성.
    """
    if chunk.get("search_text"):
        return chunk["search_text"].strip()
    # 크롤링 청크 fallback: category 추가로 Phase1 대비 소폭 개선
    parts = [
        chunk.get("category", ""),
        chunk.get("title", ""),
        chunk["content"],
    ]
    return " ".join(p for p in parts if p).strip()


def _reset_db() -> None:
    """프로젝트 내부의 고정 chroma_db 경로만 삭제한다."""
    resolved_db = DB_PATH.resolve()
    expected_db = (BASE_DIR / "chroma_db").resolve()
    if resolved_db != expected_db:
        raise RuntimeError(f"안전하지 않은 DB 경로입니다: {resolved_db}")
    if resolved_db.exists():
        shutil.rmtree(resolved_db)
        print(f"기존 벡터 DB 제거: {resolved_db}")


def build(fresh: bool = False):
    """chunks.json을 읽어 ChromaDB 벡터 DB를 구축한다."""
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)

    faq_count  = sum(1 for c in chunks if c.get("source_type") == "faq_manual")
    crawl_count = len(chunks) - faq_count
    print(f"청크 로드: {len(chunks)}건 (크롤링 {crawl_count}건 + FAQ {faq_count}건)")

    if len({chunk["id"] for chunk in chunks}) != len(chunks):
        raise ValueError("chunks.json에 중복 ID가 있어 색인을 중단합니다.")

    if fresh:
        _reset_db()

    # CNUVectorStore: DB 없으면 새로 생성, 있으면 기존 컬렉션 사용
    store = CNUVectorStore(persist_dir=str(DB_PATH))

    # chunks.json 형식 → add_documents 형식으로 변환
    docs = [
        {
            "id":      chunk["id"],
            "content": chunk["content"],      # ChromaDB 저장용 (검색 결과 표시)
            # [Phase2 수정] embed_text: search_text 우선, 없으면 category+title+content
            # FAQ: category+subcategory+title+keywords+content → 카테고리·키워드 벡터 반영
            # 크롤링: category+title+content → Phase1 대비 category 추가
            "embed_text": _get_embed_text(chunk),
            "metadata": {
                "title":       chunk.get("title", ""),
                "category":    chunk.get("category", ""),
                # [Phase2 추가] subcategory: retriever.py boost/penalty 판단에 사용
                "subcategory": chunk.get("subcategory", ""),
                "url":         chunk.get("url", ""),
            },
        }
        for chunk in chunks
    ]

    store.add_documents(docs)
    write_manifest(DB_PATH, chunks)
    errors = check_integrity(DB_PATH, chunks)
    if errors:
        raise RuntimeError("벡터 DB 구축 후 정합성 검사 실패: " + "; ".join(errors))
    print(f"벡터 DB 구축 완료 → {DB_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="고정된 프로젝트 chroma_db를 삭제한 뒤 전체 재구축",
    )
    args = parser.parse_args()
    build(fresh=args.fresh)
