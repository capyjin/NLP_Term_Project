"""
벡터 DB 구축 스크립트
입력: data/processed/chunks.json
출력: chroma_db/ (ChromaDB 영구 저장소)
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.vectordb.chroma_store import CNUVectorStore

CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.json"
DB_PATH = str(BASE_DIR / "chroma_db")


def build():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"청크 로드: {len(chunks)}건")

    store = CNUVectorStore(persist_dir=DB_PATH)

    docs = [
        {
            "id": chunk["id"],
            "content": chunk["content"],
            "metadata": {
                "title": chunk.get("title", ""),
                "category": chunk.get("category", ""),
                "url": chunk.get("url", ""),
            },
        }
        for chunk in chunks
    ]

    store.add_documents(docs)
    print(f"벡터 DB 구축 완료 → {DB_PATH}")


if __name__ == "__main__":
    build()
