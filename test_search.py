import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from src.vectordb.chroma_store import CNUVectorStore

store = CNUVectorStore(persist_dir=str(BASE_DIR / "chroma_db"))
print(f"총 저장된 청크 수: {store.collection.count()}개\n")

test_queries = [
    "장학금 신청 방법",
    "학생생활관 입주",
    "취업 지원 프로그램",
]

for query in test_queries:
    print(f"질문: {query}")
    hits = store.search(query, top_k=2)
    for i, h in enumerate(hits):
        score = h["score"]
        title = h["metadata"]["title"][:30]
        content = h["content"][:80]
        print(f"  [{i+1}] 유사도 {score:.3f} | {title}")
        print(f"       {content}")
    print()
