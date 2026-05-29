import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from src.vectordb.chroma_store import CNUVectorStore
from src.rag.retriever import HybridRetriever

store = CNUVectorStore(persist_dir=str(BASE_DIR / "chroma_db"))
retriever = HybridRetriever(store)

queries = [
    "장학금 신청 방법",
    "학생생활관 입주",
    "취업 지원 프로그램",
]

for query in queries:
    print(f"\n질문: {query}")
    hits = retriever.search(query, top_k=3)
    for i, h in enumerate(hits):
        print(f"  [{i+1}] RRF={h['score']:.4f} | 임베딩={h['embed_score']:.3f} | {h['metadata']['title'][:30]}")
        print(f"       {h['content'][:80]}")
