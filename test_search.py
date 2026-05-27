import sys
sys.path.insert(0, r"C:\충남대학교\3-1\자연어처리\Termproject")
from src.vectordb.chroma_store import CNUVectorStore

store = CNUVectorStore(persist_dir=r"C:\충남대학교\3-1\자연어처리\Termproject\chroma_db")
hits = store.search("장학금 신청 방법", top_k=3)
for h in hits:
    score = h["score"]
    title = h["metadata"]["title"][:30]
    content = h["content"][:100]
    print(f"[{score:.3f}] {title}")
    print(f"  {content}")
    print()
