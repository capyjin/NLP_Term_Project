"""
ChromaDB 벡터 스토어 — 문서 색인 및 검색
"""

import chromadb
from chromadb.config import Settings
import json
from pathlib import Path
from src.embedding.embedder import KoreanEmbedder

DB_PATH = "./chroma_db"
COLLECTION_NAME = "cnu_docs"


class CNUVectorStore:
    def __init__(self, persist_dir: str = DB_PATH):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = KoreanEmbedder()

    def add_documents(self, docs: list[dict]):
        """docs: [{"id": str, "content": str, "metadata": dict}]"""
        ids = [d["id"] for d in docs]
        contents = [d["content"] for d in docs]
        metadatas = [d.get("metadata", {}) for d in docs]
        embeddings = self.embedder.embed(contents).tolist()
        self.collection.add(
            ids=ids,
            documents=contents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        print(f"문서 {len(docs)}건 추가 완료")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        q_emb = self.embedder.embed_query(query).tolist()
        results = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for i in range(len(results["documents"][0])):
            hits.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i],
            })
        return hits

    def load_from_json(self, json_path: str):
        with open(json_path, encoding="utf-8") as f:
            raw = json.load(f)
        docs = []
        for i, item in enumerate(raw):
            content = item.get("content", "") or item.get("answer", "")
            if not content.strip():
                continue
            docs.append({
                "id": f"doc_{i}",
                "content": content,
                "metadata": {
                    "category": item.get("category", ""),
                    "title": item.get("title", item.get("doc_title", "")),
                    "question": item.get("question", ""),
                },
            })
        self.add_documents(docs)
