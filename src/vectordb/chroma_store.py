"""
ChromaDB 벡터 스토어 — 문서 임베딩 색인 및 의미 검색
──────────────────────────────────────────────────────
역할: 전처리된 청크를 KURE-v1 임베딩 벡터로 변환해 ChromaDB에 저장하고,
      질문 임베딩으로 코사인 유사도 기반 유사 청크를 검색한다.

흐름:
  색인: build_db.py → add_documents() → ChromaDB 영구 저장
  검색: retriever.py → search() → 유사 청크 반환 (id + score 포함)

[DB 경로]  BASE_DIR/chroma_db/ — Path(__file__) 기반 절대 경로 (CWD 무관)
[임베딩]   embedder.py의 KoreanEmbedder (nlpai-lab/KURE-v1, 1024d)
[유사도]   cosine distance → score = 1 - distance (1에 가까울수록 유사)

수정 이력:
  - DB_PATH: "./chroma_db" 상대경로 → Path(__file__) 기반 절대경로
  - search(): "ids" 필드 추가 반환 (retriever.py O(1) 역추적용)
  - add_documents(): collection.add() → upsert() (중복 ID 재실행 안전)
"""

import json
from pathlib import Path

import chromadb
from src.embedding.embedder import KoreanEmbedder

# chroma_store.py 위치에서 프로젝트 루트 계산 — CWD 독립
# 로컬:  C:\충남대학교\...\Termproject
# Colab: /content/drive/MyDrive/CNU-QA-chatbot
BASE_DIR = Path(__file__).parent.parent.parent
DB_PATH = str(BASE_DIR / "chroma_db")   # 절대 경로 (구: "./chroma_db" 상대경로)
COLLECTION_NAME = "cnu_docs"


class CNUVectorStore:
    """
    ChromaDB 기반 벡터 스토어.

    add_documents(docs): 청크 목록 → KURE-v1 임베딩 → ChromaDB upsert
    search(query, top_k): 질문 임베딩 → 코사인 유사도 상위 k개 반환
    """

    def __init__(self, persist_dir: str = DB_PATH):
        # PersistentClient: 디스크 영구 저장 — Colab 세션 재시작 후에도 DB 유지
        self.client = chromadb.PersistentClient(path=persist_dir)
        # hnsw:space=cosine: L2 거리 대신 코사인 거리 사용 (normalize=True와 일관성)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        # KoreanEmbedder: nlpai-lab/KURE-v1 (1024d) 래퍼
        self.embedder = KoreanEmbedder()

    def add_documents(self, docs: list[dict]):
        """
        청크 목록을 임베딩 후 ChromaDB에 색인.
        docs 형식: [{"id": str, "content": str, "metadata": dict}]

        upsert 사용: 동일 ID가 이미 존재해도 덮어씀 → 재실행 시 DuplicateIDError 없음
        """
        ids        = [d["id"] for d in docs]
        contents   = [d["content"] for d in docs]
        metadatas  = [d.get("metadata", {}) for d in docs]
        # [Phase1 수정] embed_text 우선 사용 (build_db.py가 title+content로 설정)
        # embed_text 없으면 content fallback (하위 호환)
        # → 저장(documents)은 원본 content 유지, 임베딩 벡터만 title+content 기준
        embed_texts = [d.get("embed_text", d["content"]) for d in docs]
        embeddings  = self.embedder.embed(embed_texts).tolist()
        # upsert: add()와 달리 중복 ID 허용 — chunks.json 변경 후 재구축 시 안전
        self.collection.upsert(
            ids=ids,
            documents=contents,   # 표시용: 원본 content (title 중복 없음)
            metadatas=metadatas,
            embeddings=embeddings, # 검색용: title+content 임베딩
        )
        print(f"문서 {len(docs)}건 색인 완료")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        질문 → KURE-v1 임베딩 → ChromaDB 코사인 유사도 검색.

        반환: [{"id", "content", "metadata", "score"}]
          id   : 청크 고유 ID — retriever.py의 _id_to_idx 딕셔너리로 O(1) 역추적
                 (구버전은 id 없이 content 텍스트 비교 O(N×M) — 현재 버전에서 수정)
          score: 코사인 유사도 = 1 - cosine_distance  (범위: 0~1)
        """
        q_emb = self.embedder.embed_query(query).tolist()
        results = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
            # "ids"는 chromadb 0.5.x 에서 include 파라미터 미지원
            # — IDs는 항상 results["ids"]에 자동 포함됨
        )
        hits = []
        for i in range(len(results["documents"][0])):
            hits.append({
                "id":       results["ids"][0][i],            # retriever.py O(1) 역추적용
                "content":  results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score":    1 - results["distances"][0][i],  # cosine distance → similarity
            })
        return hits

    def load_from_json(self, json_path: str):
        """
        [보조 메서드 — 현재 build_db.py에서 미사용]
        JSON 파일에서 직접 문서를 읽어 색인한다.
        현재 파이프라인은 build_db.py → add_documents() 경로를 사용.
        """
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
                    "title":    item.get("title", item.get("doc_title", "")),
                    "question": item.get("question", ""),
                },
            })
        self.add_documents(docs)
