"""
한국어 임베딩 모델 래퍼 — KoreanEmbedder
──────────────────────────────────────────
[사용 경로]  build_db.py → CNUVectorStore.add_documents() → KoreanEmbedder.embed()
             retriever.py → CNUVectorStore.search() → KoreanEmbedder.embed_query()
[Colab 관계] Colab 노트북은 src.rag.pipeline을 import → 이 파일도 자동 사용됨
[모델 변경]  src/embedding/config.py의 모델·차원 계약 변경 — 비교 테스트는 tests/test_embedding.py

모델 선택 근거 (nlpai-lab/KURE-v1):
  - BGE-M3 기반 한국어 특화 파인튜닝, 1024d
  - 비대칭형 Retrieval 특화 (짧은 쿼리 ↔ 긴 문서 매칭에 강함)
  - MTEB 한국어 검색 벤치마크 상위권
  - KoE5(768d) 대비 RAG 검색 recall ↑, NDCG ↑
  - 참고: 유사도 테스트(test_embedding.py)에서는 점수가 낮게 나올 수 있음
    → 비대칭 retrieval 모델 특성 (쿼리-문서 쌍이 다름) — 실제 검색 성능은 우수

⚠️ config.py의 모델 변경 시 chroma_db를 재구축해야 함 (차원이 달라질 수 있음)
   KURE-v1: 1024d / KoE5: 768d / ko-sroberta: 768d
"""

from sentence_transformers import SentenceTransformer
import numpy as np

from src.embedding.config import EMBEDDING_DIMENSION, MODEL_NAME

# 모델 계약은 src/embedding/config.py에서 관리한다.
# 모델 또는 차원 변경 시 manifest 검사에 의해 chroma_db 전체 재구축이 요구된다.


class KoreanEmbedder:
    def __init__(self, model_name: str = MODEL_NAME):
        # SentenceTransformer: CUDA 사용 가능하면 자동 GPU 배치
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        문서 목록 → 임베딩 행렬 반환.
        build_db.py (벡터 DB 구축)에서 청크 배치 인코딩에 사용.
        normalize_embeddings=True: 벡터 정규화 → 내적 = 코사인 유사도 (검색 속도↑)
        """
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def embed_query(self, text: str) -> np.ndarray:
        """
        단일 질문 → 임베딩 벡터 반환.
        chroma_store.py의 search()에서 쿼리 인코딩에 사용.
        """
        return self.model.encode([text], normalize_embeddings=True)[0]
