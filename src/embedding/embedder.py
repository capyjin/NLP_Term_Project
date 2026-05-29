"""
한국어 임베딩 모델 래퍼
jhgan/ko-sroberta-multitask : 한국어 특화 sentence-transformer (작고 빠름)
"""
"""

"""

from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "jhgan/ko-sroberta-multitask"


class KoreanEmbedder:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode([text], normalize_embeddings=True)[0]
