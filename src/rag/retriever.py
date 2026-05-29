"""
하이브리드 검색기 — BM25 (키워드) + 임베딩 (의미) 앙상블
결합 방식: RRF (Reciprocal Rank Fusion)
  score(d) = Σ 1 / (k + rank_i),  k=60 (standard)

한국어 BM25는 Kiwi 형태소 분석기로 토큰화.
  - 형태소 분석 없이 BM25 쓰면 한국어 성능 저하
  - Kiwi는 경량·빠름 (~50ms/문장)
"""

import json
from pathlib import Path
from typing import Optional

from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi

BASE_DIR = Path(__file__).parent.parent.parent
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.json"

RRF_K = 60          # RRF 상수 (클수록 상위 랭크 쏠림 완화)
TOP_K_BM25 = 10     # BM25에서 후보 몇 개 뽑을지 (RRF 입력용)
TOP_K_EMBED = 10    # 임베딩에서 후보 몇 개 뽑을지


class HybridRetriever:
    """
    BM25 + 임베딩 앙상블 검색기.

    사용법:
        retriever = HybridRetriever(vectorstore)
        hits = retriever.search("장학금 신청 방법", top_k=3)
        # hits: [{"content": ..., "metadata": ..., "score": float, "embed_score": float}]
    """

    def __init__(self, vectorstore, chunks_path: Optional[Path] = None):
        self.vectorstore = vectorstore
        self._kiwi = Kiwi()
        self._chunks, self._bm25 = self._build_bm25(chunks_path or CHUNKS_PATH)

    # ── BM25 인덱스 구축 ──────────────────────────────────────

    def _tokenize(self, text: str) -> list[str]:
        """Kiwi 형태소 분석 → 명사·동사·형용사 추출."""
        tokens = []
        for token in self._kiwi.tokenize(text):
            # NNG(일반명사), NNP(고유명사), VV(동사), VA(형용사), SL(외국어) 선택
            if token.tag in ("NNG", "NNP", "VV", "VA", "SL", "XR"):
                tokens.append(token.form)
        return tokens if tokens else text.split()   # fallback: 공백 분리

    def _build_bm25(self, chunks_path: Path):
        with open(chunks_path, encoding="utf-8") as f:
            chunks = json.load(f)
        tokenized = [self._tokenize(c["content"]) for c in chunks]
        bm25 = BM25Okapi(tokenized)
        print(f"BM25 인덱스 구축 완료: {len(chunks)}청크")
        return chunks, bm25

    # ── 검색 ──────────────────────────────────────────────────

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        """BM25 검색 → [{"chunk_idx": int, "score": float}] 반환."""
        q_tokens = self._tokenize(query)
        scores = self._bm25.get_scores(q_tokens)
        # 점수 높은 순 정렬
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"chunk_idx": idx, "bm25_score": score} for idx, score in ranked]

    def _embed_search(self, query: str, top_k: int) -> list[dict]:
        """임베딩 검색 → [{"chunk_idx": int, "embed_score": float, "metadata": dict, "content": str}]"""
        hits = self.vectorstore.search(query, top_k=top_k)
        results = []
        for h in hits:
            # ChromaDB hit의 content로 chunks 배열에서 idx 찾기
            for i, chunk in enumerate(self._chunks):
                if chunk["content"] == h["content"]:
                    results.append({
                        "chunk_idx": i,
                        "embed_score": h["score"],
                        "content": h["content"],
                        "metadata": h["metadata"],
                    })
                    break
        return results

    def _rrf(self, bm25_hits: list[dict], embed_hits: list[dict], top_k: int) -> list[dict]:
        """RRF로 두 랭킹 결합."""
        rrf_scores: dict[int, float] = {}

        for rank, hit in enumerate(bm25_hits):
            idx = hit["chunk_idx"]
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (RRF_K + rank + 1)

        for rank, hit in enumerate(embed_hits):
            idx = hit["chunk_idx"]
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (RRF_K + rank + 1)

        # 상위 top_k 선택
        top_idxs = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)[:top_k]

        # 임베딩 점수 맵 (할루시네이션 임계값 판단용)
        embed_score_map = {h["chunk_idx"]: h["embed_score"] for h in embed_hits}

        results = []
        for idx in top_idxs:
            chunk = self._chunks[idx]
            results.append({
                "content": chunk["content"],
                "metadata": {
                    "title": chunk.get("title", ""),
                    "category": chunk.get("category", ""),
                    "url": chunk.get("url", ""),
                },
                "score": rrf_scores[idx],           # RRF 결합 점수
                "embed_score": embed_score_map.get(idx, 0.0),  # 임베딩 유사도 (임계값용)
            })
        return results

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        하이브리드 검색 실행.
        반환: [{"content", "metadata", "score"(RRF), "embed_score"(임베딩 유사도)}]
        """
        bm25_hits = self._bm25_search(query, TOP_K_BM25)
        embed_hits = self._embed_search(query, TOP_K_EMBED)
        return self._rrf(bm25_hits, embed_hits, top_k)
