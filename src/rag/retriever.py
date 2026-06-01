"""
하이브리드 검색기 — BM25 (키워드) + 임베딩 (의미) 앙상블
──────────────────────────────────────────────────────────
결합 방식: RRF (Reciprocal Rank Fusion)
  score(d) = Σ 1 / (k + rank_i),  k = RRF_K = 60 (논문 권장값)
  → 두 랭킹 시스템의 순위만 반영, 점수 스케일 차이 무시

한국어 BM25 토큰화 — Kiwi 형태소 분석기 사용 이유:
  - 공백 분리(split) BM25: "장학금 신청" → ["장학금", "신청"] (어절 단위, 조사 포함)
  - Kiwi BM25: "장학금을 신청하려면" → ["장학금", "신청"] (핵심 형태소만 추출)
  - 결과: recall ↑, 노이즈 ↓

embed_score 처리 (B3 버그 수정):
  - RRF 결과 중 임베딩에는 없고 BM25에만 있는 청크: embed_score = None
  - pipeline.py에서 None을 제외하고 임계값 판단
  - 구버전(embed_score=0.0): BM25 전용 좋은 결과를 임계값 미달로 오판

_embed_search O(1) 개선 (B2 버그 수정):
  - 구버전: ChromaDB 결과 content 텍스트로 chunks 배열 순회 (O(N×M))
  - 현재: _id_to_idx 딕셔너리로 O(1) 역추적 (초기화 시 1회 구축)

[Phase2 수정] search_text 기반 BM25 인덱싱:
  - FAQ 청크: search_text = category+subcategory+title+keywords+content
    → 카테고리·키워드 IDF가 BM25 매칭에 반영됨
  - 크롤링 청크: search_text 없음 → category+title+content fallback
  - 구버전: title+content만 → keywords/subcategory 미반영

[Phase2 수정] 쿼리 의도 기반 Boost/Penalty (_get_boost_factor):
  - Kiwi 형태소로 쿼리 의도 감지 → 관련 subcategory 2.0~2.5x boost
  - 교수용·포스트닥 장학금 문서: 학생 장학금 쿼리에서 0.2x penalty
  - 주의: "계절학기" → Kiwi 분리: ["계절","학기"] → SEASON_TRIGGERS={"계절"}
           "수강신청" → Kiwi 분리: ["수강","신청"] → COURSE_TRIGGERS={"수강"}
  - boost는 RRF 결합 점수에 곱함 → FAQ 청크가 자연스럽게 상위 진입
"""

import json
from pathlib import Path
from typing import Optional

from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi

BASE_DIR = Path(__file__).parent.parent.parent
# chunks.json: BM25 인덱스 구축 + id→idx 역추적 딕셔너리에 모두 사용
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.json"

RRF_K       = 60   # RRF 상수 — 클수록 상위 랭크 쏠림 완화 (논문 권장값 60)
TOP_K_BM25  = 10   # BM25 후보 수 (최종 top_k보다 크게 — RRF에 충분한 후보 제공)
TOP_K_EMBED = 10   # 임베딩 후보 수

# [Phase1 수정] 형태소 KEEP 태그: NP(대명사) 추가 — "어디", "여기", "거기" 등 의문/장소 대명사 보존
# 구버전: NNG NNP VV VA SL XR만 유지 → "어디"(NP)가 소실되어 쿼리 의도 손실
KEEP_TAGS = frozenset({"NNG", "NNP", "VV", "VA", "SL", "XR", "NP"})

# [Phase1 수정] 범용 동사/대명사 불용어 — IDF 오염 방지
# "하"(VV): "신청하세요", "제출하다" 등 거의 모든 문서에 출현 → 변별력 없음
# "이", "그", "저": 지시대명사 — 단독으로는 의미 없음
STOPWORDS = frozenset({"하", "되", "있", "없", "않", "이", "그", "저", "것", "수", "들", "및"})

# ── [Phase2] 쿼리 의도 → Boost/Penalty 설정 ──────────────────────────────────
# 주의: Kiwi는 복합명사를 분리함
#   "계절학기" → ["계절"(NNG), "학기"(NNG)]  ∴ SEASON_TRIGGERS = {"계절"}
#   "수강신청" → ["수강"(NNG), "신청"(NNG)]  ∴ COURSE_TRIGGERS = {"수강"}
#   "장학금"   → ["장학금"(NNG)]             ∴ SCHOLAR_TRIGGERS = {"장학금", ...}

# 장학금 의도 트리거 — "신청"/"어디" 같은 범용어 제외해 타 의도 오염 방지
_SCHOLAR_TRIGGERS = frozenset({"장학금", "장학", "국가장학", "교내장학", "장학재단"})
# 장학 관련 subcategory → boost
_SCHOLAR_BOOST    = frozenset({"국가장학금", "교내장학금", "장학FAQ", "한국장학재단"})
# 교수·포스트닥 대상 장학 title 키워드 → 학생 장학금 쿼리에서 penalty
_FACULTY_TITLE_KW = frozenset({"풀브라이트", "포스트닥", "신약", "입학전형"})

# 졸업 의도 트리거
_GRAD_TRIGGERS = frozenset({"졸업", "학점", "이수", "요건"})
_GRAD_BOOST    = frozenset({"졸업요건", "졸업FAQ"})

# 계절학기 의도 트리거 (Kiwi 분리: "계절학기" → "계절"+"학기")
_SEASON_TRIGGERS = frozenset({"계절"})
_SEASON_BOOST    = frozenset({"계절학기"})

# 수강신청 의도 트리거 (Kiwi 분리: "수강신청" → "수강"+"신청")
# SEASON과 elif 관계: 계절학기 쿼리에서 수강 boost가 중복 발동하지 않도록
_COURSE_TRIGGERS = frozenset({"수강", "정정", "변경"})
_COURSE_BOOST    = frozenset({"수강신청", "수강신청변경"})


class HybridRetriever:
    """
    BM25 + 임베딩 앙상블 검색기.

    초기화 시 1회 수행:
      - chunks.json 로드 → BM25Okapi 인덱스 구축
      - _id_to_idx 딕셔너리 구축 (chunk["id"] → chunks 배열 인덱스, O(1) 역추적용)

    검색 흐름 (search 호출 시마다):
      1. BM25: 형태소 분석 → search_text 인덱스로 키워드 매칭 → TOP_K_BM25 후보
      2. 임베딩: ChromaDB 코사인 유사도 → TOP_K_EMBED 후보
      3. RRF: 두 결과 순위 결합 → boost/penalty 적용 → top_k 최종 반환

    반환 형식:
      [{"content", "metadata", "score"(RRF), "embed_score"(유사도 or None)}]
      embed_score=None: BM25에만 있는 청크 (임베딩 검색 미포함 — 측정 불가)
    """

    def __init__(self, vectorstore, chunks_path: Optional[Path] = None):
        self.vectorstore = vectorstore
        self._kiwi = Kiwi()
        # _build_bm25: BM25 인덱스 + _id_to_idx 딕셔너리 동시 구축
        self._chunks, self._bm25 = self._build_bm25(chunks_path or CHUNKS_PATH)

    # ── BM25 인덱스 구축 ──────────────────────────────────────

    def _tokenize(self, text: str) -> list[str]:
        """
        Kiwi 형태소 분석 → 핵심 형태소 추출.
        품사 태그 (KEEP_TAGS):
          NNG=일반명사, NNP=고유명사, VV=동사, VA=형용사, SL=외국어, XR=어근
          NP=대명사 [Phase1 추가] — "어디", "여기", "어느" 등 의문/장소 대명사 보존
        불용어 (STOPWORDS) 제거:
          "하", "되", "있" 등 IDF 낮은 범용 동사 — 변별력 없어 BM25 노이즈 유발
        tokens가 없으면 공백 split fallback (Kiwi 인식 불가 텍스트 대비).
        """
        tokens = []
        for token in self._kiwi.tokenize(text):
            if token.tag in KEEP_TAGS and token.form not in STOPWORDS:
                tokens.append(token.form)
        return tokens if tokens else text.split()

    def _get_index_text(self, chunk: dict) -> str:
        """
        BM25 인덱싱용 텍스트 결정 (우선순위 순).

        [Phase2 수정]
          1순위: chunk["search_text"] — FAQ 청크
                 category+subcategory+title+keywords+content
                 → 카테고리·키워드 IDF가 BM25 매칭에 반영
          2순위: category+title+content — 크롤링 청크 fallback
                 (구버전: title+content만 → category 추가로 소폭 개선)
        """
        if chunk.get("search_text"):
            return chunk["search_text"]
        parts = [
            chunk.get("category", ""),
            chunk.get("title", ""),
            chunk.get("content", ""),
        ]
        return " ".join(p for p in parts if p)

    def _build_bm25(self, chunks_path: Path):
        """
        chunks.json → BM25Okapi 인덱스 + _id_to_idx 딕셔너리 구축.

        _id_to_idx: {"chunk_id": array_index, ...}
          - _embed_search에서 ChromaDB가 반환하는 id로 chunks 배열 위치를 O(1) 조회

        [Phase2 수정] search_text 우선 인덱싱:
          - FAQ 청크: search_text(category+subcategory+title+keywords+content)로 인덱싱
          - 크롤링 청크: search_text 없음 → category+title+content fallback
          - 구버전: title+content 고정
        """
        with open(chunks_path, encoding="utf-8") as f:
            chunks = json.load(f)

        tokenized = [
            self._tokenize(self._get_index_text(c))
            for c in chunks
        ]
        bm25 = BM25Okapi(tokenized)
        # id → chunks 배열 인덱스 딕셔너리 (O(1) 역추적)
        self._id_to_idx = {c["id"]: i for i, c in enumerate(chunks)}

        faq_count = sum(1 for c in chunks if c.get("source_type") == "faq_manual")
        print(f"BM25 인덱스 구축 완료: {len(chunks)}청크 (크롤링 {len(chunks)-faq_count}건 + FAQ {faq_count}건)")
        return chunks, bm25

    # ── Boost/Penalty ──────────────────────────────────────────

    def _get_boost_factor(self, query_tokens: set, chunk: dict) -> float:
        """
        쿼리 형태소 토큰 × 청크 메타데이터 → RRF 점수 배율 반환.

        boost(2.0~2.5x): 쿼리 의도와 청크 subcategory가 일치하는 FAQ 청크
        penalty(0.2x):   장학금 쿼리에서 교수·포스트닥 대상 문서 (학생 무관)
        neutral(1.0x):   위 조건에 해당하지 않는 모든 청크

        subcategory 없는 크롤링 청크:
          - boost 조건 미충족 → 1.0x (중립)
          - title에 교수용 키워드 있으면 0.2x penalty 가능

        의도 감지 우선순위:
          1. 장학금 트리거 — 교수용 penalty 포함
          2. 졸업 트리거
          3. 계절학기 트리거  ← 수강보다 먼저 (elif로 중복 방지)
          4. 수강신청 트리거
        """
        sub   = chunk.get("subcategory", "")
        title = chunk.get("title", "")

        # ① 장학금 의도
        if query_tokens & _SCHOLAR_TRIGGERS:
            if sub in _SCHOLAR_BOOST:
                return 2.5
            # subcategory 없는 기존 크롤링 청크 중 교수·포스트닥 장학에 penalty
            if not sub and any(kw in title for kw in _FACULTY_TITLE_KW):
                return 0.2

        # ② 졸업 의도
        if query_tokens & _GRAD_TRIGGERS:
            if sub in _GRAD_BOOST:
                return 2.5

        # ③ 계절학기 의도 ("계절" 토큰 — Kiwi가 "계절학기"를 "계절"+"학기"로 분리)
        if query_tokens & _SEASON_TRIGGERS:
            if sub in _SEASON_BOOST:
                return 2.5

        # ④ 수강신청 의도 ("수강" 토큰 — Kiwi가 "수강신청"을 "수강"+"신청"으로 분리)
        # 계절학기와 elif: 계절학기 쿼리에서 수강신청 boost가 중복 발동하지 않도록
        elif query_tokens & _COURSE_TRIGGERS:
            if sub in _COURSE_BOOST:
                return 2.0

        return 1.0  # neutral

    # ── 검색 ──────────────────────────────────────────────────

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        """
        BM25 키워드 검색.
        쿼리 형태소 분석 → BM25 점수 계산 → 상위 top_k 반환.
        반환: [{"chunk_idx": int, "bm25_score": float}]
        """
        q_tokens = self._tokenize(query)
        scores   = self._bm25.get_scores(q_tokens)
        ranked   = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"chunk_idx": idx, "bm25_score": score} for idx, score in ranked]

    def _embed_search(self, query: str, top_k: int) -> list[dict]:
        """
        임베딩(ChromaDB) 검색.
        ChromaDB 결과의 id로 _id_to_idx에서 chunks 배열 위치를 O(1) 역추적.
        """
        hits    = self.vectorstore.search(query, top_k=top_k)
        results = []
        for h in hits:
            idx = self._id_to_idx.get(h["id"])   # O(1) 역추적
            if idx is not None:
                results.append({
                    "chunk_idx":   idx,
                    "embed_score": h["score"],
                    "content":     h["content"],
                    "metadata":    h["metadata"],
                })
        return results

    def _rrf(
        self,
        bm25_hits:  list[dict],
        embed_hits: list[dict],
        top_k:      int,
        query:      str = "",
    ) -> list[dict]:
        """
        RRF(Reciprocal Rank Fusion)로 BM25·임베딩 결과 순위 결합.
        score(d) = Σ 1 / (RRF_K + rank_i)

        [Phase2 수정] boost/penalty 적용:
          - query 파라미터로 쿼리 형태소 분석 → _get_boost_factor 호출
          - RRF 점수에 boost 계수를 곱해 재정렬
          - FAQ 관련 청크: 2.0~2.5x → 자연스럽게 상위 진입
          - 교수용 장학 문서: 0.2x → 학생 장학금 쿼리에서 하위로 밀림

        embed_score 처리:
          - 임베딩에 있는 청크: 실제 코사인 유사도 (float)
          - BM25 전용 청크: None — 임베딩으로 측정 불가
            → pipeline.py에서 None 제외 후 max() → 불필요한 "모릅니다" 방지

        metadata에 subcategory 추가 (Phase2):
          - pipeline.py의 retrieve()가 컨텍스트 구성 시 subcategory 활용 가능
        """
        rrf_scores: dict[int, float] = {}

        for rank, hit in enumerate(bm25_hits):
            idx = hit["chunk_idx"]
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (RRF_K + rank + 1)

        for rank, hit in enumerate(embed_hits):
            idx = hit["chunk_idx"]
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (RRF_K + rank + 1)

        # [Phase2] boost/penalty 적용 — query가 있을 때만 실행
        if query:
            q_tokens = set(self._tokenize(query))
            for idx in rrf_scores:
                rrf_scores[idx] *= self._get_boost_factor(q_tokens, self._chunks[idx])

        # RRF 점수 기준 상위 top_k 선택
        top_idxs = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)[:top_k]

        # embed_score 맵 (없으면 None — BM25 전용 결과)
        embed_score_map = {h["chunk_idx"]: h["embed_score"] for h in embed_hits}

        results = []
        for idx in top_idxs:
            chunk = self._chunks[idx]
            results.append({
                "content":  chunk["content"],
                "metadata": {
                    "title":       chunk.get("title", ""),
                    "category":    chunk.get("category", ""),
                    # [Phase2 추가] subcategory — boost 판단 근거 / 컨텍스트 표시용
                    "subcategory": chunk.get("subcategory", ""),
                    "url":         chunk.get("url", ""),
                },
                "score":       rrf_scores[idx],           # RRF + boost 결합 점수
                "embed_score": embed_score_map.get(idx),  # 임베딩 유사도 (BM25 전용=None)
            })
        return results

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        하이브리드 검색 메인 메서드. pipeline.py의 retrieve()에서 호출.
        BM25 + 임베딩 결과를 RRF로 결합해 top_k 반환.
        반환: [{"content", "metadata", "score"(RRF+boost), "embed_score"(float or None)}]

        [Phase2 수정] query를 _rrf에 전달 → boost/penalty 적용
        """
        bm25_hits  = self._bm25_search(query, TOP_K_BM25)
        embed_hits = self._embed_search(query, TOP_K_EMBED)
        return self._rrf(bm25_hits, embed_hits, top_k, query=query)
