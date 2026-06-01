"""
RAG 성능 파라미터 튜닝 테스트
────────────────────────────────────────────
사용법: 상단 PARAMS 블록에서 값 바꾸고 실행
  C:\...\nlp_project\python.exe tests/test_params.py

테스트 항목:
  [1] 청크 크기별 검색 품질 비교
  [2] 유사도 임계값별 응답 범위
  [3] TOP_K 변화에 따른 커버리지
  [4] BM25 단독 vs 임베딩 단독 vs 하이브리드(RRF) 비교
"""

import json, sys, time
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

# ┌─────────────────────────────────────────────────────────┐
# │  ▼▼▼  여기 값들 바꿔가면서 테스트  ▼▼▼                  │

EMBED_MODEL   = "nlpai-lab/KURE-v1"  # 임베딩 모델
# EMBED_MODEL = "nlpai-lab/KoE5"

CHUNK_SIZE    = 400    # 청크 크기 (기본값)   → preprocess.py 수정 후 재크롤링 필요
CHUNK_OVERLAP = 50     # 청크 오버랩 (기본값)

SIM_THRESHOLD = 0.40   # 임계값: 이 미만이면 "모릅니다" 답변

TOP_K         = 3      # 검색 결과 개수
TOP_K_BM25    = 10     # BM25 후보 수 (RRF용)
TOP_K_EMBED   = 10     # 임베딩 후보 수 (RRF용)
RRF_K         = 60     # RRF 상수 (클수록 상위 랭크 집중도 완화)

# └─────────────────────────────────────────────────────────┘

QUERIES = [
    "장학금 신청 방법",
    "수강신청은 언제 해야 하나요",
    "기숙사 입주 신청",
    "취업 지원 프로그램",
    "졸업 요건은 어떻게 되나요",
]


def load_chunks() -> list[dict]:
    path = BASE_DIR / "data" / "processed" / "chunks.json"
    if not path.exists():
        print("⚠ chunks.json 없음")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── 테스트 1: 임계값별 응답 범위 ─────────────────────────

def test_threshold(corpus_embs, chunks, model):
    """
    임계값을 바꿔가며 "모릅니다" 비율 확인.
    너무 높으면 실제 답 있는데도 모른다고 함 → 퀄리티 손해
    너무 낮으면 관련없는 내용도 답함 → 할루시네이션 위험
    """
    print("\n[1] 유사도 임계값별 응답 범위")
    print(f"  현재 설정: SIM_THRESHOLD = {SIM_THRESHOLD}")
    print(f"\n  {'임계값':>6}  {'답변가능':>8}  {'모릅니다':>8}")
    print("  " + "─" * 28)

    for threshold in [0.20, 0.30, 0.40, 0.45, 0.50, 0.55]:
        can_answer = 0
        for query in QUERIES:
            q_emb = model.encode([query], normalize_embeddings=True)[0]
            scores = corpus_embs @ q_emb
            best = float(scores.max())
            if best >= threshold:
                can_answer += 1
        no_answer = len(QUERIES) - can_answer
        marker = " ◀ 현재" if abs(threshold - SIM_THRESHOLD) < 0.001 else ""
        print(f"  {threshold:>6.2f}   {can_answer:>4}/{len(QUERIES)}      {no_answer:>4}/{len(QUERIES)}{marker}")


# ── 테스트 2: TOP_K 변화에 따른 커버리지 ─────────────────

def test_topk(corpus_embs, chunks, model):
    """
    TOP_K가 클수록 컨텍스트 풍부 → 정확도↑, 하지만 LLM 입력 길이↑ (속도↓)
    """
    print("\n[2] TOP_K별 검색 커버리지")
    print(f"  현재 설정: TOP_K = {TOP_K}")
    print(f"\n  {'TOP_K':>6}  {'평균 최고유사도':>14}  {'평균 최저유사도':>14}")
    print("  " + "─" * 40)

    for k in [1, 2, 3, 5, 7]:
        top_scores, bot_scores = [], []
        for query in QUERIES:
            q_emb = model.encode([query], normalize_embeddings=True)[0]
            scores = corpus_embs @ q_emb
            top_k_scores = np.sort(scores)[::-1][:k]
            top_scores.append(float(top_k_scores[0]))
            bot_scores.append(float(top_k_scores[-1]))
        marker = " ◀ 현재" if k == TOP_K else ""
        print(f"  {k:>6}   {np.mean(top_scores):>14.4f}   {np.mean(bot_scores):>14.4f}{marker}")


# ── 테스트 3: BM25 vs 임베딩 vs 하이브리드(RRF) ──────────

def test_hybrid(corpus_embs, chunks, model):
    """
    세 가지 검색 방식 결과를 질문별로 비교.
    같은 질문에 대해 어떤 방식이 더 관련 있는 청크를 가져오는지 육안으로 확인.
    """
    print("\n[3] BM25 vs 임베딩 vs 하이브리드 검색 비교")

    try:
        from kiwipiepy import Kiwi
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("  ⚠ kiwipiepy / rank_bm25 미설치 — pip install kiwipiepy rank-bm25")
        return

    kiwi = Kiwi()

    def tokenize(text: str) -> list[str]:
        tokens = []
        for t in kiwi.tokenize(text):
            if t.tag in ("NNG", "NNP", "VV", "VA", "SL"):
                tokens.append(t.form)
        return tokens or text.split()

    contents = [c["content"] for c in chunks]
    print("  BM25 인덱스 구축 중... ", end="", flush=True)
    t0 = time.time()
    tokenized = [tokenize(c) for c in contents]
    bm25 = BM25Okapi(tokenized)
    print(f"{time.time()-t0:.2f}초")

    for query in QUERIES[:3]:   # 3개만 출력
        print(f"\n  📌 질문: {query}")

        # BM25 top-3
        q_tokens = tokenize(query)
        bm25_scores = bm25.get_scores(q_tokens)
        bm25_top3 = np.argsort(bm25_scores)[::-1][:3]

        # 임베딩 top-3
        q_emb = model.encode([query], normalize_embeddings=True)[0]
        emb_scores = corpus_embs @ q_emb
        emb_top3 = np.argsort(emb_scores)[::-1][:3]

        # RRF 하이브리드
        rrf: dict[int, float] = {}
        for rank, idx in enumerate(np.argsort(bm25_scores)[::-1][:TOP_K_BM25]):
            rrf[int(idx)] = rrf.get(int(idx), 0) + 1 / (RRF_K + rank + 1)
        for rank, idx in enumerate(np.argsort(emb_scores)[::-1][:TOP_K_EMBED]):
            rrf[int(idx)] = rrf.get(int(idx), 0) + 1 / (RRF_K + rank + 1)
        hybrid_top3 = sorted(rrf, key=lambda i: rrf[i], reverse=True)[:3]

        print(f"    {'방식':<12} {'청크 제목':<28} {'점수':>8}")
        print("    " + "─" * 52)
        for idx in bm25_top3:
            title = chunks[idx].get("title", "")[:25]
            print(f"    {'BM25':<12} {title:<28} {bm25_scores[idx]:>8.4f}")
        for idx in emb_top3:
            title = chunks[idx].get("title", "")[:25]
            print(f"    {'임베딩':<12} {title:<28} {emb_scores[idx]:>8.4f}")
        for idx in hybrid_top3:
            title = chunks[idx].get("title", "")[:25]
            print(f"    {'하이브리드':<12} {title:<28} {rrf[idx]:>8.4f}")


# ── 실행 ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("RAG 파라미터 튜닝 테스트")
    print(f"  임베딩 모델: {EMBED_MODEL}")
    print(f"  청크 크기  : {CHUNK_SIZE} (변경은 preprocess.py 수정 후 재실행)")
    print(f"  임계값     : {SIM_THRESHOLD}")
    print(f"  TOP_K      : {TOP_K}  |  RRF_K: {RRF_K}")
    print("=" * 60)

    from sentence_transformers import SentenceTransformer

    chunks = load_chunks()
    contents = [c["content"] for c in chunks]

    print(f"\n청크 {len(chunks)}개 임베딩 중... ", end="", flush=True)
    t0 = time.time()
    model = SentenceTransformer(EMBED_MODEL)
    corpus_embs = model.encode(contents, normalize_embeddings=True,
                               show_progress_bar=False, batch_size=32)
    print(f"{time.time()-t0:.2f}초")

    test_threshold(corpus_embs, chunks, model)
    test_topk(corpus_embs, chunks, model)
    test_hybrid(corpus_embs, chunks, model)

    print("\n" + "=" * 60)
    print("상단 PARAMS 블록 수치 바꾸고 다시 실행하면 비교 가능")
    print("=" * 60)
