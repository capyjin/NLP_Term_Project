"""
임베딩 모델 비교 테스트
────────────────────────────────────────────
사용법: MODEL_NAME 한 줄만 바꾸고 실행
  C:\...\nlp_project\python.exe tests/test_embedding.py

테스트 항목:
  [1] 문장 유사도 — 관련/무관 문장 쌍 점수 비교
  [2] 실제 검색   — chunks.json에서 top-3 조회
  [3] 속도        — 단일/배치 인코딩 ms
"""

import json, sys, time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# ┌─────────────────────────────────────────────────────────┐
# │  ▼▼▼  여기만 바꾸면 됨  ▼▼▼                             │
MODEL_NAME = "nlpai-lab/KURE-v1"  # 한국어 특화, 768d, 1.2GB VRAM, 0.5~1ms/문장
# MODEL_NAME = "nlpai-lab/KoE5"
# MODEL_NAME = "BAAI/bge-m3"
# MODEL_NAME = "snunlp/KR-ELECTRA-discriminator"
# MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# └─────────────────────────────────────────────────────────┘

BASE_DIR = Path(__file__).parent.parent


# ── 코사인 유사도 ─────────────────────────────────────────

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    # normalize=True로 인코딩하면 내적 = 코사인 유사도
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# ── 테스트 1: 문장 유사도 ─────────────────────────────────

def test_similarity(model: SentenceTransformer) -> int:
    """
    관련 문장 쌍 → 높은 유사도 기대 (>0.55)
    무관 문장 쌍 → 낮은 유사도 기대 (<0.55)
    통과 개수 반환
    """
    print("\n[1] 문장 유사도 테스트")
    print(f"{'문장 A':<30} {'문장 B':<30} {'유사도':>6}  기대")
    print("─" * 80)

    pairs = [
        ("장학금 신청 방법이 궁금합니다",    "장학금 받으려면 어떻게 해야 하나요", True),
        ("수강신청 일정이 언제인가요",        "강의 등록은 어떻게 하나요",          True),
        ("기숙사 입주 신청 방법",             "생활관 신청은 어떻게 하나요",         True),
        ("취업 지원 센터 이용 방법",          "취업 준비 프로그램 안내",             True),
        ("장학금 신청 방법이 궁금합니다",    "학생식당 운영 시간이 언제인가요",     False),
        ("기숙사 입주 신청 방법",             "오늘 날씨가 맑습니다",                False),
        ("수강신청 일정이 언제인가요",        "졸업 논문 제출 기한",                 False),
    ]

    passed = 0
    for a, b, should_be_high in pairs:
        ea = model.encode([a], normalize_embeddings=True)[0]
        eb = model.encode([b], normalize_embeddings=True)[0]
        sim = cos_sim(ea, eb)
        ok = (sim > 0.55) if should_be_high else (sim < 0.55)
        mark = "✓" if ok else "✗"
        expect_str = "높아야함" if should_be_high else "낮아야함"
        if ok:
            passed += 1
        print(f"{a[:28]:<30} {b[:28]:<30} {sim:>6.4f}  {expect_str} {mark}")

    print(f"\n  결과: {passed}/{len(pairs)} 통과")
    return passed


# ── 테스트 2: 실제 chunks.json 검색 ─────────────────────

def test_retrieval(model: SentenceTransformer):
    """실제 데이터에서 쿼리별 top-3 검색 결과 출력"""
    print("\n[2] 실제 데이터 검색 테스트")

    chunks_path = BASE_DIR / "data" / "processed" / "chunks.json"
    if not chunks_path.exists():
        print("  ⚠ chunks.json 없음 — 스킵 (data/processed/chunks.json 필요)")
        return

    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    contents = [c["content"] for c in chunks]
    print(f"  청크 {len(chunks)}개 임베딩 중... ", end="", flush=True)
    t0 = time.time()
    corpus_embs = model.encode(contents, normalize_embeddings=True,
                               show_progress_bar=False, batch_size=32)
    print(f"{time.time() - t0:.2f}초")

    queries = [
        "장학금 신청 방법",
        "수강신청은 언제 해야 하나요",
        "기숙사 입주 신청",
        "취업 지원 프로그램",
        "졸업 요건",
    ]

    for query in queries:
        q_emb = model.encode([query], normalize_embeddings=True)[0]
        scores = corpus_embs @ q_emb
        top3 = np.argsort(scores)[::-1][:3]

        print(f"\n  📌 질문: {query}")
        for rank, idx in enumerate(top3):
            title   = chunks[idx].get("title", "")[:25]
            preview = chunks[idx]["content"][:70].replace("\n", " ")
            print(f"    [{rank+1}] {scores[idx]:.4f} | {title}")
            print(f"         {preview}...")


# ── 테스트 3: 속도 벤치마크 ──────────────────────────────

def test_speed(model: SentenceTransformer):
    """단일/배치 인코딩 속도 측정"""
    print("\n[3] 속도 벤치마크")
    text = "충남대학교 장학금 신청 방법이 궁금합니다"

    t0 = time.time()
    model.encode([text], normalize_embeddings=True)
    print(f"  단일 쿼리:   {(time.time()-t0)*1000:6.1f} ms")

    t0 = time.time()
    model.encode([text] * 100, normalize_embeddings=True, batch_size=32)
    print(f"  배치 100문장: {(time.time()-t0)*1000:6.1f} ms")

    dim = model.encode([text])[0].shape[0]
    print(f"  임베딩 차원: {dim}d")


# ── 실행 ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print(f"모델: {MODEL_NAME}")
    print("=" * 60)

    print("모델 로딩 중... ", end="", flush=True)
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    print(f"{time.time()-t0:.2f}초")

    score = test_similarity(model)
    test_retrieval(model)
    test_speed(model)

    print("\n" + "=" * 60)
    print(f"종합: 유사도 테스트 {score}/7 통과")
    print("MODEL_NAME 바꾸고 다시 실행하면 비교 가능")
    print("=" * 60)
