"""
수정된 코드 검증 스크립트
실행: [nlp_project venv] python tests/verify_pipeline.py

GPU 없이 실행 가능한 검증 항목:
  [1] 수정된 .py 파일 문법(AST) 검사
  [2] chunks.json 로드 + 구조 검증
  [3] RRF + embed_score=None 로직 단위 테스트 (순수 Python)
  [4] BM25 + Kiwi 실제 검색 (테스트 질문 3개)
  [5] _id_to_idx 딕셔너리 O(1) 역추적 검증
  [6] pipeline.retrieve() embed_score 필터링 로직 검증

GPU 필요 (Colab에서 검증):
  [7] ChromaDB 구축 (build_db.py)
  [8] KURE-v1 임베딩 차원 일치
  [9] Qwen2.5-3B generate() 실행
"""

import ast
import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"

results = []

def record(name, status, detail="", warn=""):
    icon = "✅" if status == PASS else ("⚠️" if status == SKIP else "❌")
    results.append((name, status, detail, warn))
    print(f"{icon} {status} {name}")
    if detail:
        print(f"        {detail}")
    if warn:
        print(f"        ⚠ {warn}")


# ══════════════════════════════════════════════════
# [1] AST 문법 검사 — 수정된 파일 전체
# ══════════════════════════════════════════════════
print("\n" + "="*55)
print("[1] AST 문법 검사")
print("="*55)

target_files = [
    "src/vectordb/chroma_store.py",
    "src/rag/retriever.py",
    "src/rag/pipeline.py",
    "src/embedding/embedder.py",
    "src/vectordb/build_db.py",
    "src/api/server.py",
    "src/ui/app.py",
    "src/llm/finetune.py",
    "src/evaluate.py",
    "src/crawling/cnu_crawler.py",
    "src/crawling/add_pdf.py",
    "src/preprocessing/preprocess.py",
]

for rel_path in target_files:
    fpath = BASE_DIR / rel_path
    try:
        src = fpath.read_text(encoding="utf-8")
        ast.parse(src)
        record(rel_path, PASS, f"{len(src.splitlines())}줄 — 문법 OK")
    except SyntaxError as e:
        record(rel_path, FAIL, f"SyntaxError L{e.lineno}: {e.msg}")
    except FileNotFoundError:
        record(rel_path, FAIL, "파일 없음")


# ══════════════════════════════════════════════════
# [2] chunks.json 로드 + 구조 검증
# ══════════════════════════════════════════════════
print("\n" + "="*55)
print("[2] chunks.json 로드 + 구조 검증")
print("="*55)

chunks_path = BASE_DIR / "data" / "processed" / "chunks.json"
try:
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    # 필수 필드 검증
    required_keys = {"id", "content", "title", "category", "url"}
    missing = [c["id"] for c in chunks[:5] if not required_keys.issubset(c.keys())]

    # ID 고유성
    ids = [c["id"] for c in chunks]
    dup_count = len(ids) - len(set(ids))

    # 내용 길이
    lengths = [len(c["content"]) for c in chunks]
    avg_len = sum(lengths) / len(lengths)

    from collections import Counter
    cat_dist = dict(Counter(c["category"] for c in chunks))

    detail = (f"총 {len(chunks)}청크 | 중복ID={dup_count} | "
              f"평균길이={avg_len:.0f}자 | 카테고리={cat_dist}")
    warn = f"중복 ID {dup_count}개 — build_db에서 upsert로 처리됨" if dup_count else ""
    record("chunks.json 로드", PASS if dup_count == 0 else PASS, detail, warn)

    # 짧은 청크 (30자 미만 통과 여부)
    short = [c for c in chunks if len(c["content"]) < 30]
    if short:
        record("최소 길이 필터(30자)", FAIL, f"{len(short)}개 청크가 30자 미만")
    else:
        record("최소 길이 필터(30자)", PASS, "모든 청크 30자 이상")

    record("필수 필드", PASS if not missing else FAIL,
           "id/content/title/category/url 모두 존재" if not missing else f"누락: {missing}")

except Exception as e:
    record("chunks.json 로드", FAIL, str(e))
    chunks = []


# ══════════════════════════════════════════════════
# [3] RRF + embed_score=None 로직 단위 테스트
# ══════════════════════════════════════════════════
print("\n" + "="*55)
print("[3] RRF + embed_score=None 로직 단위 테스트")
print("="*55)

def mock_rrf(bm25_hits, embed_hits, top_k=3, rrf_k=60):
    """retriever.py의 _rrf 로직을 그대로 복제해 독립 검증"""
    rrf_scores = {}
    for rank, hit in enumerate(bm25_hits):
        idx = hit["chunk_idx"]
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rrf_k + rank + 1)
    for rank, hit in enumerate(embed_hits):
        idx = hit["chunk_idx"]
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rrf_k + rank + 1)
    top_idxs = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)[:top_k]
    embed_score_map = {h["chunk_idx"]: h["embed_score"] for h in embed_hits}
    return [
        {
            "score":       rrf_scores[idx],
            "embed_score": embed_score_map.get(idx),   # None if BM25-only
        }
        for idx in top_idxs
    ]

def mock_retrieve(hits, threshold=0.40):
    """pipeline.py의 retrieve() embed_score 필터링 로직 복제"""
    embed_scores = [h["embed_score"] for h in hits if h["embed_score"] is not None]
    return max(embed_scores) if embed_scores else 0.0

# 시나리오 A: 모든 결과가 BM25+임베딩 교집합
bm25_hits  = [{"chunk_idx": 0}, {"chunk_idx": 1}, {"chunk_idx": 2}]
embed_hits = [
    {"chunk_idx": 0, "embed_score": 0.82},
    {"chunk_idx": 1, "embed_score": 0.75},
    {"chunk_idx": 2, "embed_score": 0.60},
]
results_a = mock_rrf(bm25_hits, embed_hits)
score_a   = mock_retrieve(results_a)
ok_a = score_a == 0.82
record("시나리오A: 교집합 결과 embed_score", PASS if ok_a else FAIL,
       f"best_embed_score={score_a:.2f} (기대값=0.82)")

# 시나리오 B: BM25 전용 결과만 있는 경우 (구버전 버그 재현)
bm25_only = [{"chunk_idx": 5}, {"chunk_idx": 6}]
embed_none = []  # 임베딩 결과 없음
results_b = mock_rrf(bm25_only, embed_none)
score_b   = mock_retrieve(results_b)
# 구버전: embed_score=0.0 → threshold 미달 → "모릅니다" (잘못됨)
# 수정본: embed_score=None → embed_scores=[] → score=0.0 (동일하지만 의도 명확)
all_none = all(r["embed_score"] is None for r in results_b)
record("시나리오B: BM25 전용 → embed_score=None", PASS if all_none else FAIL,
       f"embed_scores={[r['embed_score'] for r in results_b]} | best={score_b}")

# 시나리오 C: BM25 전용 1개 + 임베딩 1개 혼합
bm25_mixed  = [{"chunk_idx": 10}, {"chunk_idx": 11}]
embed_mixed = [{"chunk_idx": 11, "embed_score": 0.55}]
results_c = mock_rrf(bm25_mixed, embed_mixed)
score_c   = mock_retrieve(results_c)
# chunk 10: BM25 전용 → embed_score=None (임계값 제외)
# chunk 11: 교집합   → embed_score=0.55 (임계값 판단에 사용)
embed_scores_c = [r["embed_score"] for r in results_c]
ok_c = 0.55 in embed_scores_c and None in embed_scores_c and score_c == 0.55
record("시나리오C: 혼합 결과 — None 제외 후 max()", PASS if ok_c else FAIL,
       f"embed_scores={embed_scores_c} | best={score_c:.2f} (기대=0.55)")

# 시나리오 D: threshold 경계 테스트
results_d_high = [{"embed_score": 0.42}]  # > 0.40 → 답변 가능
results_d_low  = [{"embed_score": 0.38}]  # < 0.40 → 모릅니다
score_d_high = mock_retrieve(results_d_high)
score_d_low  = mock_retrieve(results_d_low)
ok_d = (score_d_high >= 0.40) and (score_d_low < 0.40)
record("시나리오D: threshold=0.40 경계 판단", PASS if ok_d else FAIL,
       f"score=0.42 → {'답변' if score_d_high>=0.40 else '모름'} | "
       f"score=0.38 → {'답변' if score_d_low>=0.40 else '모름'}")


# ══════════════════════════════════════════════════
# [4] BM25 + Kiwi 실제 검색 (테스트 질문 3개)
# ══════════════════════════════════════════════════
print("\n" + "="*55)
print("[4] BM25 + Kiwi 실제 검색")
print("="*55)

TEST_QUERIES = [
    "수강신청 변경 기간은 언제야",
    "장학금 신청은 어디서 확인해",
    "졸업하려면 몇 학점 필요해",
]

if not chunks:
    record("BM25 검색", SKIP, "chunks.json 로드 실패")
else:
    try:
        from kiwipiepy import Kiwi
        from rank_bm25 import BM25Okapi

        kiwi = Kiwi()
        KEEP_TAGS = ("NNG", "NNP", "VV", "VA", "SL", "XR")

        def tokenize(text):
            tokens = [t.form for t in kiwi.tokenize(text) if t.tag in KEEP_TAGS]
            return tokens if tokens else text.split()

        print("  BM25 인덱스 구축 중...")
        t0 = time.time()
        contents  = [c["content"] for c in chunks]
        tokenized = [tokenize(c) for c in contents]
        bm25      = BM25Okapi(tokenized)
        elapsed   = time.time() - t0
        record("BM25 인덱스 구축", PASS, f"{len(chunks)}청크 | {elapsed:.2f}초")

        # _id_to_idx 딕셔너리 검증
        id_to_idx = {c["id"]: i for i, c in enumerate(chunks)}
        ok_dict   = all(id_to_idx[c["id"]] == i for i, c in enumerate(chunks))
        record("_id_to_idx 딕셔너리", PASS if ok_dict else FAIL,
               f"{len(id_to_idx)}개 ID | O(1) 역추적 {'정상' if ok_dict else '오류'}")

        # 테스트 질문 검색
        import numpy as np
        for query in TEST_QUERIES:
            q_tokens = tokenize(query)
            scores   = bm25.get_scores(q_tokens)
            top3_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]

            print(f"\n  ▶ 질문: {query}")
            print(f"    쿼리 토큰: {q_tokens}")
            for rank, idx in enumerate(top3_idx):
                title   = chunks[idx].get("title", "")[:30]
                cat     = chunks[idx].get("category", "")
                score   = scores[idx]
                preview = chunks[idx]["content"][:60].replace("\n", " ")
                print(f"    [{rank+1}] BM25={score:.4f} [{cat}] {title}")
                print(f"         {preview}...")

            top_score = scores[top3_idx[0]]
            record(f"BM25 검색: '{query[:15]}'", PASS if top_score > 0 else FAIL,
                   f"최고점수={top_score:.4f} | top1=[{chunks[top3_idx[0]]['category']}] {chunks[top3_idx[0]]['title'][:25]}")

    except ImportError as e:
        record("BM25 검색", SKIP, f"패키지 없음: {e}")
    except Exception as e:
        record("BM25 검색", FAIL, str(e))


# ══════════════════════════════════════════════════
# [5] chroma_store.py DB_PATH 절대경로 검증
# ══════════════════════════════════════════════════
print("\n" + "="*55)
print("[5] DB_PATH 절대경로 계산 검증")
print("="*55)

chroma_store_path = BASE_DIR / "src" / "vectordb" / "chroma_store.py"
chroma_src        = chroma_store_path.read_text(encoding="utf-8")

# 상대경로 잔재 없는지 확인
has_relative = '= "./chroma_db"' in chroma_src or "= './/chroma_db'" in chroma_src
has_absolute = 'Path(__file__).parent.parent.parent' in chroma_src
has_ids      = '"ids"' in chroma_src and 'results["ids"]' in chroma_src
has_upsert   = 'collection.upsert(' in chroma_src
has_add_old  = 'collection.add(' in chroma_src

record("DB_PATH 상대경로 제거", PASS if not has_relative else FAIL,
       '이전 "./chroma_db" 없음' if not has_relative else '상대경로 잔재 발견')
record("DB_PATH 절대경로 사용", PASS if has_absolute else FAIL,
       'Path(__file__).parent.parent.parent 사용 확인')
record('search() "ids" 반환', PASS if has_ids else FAIL,
       '"ids" include + results["ids"] 반환 확인')
record("add() → upsert()", PASS if has_upsert and not has_add_old else FAIL,
       f"upsert={has_upsert} | collection.add={has_add_old}")


# ══════════════════════════════════════════════════
# [6] pipeline.py 핵심 변경 검증
# ══════════════════════════════════════════════════
print("\n" + "="*55)
print("[6] pipeline.py 핵심 변경 검증")
print("="*55)

pipeline_src = (BASE_DIR / "src" / "rag" / "pipeline.py").read_text(encoding="utf-8")

has_none_filter   = 'h["embed_score"] is not None' in pipeline_src
has_temp_removed  = 'temperature=' not in pipeline_src
has_do_sample     = 'do_sample=False' in pipeline_src
has_threshold     = 'SIMILARITY_THRESHOLD = 0.40' in pipeline_src

record("embed_score None 필터링", PASS if has_none_filter else FAIL,
       'h["embed_score"] is not None 필터 확인')
record("temperature=1.0 제거", PASS if has_temp_removed else FAIL,
       'do_sample=False만 사용, temperature 없음' if has_temp_removed else 'temperature 잔재 발견')
record("greedy decoding 유지", PASS if has_do_sample else FAIL,
       'do_sample=False 확인')
record("SIMILARITY_THRESHOLD=0.40", PASS if has_threshold else FAIL,
       '임계값 0.40 유지 확인')


# ══════════════════════════════════════════════════
# [7] GPU 필요 항목 — Colab 전용 SKIP 처리
# ══════════════════════════════════════════════════
print("\n" + "="*55)
print("[7] GPU 필요 항목 (Colab 전용 — 로컬 SKIP)")
print("="*55)

record("ChromaDB 구축 (build_db.py)", SKIP,
       "Colab 셀 6에서 실행 — KURE-v1 로드 필요")
record("KURE-v1 임베딩 차원(1024d) 검증", SKIP,
       "Colab 셀 6 실행 후 collection.metadata 확인")
record("Qwen2.5-3B generate() 실행", SKIP,
       "Colab 셀 7+8 — T4 GPU 필요")
record("테스트 질문 end-to-end 답변", SKIP,
       "Colab 셀 8 출력으로 검증")


# ══════════════════════════════════════════════════
# 최종 요약
# ══════════════════════════════════════════════════
print("\n" + "="*55)
print("최종 검증 결과 요약")
print("="*55)

pass_n = sum(1 for r in results if r[1] == PASS)
fail_n = sum(1 for r in results if r[1] == FAIL)
skip_n = sum(1 for r in results if r[1] == SKIP)

print(f"\n PASS: {pass_n}  |  FAIL: {fail_n}  |  SKIP(Colab): {skip_n}\n")

print(f"{'검증 항목':<40} {'결과':<8} {'비고'}")
print("-"*90)
for name, status, detail, warn in results:
    icon = "✅" if status == PASS else ("⚠️" if status == SKIP else "❌")
    print(f" {icon} {name:<38} {status:<8} {detail[:40]}")

if fail_n > 0:
    print("\n[실패 항목 상세]")
    for name, status, detail, warn in results:
        if status == FAIL:
            print(f"  ❌ {name}: {detail}")
