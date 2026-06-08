"""
SIMILARITY_THRESHOLD 실험 스크립트
────────────────────────────────────
목적: threshold 값(0.35 / 0.38 / 0.40 / 0.42)별 miss/fallback 수 비교
입력: data/fake_test_chat.json
출력: 콘솔 표 + outputs/threshold_eval.txt

실행:
  python scripts/test_threshold.py                   # 기본 (4개 threshold 실험)
  python scripts/test_threshold.py --thresholds 0.35 0.40 0.45

주의:
  - ChromaDB + KURE-v1 임베딩 필요 (Colab 또는 chroma_db 로컬 빌드 완료 후 실행)
  - Qwen LLM은 로드하지 않음 — 검색 점수만 비교
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

LABEL_NAMES = {0: "졸업요건", 1: "학교공지", 2: "학사일정", 3: "식단", 4: "셔틀버스"}

# label → 예상 라우팅
# 3(식단), 4(셔틀) → handler 처리 (threshold 미적용)
# 0,1,2 → RAG 경유 (threshold 적용)
RAG_LABELS = {0, 1, 2}


def load_retriever():
    """ChromaDB + BM25 HybridRetriever 로드 (LLM 없이)."""
    from src.vectordb.chroma_store import CNUVectorStore
    from src.rag.retriever import HybridRetriever
    store = CNUVectorStore()
    return HybridRetriever(store)


def run_experiment(retriever, items: list[dict], thresholds: list[float]) -> dict:
    """
    각 질문의 embed_score를 수집하고 threshold별 miss 수를 계산.

    반환:
      {
        threshold: {
          "miss": int,        # score < threshold 건수 (RAG 경유 질문만)
          "hit": int,
          "miss_rate": float,
          "avg_score": float, # RAG 경유 질문의 평균 embed_score
          "per_label": {...}
        },
        ...
        "scores": [(question, label, score), ...]
      }
    """
    print(f"  {len(items)}개 질문 retrieval 중...")
    scores = []

    for i, item in enumerate(items, 1):
        q     = item.get("user", "")
        label = item.get("label", -99)

        # 식단(3)/셔틀(4)은 handler로 처리 → threshold 미적용, score=None
        if label not in RAG_LABELS:
            scores.append((q, label, None))
            continue

        hits = retriever.search(q, top_k=3)
        embed_scores = [h["embed_score"] for h in hits if h.get("embed_score") is not None]
        best = max(embed_scores) if embed_scores else 0.0
        scores.append((q, label, best))

        if i % 10 == 0:
            print(f"    {i}/{len(items)} 완료")

    # threshold별 집계
    results = {}
    rag_scores = [(q, lbl, s) for q, lbl, s in scores if s is not None]

    for thr in thresholds:
        miss_items = [(q, lbl, s) for q, lbl, s in rag_scores if s < thr]
        hit_items  = [(q, lbl, s) for q, lbl, s in rag_scores if s >= thr]
        avg = sum(s for _, _, s in rag_scores) / len(rag_scores) if rag_scores else 0.0

        per_label = defaultdict(lambda: {"miss": 0, "total": 0})
        for q, lbl, s in rag_scores:
            per_label[lbl]["total"] += 1
            if s < thr:
                per_label[lbl]["miss"] += 1

        results[thr] = {
            "miss":      len(miss_items),
            "hit":       len(hit_items),
            "rag_total": len(rag_scores),
            "miss_rate": round(len(miss_items) / len(rag_scores), 3) if rag_scores else 0.0,
            "avg_score": round(avg, 4),
            "per_label": {
                LABEL_NAMES.get(lbl, str(lbl)): v
                for lbl, v in per_label.items()
            },
            "miss_examples": [(q, s) for q, _, s in miss_items[:5]],
        }

    results["_raw_scores"] = scores
    return results


def format_report(results: dict, thresholds: list[float]) -> str:
    lines = [
        "=" * 70,
        "SIMILARITY_THRESHOLD 실험 결과",
        "=" * 70,
        f"{'Threshold':>12} | {'Miss':>6} | {'Hit':>6} | {'Miss%':>7} | {'AvgScore':>9} | 추천",
        "-" * 70,
    ]

    # 추천 기준: miss_rate가 낮고, avg_score가 합리적이면 OK
    # 현재 fallback 메시지가 충분히 유용하므로 약간 낮은 threshold도 허용
    best_thr = None
    best_score = float("inf")

    for thr in thresholds:
        r = results[thr]
        # 점수 기준: miss_rate * 2 + (0.45 - avg_score).clip(0) → 낮을수록 좋음
        heuristic = r["miss_rate"] * 2 + max(0, 0.45 - r["avg_score"])
        if heuristic < best_score:
            best_score = heuristic
            best_thr = thr

    for thr in thresholds:
        r = results[thr]
        rec = "<-- 추천" if thr == best_thr else ""
        lines.append(
            f"  {thr:>10.2f} | {r['miss']:>6} | {r['hit']:>6} | "
            f"{r['miss_rate']*100:>6.1f}% | {r['avg_score']:>9.4f} | {rec}"
        )

    lines += ["", "라벨별 miss 수 (threshold=추천값):"]
    best = results[best_thr]
    for lname, v in sorted(best["per_label"].items()):
        lines.append(f"  {lname:8s}: {v['miss']}/{v['total']} miss")

    lines += ["", "miss 예시 (추천 threshold):"]
    for q, s in best["miss_examples"]:
        lines.append(f"  [{s:.4f}] {q}")

    lines += [
        "",
        f"최종 추천 threshold: {best_thr}",
        f"  miss {best['miss']}건 / RAG 질문 {best['rag_total']}건",
        f"  (fallback 메시지가 유용하므로 miss는 허용 범위)",
        "=" * 70,
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default="data/fake_test_chat.json")
    parser.add_argument("--output",     default="outputs/threshold_eval.txt")
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[0.35, 0.38, 0.40, 0.42])
    args = parser.parse_args()

    input_path = BASE_DIR / args.input
    if not input_path.exists():
        print(f"입력 파일 없음: {input_path}")
        sys.exit(1)

    items = json.loads(input_path.read_text(encoding="utf-8"))
    print(f"질문 {len(items)}개 로드")
    print(f"실험 threshold: {args.thresholds}")
    print()

    print("retriever 로드 중...")
    try:
        retriever = load_retriever()
        print("retriever 로드 완료")
    except Exception as e:
        print(f"retriever 로드 실패: {e}")
        print("chroma_db 미구축 상태 — Colab에서 build_db.py 실행 후 재시도하세요.")
        sys.exit(1)

    results = run_experiment(retriever, items, args.thresholds)
    report  = format_report(results, args.thresholds)

    out = BASE_DIR / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")

    print(report)
    print(f"\n저장 완료: {out}")


if __name__ == "__main__":
    main()
