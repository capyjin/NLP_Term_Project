"""
챗봇 품질 평가 스크립트
────────────────────────
입력: data/fake_test_chat.json  (user + label 포함)
출력: outputs/eval_result.json  (질문별 평가 결과)
      outputs/eval_summary.txt  (요약 리포트)

실행:
  python scripts/evaluate_chatbot.py
  python scripts/evaluate_chatbot.py --input data/fake_test_chat.json --model_eval false

평가 항목 (자동):
  1. 분류 정확성  — detect_category() 결과 vs label
  2. fallback 여부 — "찾을 수 없습니다" / "공식 페이지" 포함 여부
  3. 응답 길이    — 너무 짧으면 품질 의심

평가 항목 (Qwen 모델 실행 시):
  4. hallucination 의심 — 구체적 수치(학점/날짜/금액) 단언 포함 여부
  5. 응답 자연스러움 — 한국어 비율

Colab에서 Qwen 포함 전체 평가:
  python scripts/evaluate_chatbot.py --model_eval true
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.chatbot_router import detect_category

LABEL_NAMES = {0: "졸업요건", 1: "학교공지", 2: "학사일정", 3: "식단", 4: "셔틀버스"}
ROUTE_NAMES = {3: "meal_handler", 4: "shuttle_handler", -1: "rag_pipeline"}

# hallucination 의심 패턴: 구체적 수치를 단언하는 표현
_HALLUCINATION_PATTERNS = [
    r"\d+학점(?:이|을|은|가) 필요",   # "130학점이 필요"
    r"\d+월 \d+일(?:까지|에|부터)",    # "6월 15일까지"
    r"\d+,\d{3}원",                    # "1,000원"
    r"소득\s*\d+분위",                 # "소득 3분위"
    r"\d+시\s*\d+분",                  # "09시 30분"
]

_FALLBACK_MARKERS = [
    "찾을 수 없습니다",
    "포털(plus.cnu.ac.kr)을 확인",
    "공식 페이지에서 확인",
    "관련 부서에 문의",
    "해당 정보를 찾을 수 없",
]


def _check_hallucination(answer: str) -> list[str]:
    """구체적 수치 단언 패턴 감지 — 실제 데이터 없이 답변하면 hallucination 위험."""
    found = []
    for pat in _HALLUCINATION_PATTERNS:
        if re.search(pat, answer):
            found.append(pat)
    return found


def _is_fallback(answer: str) -> bool:
    return any(m in answer for m in _FALLBACK_MARKERS)


def _korean_ratio(text: str) -> float:
    if not text:
        return 0.0
    korean = sum(1 for c in text if "가" <= c <= "힣")
    return korean / len(text)


def evaluate_routing(items: list[dict]) -> dict:
    """라우팅 정확도만 평가 (모델 불필요)."""
    results = []
    label_correct = defaultdict(int)
    label_total   = defaultdict(int)

    for item in items:
        q   = item.get("user", "")
        exp = item.get("label", -99)
        got = detect_category(q)

        # label → expected cat 매핑
        # label 0,1,2 → RAG(-1), label 3→meal(3), label 4→shuttle(4)
        expected_cat = exp if exp in (3, 4) else -1
        route_correct = (got == expected_cat)

        label_total[exp] += 1
        if route_correct:
            label_correct[exp] += 1

        results.append({
            "question":      q,
            "label":         exp,
            "label_name":    LABEL_NAMES.get(exp, "?"),
            "expected_route": ROUTE_NAMES.get(expected_cat, "rag"),
            "actual_route":   ROUTE_NAMES.get(got, "rag"),
            "route_correct":  route_correct,
        })

    routing_acc = sum(r["route_correct"] for r in results) / len(results)

    per_label = {}
    for lbl in sorted(label_total):
        acc = label_correct[lbl] / label_total[lbl] if label_total[lbl] else 0
        per_label[LABEL_NAMES.get(lbl, str(lbl))] = {
            "correct": label_correct[lbl],
            "total":   label_total[lbl],
            "accuracy": round(acc, 3),
        }

    return {"routing_accuracy": round(routing_acc, 3), "per_label": per_label, "details": results}


def evaluate_with_model(items: list[dict]) -> list[dict]:
    """Qwen 포함 전체 평가 (GPU 필요)."""
    print("모델 로드 중...")
    from src.rag.pipeline   import RAGPipeline
    from src.chatbot_router import CNUChatRouter

    pipeline = RAGPipeline()
    router   = CNUChatRouter(pipeline, BASE_DIR)

    results = []
    for i, item in enumerate(items, 1):
        q   = item.get("user", "")
        exp = item.get("label", -99)
        print(f"[{i:3d}/{len(items)}] {q[:50]}")

        answer, source = router.chat(q)
        is_fb    = _is_fallback(answer)
        hall     = _check_hallucination(answer)
        kr_ratio = _korean_ratio(answer)

        results.append({
            "question":          q,
            "label":             exp,
            "label_name":        LABEL_NAMES.get(exp, "?"),
            "source":            source,
            "answer":            answer,
            "answer_len":        len(answer),
            "is_fallback":       is_fb,
            "hallucination_risk": len(hall) > 0,
            "hallucination_patterns": hall,
            "korean_ratio":      round(kr_ratio, 3),
            "quality_score":     _quality_score(answer, is_fb, hall, kr_ratio),
        })

    return results


def _quality_score(answer: str, is_fb: bool, hall: list, kr_ratio: float) -> int:
    """0~5 품질 점수 (빠른 휴리스틱)."""
    score = 5
    if is_fb:                  score -= 2   # fallback
    if hall:                   score -= 2   # hallucination 위험
    if len(answer) < 30:       score -= 1   # 너무 짧음
    if kr_ratio < 0.3:         score -= 1   # 한국어 너무 적음
    return max(0, score)


def print_summary(routing: dict, model_results: list = None) -> str:
    lines = ["=" * 60, "챗봇 품질 평가 결과", "=" * 60, ""]

    lines.append(f"[라우팅 정확도] {routing['routing_accuracy']*100:.1f}%")
    lines.append("")
    lines.append("카테고리별 라우팅 정확도:")
    for name, stat in routing["per_label"].items():
        bar = "█" * stat["correct"] + "░" * (stat["total"] - stat["correct"])
        lines.append(f"  {name:8s}: {stat['correct']}/{stat['total']}  [{bar}]  {stat['accuracy']*100:.0f}%")

    # 라우팅 실패 목록
    fails = [r for r in routing["details"] if not r["route_correct"]]
    if fails:
        lines.append(f"\n라우팅 실패 {len(fails)}건:")
        for r in fails:
            lines.append(f"  [{r['label_name']}] \"{r['question']}\"")
            lines.append(f"    예상={r['expected_route']}  실제={r['actual_route']}")

    if model_results:
        lines.append("\n" + "=" * 60)
        lines.append("[모델 응답 평가]")
        fallbacks = [r for r in model_results if r["is_fallback"]]
        halls     = [r for r in model_results if r["hallucination_risk"]]
        avg_len   = sum(r["answer_len"] for r in model_results) / len(model_results)
        avg_score = sum(r["quality_score"] for r in model_results) / len(model_results)

        lines.append(f"  총 질문: {len(model_results)}개")
        lines.append(f"  fallback 응답: {len(fallbacks)}개 ({len(fallbacks)/len(model_results)*100:.1f}%)")
        lines.append(f"  hallucination 위험: {len(halls)}개")
        lines.append(f"  평균 응답 길이: {avg_len:.0f}자")
        lines.append(f"  평균 품질 점수: {avg_score:.2f}/5")

        if fallbacks:
            lines.append("\nfallback 발생 질문:")
            for r in fallbacks[:10]:
                lines.append(f"  [{r['label_name']}] \"{r['question']}\"")

        if halls:
            lines.append("\nhallucination 위험 질문:")
            for r in halls:
                lines.append(f"  [{r['label_name']}] \"{r['question']}\"")
                lines.append(f"    패턴: {r['hallucination_patterns']}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       default="data/fake_test_chat.json")
    parser.add_argument("--output",      default="outputs/eval_result.json")
    parser.add_argument("--summary",     default="outputs/eval_summary.txt")
    parser.add_argument("--model_eval",  default="false",
                        help="true이면 Qwen 포함 전체 평가 (GPU 필요)")
    args = parser.parse_args()

    input_path = BASE_DIR / args.input
    if not input_path.exists():
        print(f"입력 파일 없음: {input_path}")
        sys.exit(1)

    with open(input_path, encoding="utf-8") as f:
        items = json.load(f)

    print(f"질문 {len(items)}개 로드")

    # 1. 라우팅 평가 (항상 실행)
    routing = evaluate_routing(items)
    print(f"라우팅 정확도: {routing['routing_accuracy']*100:.1f}%")

    # 2. 모델 평가 (선택)
    model_results = None
    if args.model_eval.lower() == "true":
        model_results = evaluate_with_model(items)

    # 저장
    output_path = BASE_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {"routing": routing}
    if model_results:
        result["model"] = model_results

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    summary = print_summary(routing, model_results)
    summary_path = BASE_DIR / args.summary
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"\n저장 완료:\n  {output_path}\n  {summary_path}")
    print("\n" + summary)


if __name__ == "__main__":
    main()
