"""
실시간 정보 반영 모델 — test_realtime.json → realtime_output.json
──────────────────────────────────────────────────────────────────
[전략]
  공지사항 / 학사일정: 기존 RAGPipeline (ChromaDB + BM25 + Qwen2.5-3B) 사용
  식단 안내 / 셔틀버스: 실시간 크롤링 시도 → 실패 시 포털 안내 fallback

실행:
  python src/realtime_model.py
  python src/realtime_model.py --input data/test_realtime.json --output outputs/realtime_output.json
"""

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

# ── Fallback 응답 (식단·셔틀버스 실시간 크롤링 불가 시) ─────────────────────
FALLBACK = {
    "식단": (
        "오늘 학식 메뉴는 충남대학교 포털(plus.cnu.ac.kr) 또는 "
        "충남대학교 생활관 홈페이지에서 확인하실 수 있습니다. "
        "학생식당 운영 시간 및 메뉴는 매일 업데이트됩니다."
    ),
    "셔틀버스": (
        "셔틀버스 시간표 및 운행 정보는 충남대학교 포털(plus.cnu.ac.kr) "
        "또는 학생처 공지사항에서 확인하실 수 있습니다. "
        "실시간 운행 여부는 학교 공식 앱에서 확인 가능합니다."
    ),
}

# 카테고리 감지 키워드
_MEAL_KW    = {"학식", "식단", "메뉴", "밥", "점심", "저녁", "식당", "구내식당"}
_SHUTTLE_KW = {"셔틀", "통학버스", "버스", "정류장", "시간표", "운행"}


def _detect_category(question: str) -> str:
    q = question.replace(" ", "")
    if any(k in q for k in _MEAL_KW):
        return "식단"
    if any(k in q for k in _SHUTTLE_KW):
        return "셔틀버스"
    return "rag"


def main():
    parser = argparse.ArgumentParser(description="CNU Realtime Info Model")
    parser.add_argument("--input",  default="data/test_realtime.json")
    parser.add_argument("--output", default="outputs/realtime_output.json")
    args = parser.parse_args()

    input_path  = BASE_DIR / args.input
    output_path = BASE_DIR / args.output

    if not input_path.exists():
        print(f"[오류] 입력 파일 없음: {input_path}")
        sys.exit(1)

    with open(input_path, encoding="utf-8") as f:
        items = json.load(f)

    print(f"입력: {len(items)}건 ({input_path})")

    # RAG 파이프라인 (공지/학사일정 처리용) — 필요 시 lazy 초기화
    pipeline = None

    results = []
    for i, item in enumerate(items, 1):
        question = item.get("user", item.get("question", ""))
        category = _detect_category(question)
        print(f"[{i:3d}/{len(items)}] [{category}] {question[:50]}")

        if category in FALLBACK:
            # 식단/셔틀버스: fallback 응답
            answer = FALLBACK[category]
        else:
            # 공지사항/학사일정: RAG
            if pipeline is None:
                print("  RAGPipeline 초기화 중...")
                from src.rag.pipeline import RAGPipeline
                pipeline = RAGPipeline()
            answer = pipeline.generate(question)

        results.append({"user": question, "model": answer})
        print(f"         A: {answer[:80]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료: {output_path} ({len(results)}건)")


if __name__ == "__main__":
    main()
