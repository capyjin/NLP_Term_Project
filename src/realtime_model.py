"""
실시간 정보 반영 모델 — test_realtime.json → realtime_output.json
──────────────────────────────────────────────────────────────────
[전략]
  공지사항 / 학사일정: 기존 RAGPipeline (ChromaDB + BM25 + Qwen2.5-3B) 사용
  식단 안내:  MealHandler (meal_crawler.py 결과 → 공식 URL fallback)
  셔틀버스:   ShuttleHandler (shuttle_bus.json → known_data fallback)

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

from src.chatbot_router import CNUChatRouter, detect_category


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

    # RAG 파이프라인 — 식단/셔틀 질문에는 로드하지 않음 (lazy 초기화)
    pipeline = None
    router   = None

    # 핸들러 사전 초기화 (식단/셔틀 전용 — Qwen 로드 없이)
    from src.handlers.meal_handler    import MealHandler
    from src.handlers.shuttle_handler import ShuttleHandler
    _meal_h    = MealHandler(BASE_DIR)
    _shuttle_h = ShuttleHandler(BASE_DIR)

    results = []
    for i, item in enumerate(items, 1):
        question = item.get("user", item.get("question", ""))
        cat = detect_category(question)
        # cat: 3=식단, 4=셔틀, 5=장학, 6=공지, -1=RAG
        cat_names = {3: "식단", 4: "셔틀버스", 5: "장학", 6: "공지", -1: "rag"}
        cat_label = cat_names.get(cat, "rag")
        print(f"[{i:3d}/{len(items)}] [{cat_label}] {question[:50]}")

        if cat == 3:
            answer, source = _meal_h.answer(question)
        elif cat == 4:
            answer, source = _shuttle_h.answer(question)
        else:
            # 장학(5), 공지(6), 학사일정/기타(-1): CNUChatRouter → RAG/Handler
            if pipeline is None:
                print("  RAGPipeline 초기화 중...")
                from src.rag.pipeline import RAGPipeline
                pipeline = RAGPipeline()
                router   = CNUChatRouter(pipeline, BASE_DIR)
            answer, source = router.chat(question)

        results.append({"user": question, "model": answer})
        print(f"  [{source}] A: {answer[:80]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료: {output_path} ({len(results)}건)")


if __name__ == "__main__":
    main()
