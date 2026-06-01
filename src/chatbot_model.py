"""
챗봇 모델 CLI — test_chat.json → chat_output.json
─────────────────────────────────────────────────
실행:
  python src/chatbot_model.py
  python src/chatbot_model.py --input data/test_chat.json --output outputs/chat_output.json

chatbot.sh에서 자동 호출됨.

라우팅 흐름:
  식단 키워드   → MealHandler   (크롤링/수동 JSON → 공식 URL fallback)
  셔틀 키워드   → ShuttleHandler (크롤링/수동 JSON → known_data fallback)
  그 외         → RAGPipeline   (BM25 + KURE-v1 + RRF + Qwen2.5-3B)
"""

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))


def main():
    parser = argparse.ArgumentParser(description="CNU Campus ChatBot — batch inference")
    parser.add_argument("--input",  default="data/test_chat.json",     help="입력 JSON (user 질문 목록)")
    parser.add_argument("--output", default="outputs/chat_output.json", help="출력 JSON (model 응답 포함)")
    args = parser.parse_args()

    input_path  = BASE_DIR / args.input
    output_path = BASE_DIR / args.output

    if not input_path.exists():
        print(f"[오류] 입력 파일 없음: {input_path}")
        print("조교 제공 test_chat.json을 data/ 폴더에 넣어주세요.")
        sys.exit(1)

    with open(input_path, encoding="utf-8") as f:
        items = json.load(f)

    print(f"입력 로드: {len(items)}건 ({input_path})")
    print("ChatBot 초기화 중... (첫 실행 시 모델 다운로드 ~10분)")

    from src.rag.pipeline   import RAGPipeline
    from src.chatbot_router import CNUChatRouter

    pipeline = RAGPipeline()
    router   = CNUChatRouter(pipeline, BASE_DIR)

    print("추론 시작...\n")
    results = []
    for i, item in enumerate(items, 1):
        question = item.get("user", item.get("question", ""))
        if not question.strip():
            continue
        answer, source = router.chat(question)
        results.append({"user": question, "model": answer})
        print(f"[{i:3d}/{len(items)}] [{source}]")
        print(f"  Q: {question[:60]}")
        print(f"  A: {answer[:100]}\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"저장 완료: {output_path} ({len(results)}건)")


if __name__ == "__main__":
    main()
