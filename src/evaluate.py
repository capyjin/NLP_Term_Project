"""
평가 스크립트 — questions.json → answers.json 일괄 생성
──────────────────────────────────────────────────────
사용:
  python src/evaluate.py --questions questions.json --output answers.json

입력 형식 (questions.json):
  ["질문1", "질문2", ...]              # 문자열 리스트
  [{"question": "질문1"}, ...]         # 딕셔너리 리스트

출력 형식 (answers.json):
  [{"question": "질문1", "answer": "답변1"}, ...]

⚠️ GPU 필요: RAGPipeline 기본값 use_4bit=True → GPU 없으면 bitsandbytes 오류
   CPU 실행 시: RAGPipeline(use_4bit=False) 으로 변경 (속도 매우 느림)
"""

import json
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.rag.pipeline import RAGPipeline


def run_evaluation(questions_path: str, output_path: str):
    """질문 파일을 읽어 RAGPipeline으로 답변 생성 후 저장."""
    with open(questions_path, encoding="utf-8") as f:
        data = json.load(f)

    # 입력 형식 자동 감지
    if isinstance(data, list) and isinstance(data[0], str):
        questions = data
    elif isinstance(data, list) and isinstance(data[0], dict):
        questions = [d["question"] for d in data]
    else:
        raise ValueError("질문 파일 형식이 올바르지 않습니다. (문자열 리스트 또는 {question} 딕셔너리 리스트)")

    print(f"질문 {len(questions)}건 로드. 모델 로딩 중...")
    # RAGPipeline: CNUVectorStore + HybridRetriever + Qwen2.5-3B 4-bit 로드
    pipeline = RAGPipeline()

    results = []
    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] {q[:40]}")
        answer = pipeline.generate(q)
        results.append({"question": q, "answer": answer})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n완료: {output_path} ({len(results)}건)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Q&A 일괄 평가")
    parser.add_argument("--questions", default="questions.json", help="질문 파일 경로")
    parser.add_argument("--output",    default="answers.json",   help="답변 저장 경로")
    args = parser.parse_args()
    run_evaluation(args.questions, args.output)
