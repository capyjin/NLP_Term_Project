"""
평가 스크립트
교수님이 제공하는 questions.json → answers.json 생성
형식: [{"question": "...", "answer": "..."}]
"""

import json
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.rag.pipeline import RAGPipeline


def run_evaluation(questions_path: str, output_path: str):
    with open(questions_path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list) and isinstance(data[0], str):
        questions = data
    elif isinstance(data, list) and isinstance(data[0], dict):
        questions = [d["question"] for d in data]
    else:
        raise ValueError("질문 파일 형식이 올바르지 않습니다.")

    print(f"질문 {len(questions)}건 로드. 모델 로딩 중...")
    pipeline = RAGPipeline()

    results = []
    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] {q[:40]}")
        answer = pipeline.generate(q)
        results.append({"question": q, "answer": answer})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n완료: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="questions.json")
    parser.add_argument("--output", default="answers.json")
    args = parser.parse_args()
    run_evaluation(args.questions, args.output)
