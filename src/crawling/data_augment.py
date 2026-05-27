"""
LLM을 이용한 QA 쌍 자동 생성 (데이터 증강)
크롤링한 문서 → Claude/GPT API → Q&A 쌍
"""

import json
from pathlib import Path

RAW_DIR = Path("../../data/raw")
QA_DIR = Path("../../data/qa_pairs")

PROMPT_TEMPLATE = """다음 문서를 읽고, 학생이 궁금해할 만한 질문과 답변 쌍을 {n}개 만들어주세요.
답변은 문서 내용만을 근거로 하고, 각 쌍은 JSON 배열 형태로 반환하세요.

형식:
[
  {{"question": "질문 내용", "answer": "답변 내용"}},
  ...
]

문서:
{document}
"""


def generate_qa_pairs_from_doc(doc: dict, n: int = 5) -> list[dict]:
    """실제 사용 시 Anthropic API 또는 OpenAI API 호출로 교체"""
    # TODO: API 키 설정 후 실제 LLM 호출 구현
    raise NotImplementedError("API 키 설정 후 구현 필요")


def build_qa_dataset(input_path: str, output_path: str, n_per_doc: int = 3):
    with open(input_path, encoding="utf-8") as f:
        docs = json.load(f)

    qa_pairs = []
    for i, doc in enumerate(docs):
        print(f"[{i+1}/{len(docs)}] {doc.get('title', '')[:30]}")
        try:
            pairs = generate_qa_pairs_from_doc(doc, n=n_per_doc)
            for pair in pairs:
                pair["source"] = doc.get("category", "")
                pair["doc_title"] = doc.get("title", "")
            qa_pairs.extend(pairs)
        except NotImplementedError:
            pass

    QA_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    print(f"QA 쌍 생성 완료: {len(qa_pairs)}건 → {output_path}")


if __name__ == "__main__":
    build_qa_dataset(
        input_path=str(RAW_DIR / "notices.json"),
        output_path=str(QA_DIR / "qa_pairs.json"),
    )
