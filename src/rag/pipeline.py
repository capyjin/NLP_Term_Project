"""
RAG 파이프라인
질문 → 벡터 검색 → 컨텍스트 조합 → LLM 생성
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from src.vectordb.chroma_store import CNUVectorStore

# 속도 기준: Qwen2.5-3B (4bit ~2.5GB VRAM, 7B 대비 2~3배 빠름)
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# 할루시네이션 방지: 유사도 임계값 미달 시 모른다고 답변
SIMILARITY_THRESHOLD = 0.40

SYSTEM_PROMPT = """당신은 충남대학교 재학생을 위한 Q&A 챗봇입니다.
주어진 참고 자료를 바탕으로 질문에 정확하고 친절하게 답변하세요.
참고 자료에 없는 내용은 "해당 정보를 찾을 수 없습니다. 충남대학교 포털(plus.cnu.ac.kr)을 확인하거나 관련 부서에 문의하세요."라고 답하세요."""

RAG_PROMPT_TEMPLATE = """[참고 자료]
{context}

[질문]
{question}

[답변]"""


class RAGPipeline:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        vectorstore: CNUVectorStore = None,
        use_4bit: bool = True,
    ):
        self.vectorstore = vectorstore or CNUVectorStore()
        self._load_model(model_name, use_4bit)

    def _load_model(self, model_name: str, use_4bit: bool):
        quant_config = None
        if use_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def retrieve(self, question: str, top_k: int = 3) -> tuple[str, float]:
        """(context_str, best_score) 반환. 점수가 낮으면 관련 문서 없음."""
        hits = self.vectorstore.search(question, top_k=top_k)
        if not hits:
            return "", 0.0
        best_score = hits[0]["score"]
        context = "\n\n".join(
            f"[출처: {h['metadata'].get('title', '')}]\n{h['content']}" for h in hits
        )
        return context, best_score

    def generate(self, question: str, max_new_tokens: int = 512) -> str:
        context, score = self.retrieve(question)
        if score < SIMILARITY_THRESHOLD:
            return "해당 정보를 찾을 수 없습니다. 충남대학교 포털(plus.cnu.ac.kr)을 확인하거나 관련 부서에 문의하세요."
        user_content = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                repetition_penalty=1.1,
            )
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
