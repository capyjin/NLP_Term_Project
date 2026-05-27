"""
RAG 파이프라인
질문 → 벡터 검색 → 컨텍스트 조합 → LLM 생성
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from src.vectordb.chroma_store import CNUVectorStore

# Colab Free Tier 기준 최적 모델
# - VRAM 15GB 이내, 한국어 성능 우수
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # 4bit 시 ~5GB VRAM

SYSTEM_PROMPT = """당신은 충남대학교 재학생을 위한 Q&A 챗봇입니다.
주어진 참고 자료를 바탕으로 질문에 정확하고 친절하게 답변하세요.
참고 자료에 없는 내용은 모른다고 솔직하게 답하세요."""

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

    def retrieve(self, question: str, top_k: int = 3) -> str:
        hits = self.vectorstore.search(question, top_k=top_k)
        return "\n\n".join(
            f"[출처: {h['metadata'].get('title', '')}]\n{h['content']}" for h in hits
        )

    def generate(self, question: str, max_new_tokens: int = 512) -> str:
        context = self.retrieve(question)
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
