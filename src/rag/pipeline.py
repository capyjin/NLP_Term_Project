"""
RAG 파이프라인 — 질문 → 검색 → LLM 생성
──────────────────────────────────────────
흐름:
  question
    → HybridRetriever.search()        # BM25 + 임베딩 + RRF
    → context 조합 + embed_score 확인
    → SIMILARITY_THRESHOLD 미달 시 조기 반환 (LLM 미호출)
    → Qwen2.5-3B 4-bit greedy decoding

할루시네이션 방지 (SIMILARITY_THRESHOLD = 0.40):
  - retrieve()가 embed_score 목록을 수집
  - embed_score=None(BM25 전용 결과)은 임계값 판단에서 제외
    → "BM25에는 있지만 임베딩에는 없다"는 이유로 답변 차단하지 않음
  - embed_score가 있는 결과 중 최고값이 0.40 미만이면 "찾을 수 없습니다" 반환

모델 설정:
  - Qwen/Qwen2.5-3B-Instruct: T4 4-bit NF4 ~2.5GB VRAM
  - do_sample=False: greedy decoding (재현성 보장, 속도↑)
  - finetune.py는 7B 기준이므로 파인튜닝 모델 사용 시 DEFAULT_MODEL 변경 필요
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from src.vectordb.chroma_store import CNUVectorStore
from src.rag.retriever import HybridRetriever

# Drive 저장 모델 우선 로드 → 재시작 시 재다운로드 없음
# 없으면 HuggingFace에서 자동 다운로드
import os as _os
_DRIVE_MODEL = "/content/drive/MyDrive/models/qwen2.5-3b-4bit"
DEFAULT_MODEL = _DRIVE_MODEL if _os.path.exists(_DRIVE_MODEL) else "Qwen/Qwen2.5-3B-Instruct"

# 임계값: 가장 유사한 청크의 embed_score가 이 값 미만이면 모른다고 답변
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
        # vectorstore 미전달 시 DB_PATH(BASE_DIR/chroma_db)에서 자동 로드
        self.vectorstore = vectorstore or CNUVectorStore()
        # HybridRetriever: BM25 인덱스 구축 + id→idx 딕셔너리 초기화
        self.retriever = HybridRetriever(self.vectorstore)
        self._load_model(model_name, use_4bit)

    def _load_model(self, model_name: str, use_4bit: bool):
        """
        Qwen2.5 LLM 로드.
        use_4bit=True: NF4 4-bit 양자화 (T4 VRAM 절약)
          - bnb_4bit_use_double_quant=True: 이중 양자화로 추가 메모리 절약
          - bnb_4bit_quant_type="nf4": NormalFloat4, 가중치 분포에 최적
        device_map="auto": GPU/CPU 자동 배치 (GPU 없으면 CPU fallback)
        """
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
        """
        HybridRetriever로 관련 청크 검색 후 context 문자열 조합.

        반환: (context_str, best_embed_score)
          context_str     : 상위 청크를 [출처: 제목]\n내용 형식으로 \n\n 구분 병합
          best_embed_score: embed_score가 있는 결과 중 최고값
            → embed_score=None (BM25 전용) 제외 — 임계값 판단 정확도 향상
            → embed_score 있는 결과가 전혀 없으면 0.0 반환 (임계값 미달 처리)
        """
        hits = self.retriever.search(question, top_k=top_k)
        if not hits:
            return "", 0.0

        # embed_score=None(BM25 전용) 제외하고 임계값 판단
        # 구버전(0.0 sentinel): BM25 전용 결과가 있을 때 best_score=0.0 → 불필요한 "모릅니다"
        embed_scores     = [h["embed_score"] for h in hits if h["embed_score"] is not None]
        best_embed_score = max(embed_scores) if embed_scores else 0.0

        # 각 청크 앞에 출처 표시, \n\n으로 구분 → LLM 프롬프트에서 섹션 구별
        context = "\n\n".join(
            f"[출처: {h['metadata'].get('title', '')}]\n{h['content']}" for h in hits
        )
        return context, best_embed_score

    def generate(self, question: str, max_new_tokens: int = 512) -> str:
        """
        RAG 전체 파이프라인 실행.
        1. retrieve()로 context + best_embed_score 획득
        2. 임계값 미달 → "찾을 수 없습니다" 조기 반환 (LLM 호출 없이 빠름)
        3. Qwen chat template 적용 → greedy decoding
           do_sample=False: 결정론적 생성 (재현성 보장)
           repetition_penalty=1.1: 반복 문장 억제
        """
        context, score = self.retrieve(question)
        if score < SIMILARITY_THRESHOLD:
            # 관련 문서 없음 — LLM에 넘기지 않고 즉시 반환 (할루시네이션 방지)
            return "해당 정보를 찾을 수 없습니다. 충남대학교 포털(plus.cnu.ac.kr)을 확인하거나 관련 부서에 문의하세요."

        user_content = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]
        # apply_chat_template: Qwen 특유의 <|im_start|>/<|im_end|> 형식으로 변환
        text   = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,         # greedy decoding: 재현성 + 속도 (temperature 불필요)
                repetition_penalty=1.1,  # 반복 억제 (greedy decoding에서도 유효)
            )
        # 입력 토큰 이후 생성 부분만 슬라이싱 후 디코딩
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
