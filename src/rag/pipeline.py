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
  - embed_score가 있는 결과 중 최고값이 0.40 미만이면 카테고리별 안내 반환

모델 설정 (Colab T4 기준):
  - Qwen/Qwen2.5-3B-Instruct: T4 4-bit NF4 ~2.5GB VRAM  ← 현재 기본값 (안정성 우선)
  - Qwen/Qwen2.5-7B-Instruct: T4 4-bit NF4 ~5.0GB VRAM  ← EXPERIMENT_MODEL (실험용)
  - do_sample=False: greedy decoding (재현성 보장, 속도↑)
  - 3B: ~40tok/s / 7B: ~20tok/s
  - Handler 커버율 70%+ 이므로 Qwen 의존도 낮음 → 3B 품질로 충분

모델 선택:
  - DEFAULT_MODEL: Drive 3B 캐시 → 3B HuggingFace 순서로 로드
  - 7B 실험 시: pipeline = RAGPipeline(model_name=EXPERIMENT_MODEL)
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from src.vectordb.chroma_store import CNUVectorStore
from src.rag.retriever import HybridRetriever

# Drive 저장 모델 우선 로드 → 재시작 시 재다운로드 없음
# 없으면 HuggingFace에서 자동 다운로드
import os as _os

# ── 모델 경로 설정 ────────────────────────────────────────────────────────────
# 우선순위: Drive 3B 캐시 → HuggingFace 3B (안정성 우선)
_DRIVE_7B = "/content/drive/MyDrive/models/qwen2.5-7b-4bit"   # 7B Drive 캐시 (실험용)
_DRIVE_3B = "/content/drive/MyDrive/models/qwen2.5-3b-4bit"   # 3B Drive 캐시 (기본)
_HF_7B    = "Qwen/Qwen2.5-7B-Instruct"                        # HuggingFace 7B (실험용)
_HF_3B    = "Qwen/Qwen2.5-3B-Instruct"                        # HuggingFace 3B (기본)

# 기본 모델: 3B (T4 4-bit NF4 ~2.5GB VRAM, 안정성 우선)
# Colab T4: KURE-v1(~2GB) + 3B(~2.5GB) = ~4.5GB → 여유 ~10GB 확보
DEFAULT_MODEL  = _DRIVE_3B if _os.path.exists(_DRIVE_3B) else _HF_3B   # 안정 운영

# 실험용 7B: 전용 환경에서 VRAM 여유 확보 후 사용
# pipeline = RAGPipeline(model_name=EXPERIMENT_MODEL)
EXPERIMENT_MODEL = _DRIVE_7B if _os.path.exists(_DRIVE_7B) else _HF_7B  # 실험용 보존

# 하위 호환성
FALLBACK_MODEL = DEFAULT_MODEL

# 임계값: 가장 유사한 청크의 embed_score가 이 값 미만이면 모른다고 답변
SIMILARITY_THRESHOLD = 0.40

SYSTEM_PROMPT = """당신은 충남대학교 재학생을 위한 Q&A 챗봇입니다.
주어진 참고 자료를 바탕으로 질문에 정확하고 친절하게 답변하세요.
참고 자료에 정확한 날짜·금액·학점 수치가 없다면 단정하지 말고, 일반적인 학교 행정 흐름과 확인 경로를 안내해 주세요.
답변은 반드시 학생이 다음 행동을 취할 수 있도록 구체적인 확인 방법을 포함하세요."""

# ── threshold miss 시 카테고리별 안내 (키워드 매칭) ────────────────────────────
# 목적: "찾을 수 없습니다"로만 끝내지 않고 일반 행정 흐름 + 확인 경로 제공
# 우선순위 순서대로 배치 (앞쪽이 먼저 매칭)
_THRESHOLD_FALLBACKS = [
    (
        ("졸업학점", "전공학점", "교양학점", "이수학점", "졸업요건", "졸업조건",
         "졸업인증", "조기졸업", "학생편람", "교육과정"),
        "졸업 요건은 학과·학번·입학년도에 따라 다릅니다. 일반적으로 총 이수학점(교양+전공+일반선택) 충족, "
        "전공필수 이수, 교양필수 이수, 졸업인증(영어·봉사·SW 등) 요건을 갖춰야 합니다.\n"
        "정확한 기준 확인: 충남대 포털(plus.cnu.ac.kr) → 학사서비스 → 졸업요건 조회, 또는 소속 학과 사무실"
    ),
    (
        ("수강신청", "수강정정", "수강변경", "정정기간", "수강취소", "수강포기", "강의변경"),
        "수강신청·정정 일정은 학기마다 달라 정확한 날짜를 현재 자료로 안내하기 어렵습니다. "
        "일반적으로 수강신청은 개강 전, 수강정정은 개강 직후 1~2주 내 진행됩니다.\n"
        "확인 경로: 충남대 포털(plus.cnu.ac.kr) → 학사서비스 → 학사일정 또는 수강신청 메뉴"
    ),
    (
        ("계절학기", "하기계절", "동기계절"),
        "계절학기(하기·동기)는 정규 학기 종료 후 별도 신청 기간에 포털에서 신청합니다. "
        "신청 후 수강료를 납부해야 수강이 확정되며, 이수 학점은 졸업 학점에 포함됩니다.\n"
        "확인 경로: 충남대 포털(plus.cnu.ac.kr) → 학사공지 또는 수강신청 메뉴"
    ),
    (
        ("등록금", "수업료", "납부기간", "납부방법"),
        "등록금 납부 기간은 매 학기 초 포털 공지를 통해 안내됩니다. "
        "포털 로그인 후 학사서비스 → 등록금 납부 메뉴에서 가상계좌를 확인하고 기한 내 납부하세요.\n"
        "확인 경로: 충남대 포털(plus.cnu.ac.kr) → 학사서비스 → 등록금 납부"
    ),
    (
        ("국가장학금", "교내장학금", "장학금", "장학", "근로장학", "긴급장학", "성적장학"),
        "장학금은 종류에 따라 신청 경로와 자격이 다릅니다. "
        "국가장학금은 한국장학재단(www.kosaf.go.kr), 교내장학금은 포털 → 학생서비스 → 장학 메뉴에서 신청합니다. "
        "신청 기간은 학기 초 공지되므로 포털 장학공지를 꼭 확인하세요.\n"
        "확인 경로: 충남대 포털(plus.cnu.ac.kr) → 장학공지, 또는 한국장학재단(www.kosaf.go.kr)"
    ),
    (
        ("중간고사", "기말고사", "시험기간", "성적공시", "성적발표", "이의신청"),
        "시험 기간과 성적 공시 일정은 매 학기 학사일정에 따라 다릅니다. "
        "성적 이의신청은 공시 기간 내에만 가능하므로 일정을 꼭 확인하세요.\n"
        "확인 경로: 충남대 포털(plus.cnu.ac.kr) → 학사서비스 → 학사일정"
    ),
    (
        ("개강", "종강", "방학", "학사일정"),
        "개강·종강·방학 일정은 매 학년도 초에 학사일정으로 공지됩니다.\n"
        "확인 경로: 충남대 포털(plus.cnu.ac.kr) → 학사서비스 → 학사일정"
    ),
    (
        ("휴학", "복학", "군휴학"),
        "휴학·복학 신청은 포털에서 진행하며 신청 기간은 학기 초 공지됩니다. "
        "군 휴학 등 특수 사유는 기간 외 신청도 가능한 경우가 있습니다.\n"
        "확인 경로: 충남대 포털(plus.cnu.ac.kr) → 학사서비스, 또는 학과 사무실·학생처"
    ),
    (
        ("취업", "채용", "공채", "인턴", "취직"),
        "취업 관련 공지·채용설명회 일정은 포털 취업공지와 인재개발원 홈페이지에서 확인할 수 있습니다. "
        "취업 상담 및 지원 프로그램도 인재개발원을 통해 이용하실 수 있습니다.\n"
        "확인 경로: 충남대 포털(plus.cnu.ac.kr) → 취업공지, 또는 인재개발원 홈페이지"
    ),
    (
        ("공지", "공지사항", "학교소식", "최근공지"),
        "학교 공지사항은 충남대 포털(plus.cnu.ac.kr) → 공지사항 메뉴에서 "
        "학사공지·장학공지·일반공지·취업공지를 확인할 수 있습니다.\n"
        "확인 경로: 충남대 포털(plus.cnu.ac.kr) → 공지사항"
    ),
    (
        ("증명서", "재학증명", "성적증명", "졸업증명"),
        "각종 증명서(재학·성적·졸업증명서 등)는 포털 → 학사서비스 → 증명서 발급 메뉴에서 "
        "온라인 발급 또는 무인민원발급기를 이용할 수 있습니다.\n"
        "확인 경로: 충남대 포털(plus.cnu.ac.kr) → 학사서비스 → 증명서 발급"
    ),
]


def _get_threshold_fallback(question: str) -> str:
    """
    embed_score 임계값 미달 시 질문 키워드로 카테고리 추정 → 일반 안내 반환.
    "찾을 수 없습니다"로만 끝내지 않고 행정 흐름 + 확인 경로를 포함한 유용한 답변 생성.
    """
    for keywords, fallback in _THRESHOLD_FALLBACKS:
        if any(k in question for k in keywords):
            return fallback
    return (
        "관련 정보를 저장된 자료에서 찾지 못했습니다. "
        "충남대학교 포털(plus.cnu.ac.kr)에서 학사서비스·공지사항을 확인하거나 "
        "소속 학과 또는 관련 부서에 문의하세요."
    )

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

        VRAM 예상 (4-bit NF4):
          3B → ~2.5GB  (기본값, T4 15GB 기준 여유 ~10GB)
          7B → ~5.0GB  (실험용, KURE-v1 2GB와 합산 시 T4에서 경합 가능)
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
            device_map={"": 0},      # "auto" 대신 GPU 0 강제 배치 (CPU 오프로드 방지)
            trust_remote_code=True,
        )
        self.model.eval()

    def retrieve(self, question: str, top_k: int = 2) -> tuple[str, float]:
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

    def generate(self, question: str, max_new_tokens: int = 300) -> str:
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
            # 관련 문서 없음 — 카테고리별 일반 안내 반환 (단순 "모름"이 아닌 유용한 안내)
            return _get_threshold_fallback(question)

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
