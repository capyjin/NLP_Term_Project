# ============================================================
# CNU Q&A 챗봇 - Colab 실행 파일
# 각 셀(# %%)을 Colab에 순서대로 붙여넣으세요
# ============================================================

# %% [1] 패키지 설치 (처음 1회만)
!pip install -q \
  transformers==4.46.0 \
  peft==0.13.0 \
  bitsandbytes==0.44.1 \
  accelerate==1.1.1 \
  sentence-transformers==3.2.1 \
  chromadb==0.5.18 \
  gradio==5.5.0

# %% [2] Google Drive 마운트 & 프로젝트 설정
from google.colab import drive
drive.mount('/content/drive')

import os
PROJECT = '/content/drive/MyDrive/cnu-qa-chatbot'
os.makedirs(PROJECT, exist_ok=True)
os.chdir(PROJECT)
print("작업 디렉토리:", os.getcwd())

# %% [3] 저장소 클론 (처음 1회만)
# GitHub에 push한 뒤 아래 URL을 실제 주소로 교체하세요
!git clone https://github.com/capyjin/NLP_Term_Project.git . 2>/dev/null || (git pull && echo "업데이트 완료")

# %% [4] 크롤링 데이터 확인 (Drive에 업로드한 경우 스킵)
import json, os
chunks_path = "data/processed/chunks.json"
if os.path.exists(chunks_path):
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"청크 로드: {len(chunks)}건")
else:
    print("chunks.json 없음 - Drive에 data/processed/chunks.json 업로드 필요")

# %% [5] ChromaDB 구축 (처음 1회만 - 이미 chroma_db/ 있으면 스킵)
import chromadb
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "jhgan/ko-sroberta-multitask"
DB_PATH = "./chroma_db"
COLLECTION = "cnu_docs"

if not os.path.exists(DB_PATH) or len(os.listdir(DB_PATH)) == 0:
    print("벡터 DB 구축 중...")
    embedder = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=DB_PATH)
    col = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    batch = 64
    for start in range(0, len(chunks), batch):
        batch_chunks = chunks[start:start+batch]
        texts = [c["content"] for c in batch_chunks]
        ids   = [c["id"] for c in batch_chunks]
        metas = [{"title": c.get("title",""), "category": c.get("category",""), "url": c.get("url","")} for c in batch_chunks]
        embs  = embedder.encode(texts, normalize_embeddings=True).tolist()
        col.add(ids=ids, documents=texts, metadatas=metas, embeddings=embs)
        print(f"  {min(start+batch, len(chunks))}/{len(chunks)} 색인 완료")
    print("벡터 DB 구축 완료!")
else:
    print("기존 벡터 DB 사용")

# %% [6] 임베딩 모델 & DB 로드
from sentence_transformers import SentenceTransformer
import chromadb

embedder = SentenceTransformer("jhgan/ko-sroberta-multitask")
col = chromadb.PersistentClient("./chroma_db").get_collection("cnu_docs")
print(f"DB 문서 수: {col.count()}")

# %% [7] LLM 로드 (Qwen2.5-3B, 4-bit 양자화)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=quant, device_map="auto", trust_remote_code=True
)
model.eval()
print("모델 로드 완료:", MODEL_ID)

# %% [8] RAG 파이프라인 함수 정의
SYSTEM_PROMPT = (
    "당신은 충남대학교 재학생을 위한 Q&A 챗봇입니다. "
    "주어진 참고 자료를 바탕으로 질문에 정확하고 친절하게 한국어로 답변하세요. "
    "참고 자료에 없는 내용은 '해당 정보를 찾을 수 없습니다. "
    "충남대학교 포털(plus.cnu.ac.kr)을 확인하거나 관련 부서에 문의하세요.'라고 답하세요."
)
THRESHOLD = 0.40   # 유사도 임계값: 미달 시 모른다고 답변
TOP_K = 3


def retrieve(question: str):
    q_emb = embedder.encode([question], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=q_emb, n_results=TOP_K,
                    include=["documents", "metadatas", "distances"])
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    scores = [1 - d for d in res["distances"][0]]
    return docs, metas, scores


def rag_answer(question: str, max_new_tokens: int = 400) -> str:
    docs, metas, scores = retrieve(question)

    if not docs or scores[0] < THRESHOLD:
        return ("해당 정보를 찾을 수 없습니다. "
                "충남대학교 포털(plus.cnu.ac.kr)을 확인하거나 관련 부서에 문의하세요.")

    context = "\n\n".join(
        f"[{m.get('title','')}]\n{d}" for d, m in zip(docs, metas)
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"[참고 자료]\n{context}\n\n[질문]\n{question}"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# 빠른 테스트
print(rag_answer("장학금 신청은 어떻게 하나요?"))

# %% [9] Gradio UI 실행
import gradio as gr

def chat(message, history):
    return rag_answer(message)

demo = gr.ChatInterface(
    fn=chat,
    title="충남대학교 Q&A 챗봇",
    description="학사, 장학금, 취업 등 충남대 관련 질문을 입력하세요.",
    examples=[
        "장학금 신청은 어떻게 하나요?",
        "학생생활관 입주 신청 방법이 뭔가요?",
        "취업 지원 프로그램이 있나요?",
    ],
    type="messages",
)
demo.launch(share=True)

# %% [10] 평가용 일괄 답변 생성 (교수님 제출용)
import json

with open("questions.json", encoding="utf-8") as f:
    data = json.load(f)

qs = [q["question"] if isinstance(q, dict) else q for q in data]

results = []
for i, q in enumerate(qs):
    print(f"[{i+1}/{len(qs)}] {q[:40]}")
    ans = rag_answer(q)
    results.append({"question": q, "answer": ans})

with open("answers.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"answers.json 생성 완료 ({len(results)}건)")
