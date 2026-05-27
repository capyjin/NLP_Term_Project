# ============================================================
# CNU Q&A 챗봇 - Colab 실행 파일
# 이 파일의 각 셀(# %% 구분)을 Colab에 순서대로 붙여넣기 하세요
# ============================================================

# %% [1] 환경 설치 (처음 1회만)
"""
!pip install -q \
  transformers==4.40.0 \
  peft==0.10.0 \
  bitsandbytes==0.43.0 \
  accelerate==0.28.0 \
  trl==0.8.0 \
  sentence-transformers==2.6.0 \
  chromadb==0.4.24 \
  gradio==4.20.0 \
  fastapi==0.110.0 \
  uvicorn==0.27.0 \
  beautifulsoup4 \
  requests \
  datasets
"""

# %% [2] Google Drive 마운트 & 프로젝트 루트 설정
"""
from google.colab import drive
drive.mount('/content/drive')

import os
PROJECT_ROOT = '/content/drive/MyDrive/cnu-qa-chatbot'
os.makedirs(PROJECT_ROOT, exist_ok=True)
os.chdir(PROJECT_ROOT)
!git clone https://github.com/YOUR_REPO_URL . 2>/dev/null || echo "이미 존재"
"""

# %% [3] 데이터 크롤링
"""
import requests
from bs4 import BeautifulSoup
import json, time

BASE_URL = "https://www.cnu.ac.kr"
HEADERS = {"User-Agent": "Mozilla/5.0"}

TARGETS = {
    "academic_notice": "/main/kr/sub05_01_01.do",
    "scholarship":     "/main/kr/sub05_01_02.do",
    "general_notice":  "/main/kr/sub05_01_03.do",
}

all_docs = []
for category, path in TARGETS.items():
    for page in range(1, 6):
        url = f"{BASE_URL}{path}?pageIndex={page}"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(resp.text, "lxml")
        rows = soup.select("table tbody tr")
        for row in rows:
            a = row.select_one("td a")
            if a:
                href = a.get("href","")
                title = a.get_text(strip=True)
                # 상세 페이지 크롤링
                full_url = href if href.startswith("http") else BASE_URL + href
                try:
                    detail = requests.get(full_url, headers=HEADERS, timeout=10)
                    dsoup = BeautifulSoup(detail.text, "lxml")
                    content_tag = dsoup.select_one(".board-view-content,.view-content,.board_content")
                    content = content_tag.get_text(strip=True) if content_tag else ""
                except:
                    content = ""
                all_docs.append({"category": category, "title": title, "content": content})
                time.sleep(0.3)

os.makedirs("data/raw", exist_ok=True)
with open("data/raw/notices.json","w",encoding="utf-8") as f:
    json.dump(all_docs, f, ensure_ascii=False, indent=2)
print(f"크롤링 완료: {len(all_docs)}건")
"""

# %% [4] QA 쌍 자동 생성 (Claude API 또는 GPT API 사용)
"""
import anthropic, json, os

client = anthropic.Anthropic(api_key="YOUR_API_KEY")  # 또는 os.environ["ANTHROPIC_API_KEY"]

with open("data/raw/notices.json", encoding="utf-8") as f:
    docs = json.load(f)

qa_pairs = []
for i, doc in enumerate(docs[:50]):  # 먼저 50건으로 테스트
    if not doc.get("content","").strip():
        continue
    prompt = f'''다음 문서를 읽고 학생이 궁금해할 질문과 답변 쌍을 3개 만들어주세요.
JSON 배열로만 반환하세요: [{{"question":"...","answer":"..."}}]

문서 제목: {doc["title"]}
내용: {doc["content"][:800]}'''

    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role":"user","content":prompt}]
    )
    try:
        text = msg.content[0].text
        start = text.find("[")
        end = text.rfind("]") + 1
        pairs = json.loads(text[start:end])
        for p in pairs:
            p["source"] = doc["category"]
            p["doc_title"] = doc["title"]
        qa_pairs.extend(pairs)
    except:
        pass
    if (i+1) % 10 == 0:
        print(f"{i+1}/{len(docs)} 처리 중... 현재 {len(qa_pairs)}개 QA쌍")

os.makedirs("data/qa_pairs", exist_ok=True)
with open("data/qa_pairs/qa_pairs.json","w",encoding="utf-8") as f:
    json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
print(f"QA 쌍 생성: {len(qa_pairs)}건")
"""

# %% [5] 벡터 DB 구축
"""
from sentence_transformers import SentenceTransformer
import chromadb, json

embedder = SentenceTransformer("jhgan/ko-sroberta-multitask")
client = chromadb.PersistentClient(path="./chroma_db")
col = client.get_or_create_collection("cnu_docs", metadata={"hnsw:space":"cosine"})

with open("data/qa_pairs/qa_pairs.json", encoding="utf-8") as f:
    qa = json.load(f)

# QA 쌍 + 원본 문서 모두 색인
texts, ids, metas = [], [], []
for i, item in enumerate(qa):
    texts.append(item["answer"])
    ids.append(f"qa_{i}")
    metas.append({"category": item.get("source",""), "question": item["question"], "title": item.get("doc_title","")})

batch_size = 100
for start in range(0, len(texts), batch_size):
    batch_texts = texts[start:start+batch_size]
    batch_embs = embedder.encode(batch_texts, normalize_embeddings=True).tolist()
    col.add(ids=ids[start:start+batch_size], documents=batch_texts,
            metadatas=metas[start:start+batch_size], embeddings=batch_embs)
    print(f"색인: {min(start+batch_size, len(texts))}/{len(texts)}")

print("벡터 DB 구축 완료!")
"""

# %% [6] LLM 로드 + 파인튜닝
"""
# [주의] 파인튜닝은 데이터가 충분할 때 (최소 500쌍 이상 권장)
# 먼저 베이스 모델로 성능 확인 후 파인튜닝 여부 결정

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import Dataset
import json

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

quant_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=quant_cfg, device_map="auto", trust_remote_code=True
)

lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj"], bias="none"
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

SYSTEM = "당신은 충남대학교 재학생을 위한 Q&A 챗봇입니다."
with open("data/qa_pairs/qa_pairs.json", encoding="utf-8") as f:
    raw = json.load(f)

def fmt(x):
    msgs = [{"role":"system","content":SYSTEM},
            {"role":"user","content":x["question"]},
            {"role":"assistant","content":x["answer"]}]
    return {"text": tokenizer.apply_chat_template(msgs, tokenize=False)}

ds = Dataset.from_list(raw).map(fmt)

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    args=TrainingArguments(
        output_dir="./models/cnu-qa-lora",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4, fp16=True,
        logging_steps=20, save_strategy="epoch",
        warmup_ratio=0.05, report_to="none",
    ),
    train_dataset=ds, dataset_text_field="text", max_seq_length=1024,
)
trainer.train()
trainer.save_model("./models/cnu-qa-lora")
print("파인튜닝 완료!")
"""

# %% [7] RAG 파이프라인 테스트
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import chromadb

# 모델 로드 (파인튜닝 했으면 lora 적용, 안 했으면 base만)
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = "./models/cnu-qa-lora"

quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=quant_cfg,
                                              device_map="auto", trust_remote_code=True)
import os
if os.path.exists(LORA_PATH):
    model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

embedder = SentenceTransformer("jhgan/ko-sroberta-multitask")
col = chromadb.PersistentClient("./chroma_db").get_collection("cnu_docs")

def rag_answer(question, top_k=3, max_new=512):
    q_emb = embedder.encode([question], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=q_emb, n_results=top_k,
                    include=["documents","metadatas"])
    context = "\n\n".join(
        f"[{r['title']}]\n{d}" for d,r in zip(res["documents"][0], res["metadatas"][0])
    )
    messages = [
        {"role":"system","content":"당신은 충남대학교 재학생을 위한 Q&A 챗봇입니다. 참고 자료를 바탕으로 답변하세요."},
        {"role":"user","content":f"[참고 자료]\n{context}\n\n[질문]\n{question}"}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.1)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

# 테스트
print(rag_answer("충남대 졸업 요건이 어떻게 되나요?"))
"""

# %% [8] Gradio UI 실행 (share=True → 공개 URL)
"""
import gradio as gr

def chat(message, history):
    answer = rag_answer(message)
    return answer

demo = gr.ChatInterface(
    fn=chat,
    title="충남대학교 Q&A 챗봇",
    description="학사, 장학금, 졸업요건 등 충남대 관련 질문을 해보세요.",
    examples=["충남대 졸업학점이 몇 점이에요?", "장학금 신청은 어떻게 하나요?"],
)
demo.launch(share=True)
"""

# %% [9] 평가용 일괄 답변 생성 (교수님 제출용)
"""
import json

# 교수님이 제공하는 questions.json 로드
with open("questions.json", encoding="utf-8") as f:
    questions = json.load(f)

# 질문 형식에 따라 처리
if isinstance(questions[0], dict):
    qs = [q["question"] for q in questions]
else:
    qs = questions

results = []
for i, q in enumerate(qs):
    print(f"[{i+1}/{len(qs)}] 처리 중...")
    ans = rag_answer(q)
    results.append({"question": q, "answer": ans})

with open("answers.json","w",encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("제출용 answers.json 생성 완료!")
"""
