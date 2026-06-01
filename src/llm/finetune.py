"""
QLoRA 파인튜닝 스크립트
────────────────────────
베이스 모델: Qwen/Qwen2.5-7B-Instruct
방식: QLoRA (4-bit 양자화 + LoRA 어댑터) → VRAM ~8GB (T4 Pro/A100 필요)
데이터: data/qa_pairs/qa_pairs.json  형식: [{"question": ..., "answer": ...}]
출력: models/cnu-qa-lora/  (LoRA 어댑터 가중치)

⚠️ 주의사항:
  1. Colab Free T4 (16GB VRAM)에서는 7B 파인튜닝이 빡빡함 — A100 권장
  2. qa_pairs.json이 없으면 실행 불가 (별도 생성 필요)
  3. pipeline.py는 3B 모델 기본 사용 — 파인튜닝 후 DEFAULT_MODEL 변경 필요:
       pipeline.py: DEFAULT_MODEL = "./models/cnu-qa-lora"

LoRA 설정 (lora_config):
  r=16: LoRA 랭크 (클수록 표현력↑, VRAM↑)
  lora_alpha=32: 스케일링 계수 (alpha/r = 학습률 스케일)
  target_modules: Attention 레이어만 학습 (q/k/v/o_proj)
  lora_dropout=0.05: 과적합 방지

학습 설정:
  num_train_epochs=3, per_device_train_batch_size=2, gradient_accumulation_steps=4
  effective_batch_size = 2 × 4 = 8
  lr=2e-4, cosine scheduler, warmup_ratio=0.05
"""

import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# ⚠️ pipeline.py는 3B 사용, 파인튜닝은 7B — 결과 모델 사용 시 pipeline.py 수정 필요
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "./models/cnu-qa-lora"
QA_PATH    = "./data/qa_pairs/qa_pairs.json"   # 미생성 시 FileNotFoundError

SYSTEM_PROMPT = "당신은 충남대학교 재학생을 위한 Q&A 챗봇입니다. 질문에 정확하고 친절하게 답변하세요."


def load_dataset(path: str) -> Dataset:
    """qa_pairs.json → HuggingFace Dataset 변환."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)


def format_sample(sample: dict, tokenizer) -> str:
    """
    {"question": ..., "answer": ...} → Qwen chat template 형식 변환.
    SFTTrainer가 이 형식으로 학습.
    """
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": sample["question"]},
        {"role": "assistant", "content": sample["answer"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def train():
    """QLoRA 파인튜닝 메인 함수."""
    # 4-bit 양자화 설정 (파인튜닝 시에도 기반 가중치 압축)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token   # Qwen은 pad_token 없음 → eos로 대체

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA 어댑터 설정: Attention 레이어 4개에만 학습 파라미터 추가
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,                # 랭크: 16 (표준값)
        lora_alpha=32,       # 스케일: alpha/r=2.0
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()   # 전체 파라미터 중 학습 비율 출력

    dataset = load_dataset(QA_PATH)
    dataset = dataset.map(lambda x: {"text": format_sample(x, tokenizer)})

    train_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # effective batch = 2×4 = 8
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=1024,
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"파인튜닝 완료: {OUTPUT_DIR}")
    print(f"사용하려면 pipeline.py의 DEFAULT_MODEL = '{OUTPUT_DIR}'로 변경하세요.")


if __name__ == "__main__":
    train()
