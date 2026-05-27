"""
LoRA 파인튜닝 스크립트 (Colab Free Tier 기준)
베이스 모델: Qwen2.5-7B-Instruct
방식: QLoRA (4-bit + LoRA) → VRAM ~8GB
데이터: data/qa_pairs/qa_pairs.json ({"question": ..., "answer": ...} 형식)
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

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "./models/cnu-qa-lora"
QA_PATH = "./data/qa_pairs/qa_pairs.json"

SYSTEM_PROMPT = "당신은 충남대학교 재학생을 위한 Q&A 챗봇입니다. 질문에 정확하고 친절하게 답변하세요."


def load_dataset(path: str) -> Dataset:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)


def format_sample(sample: dict, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": sample["question"]},
        {"role": "assistant", "content": sample["answer"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def train():
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset(QA_PATH)
    dataset = dataset.map(lambda x: {"text": format_sample(x, tokenizer)})

    train_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
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


if __name__ == "__main__":
    train()
