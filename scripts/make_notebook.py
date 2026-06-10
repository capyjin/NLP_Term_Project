"""
classifier_박연진.ipynb 를 프로그래밍 방식으로 생성.
json.dumps 를 통해 올바른 JSON 직렬화(이스케이핑) 보장.

실행: python scripts/make_notebook.py
"""
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
OUT_PATH = BASE_DIR / "classifier_박연진.ipynb"


def code_cell(cell_id: str, source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def md_cell(cell_id: str, source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source,
    }


# ── 셀 소스 정의 ─────────────────────────────────────────────────────────────
CELL_MD = """\
# CNU Campus ChatBot — 질문 유형 분류기
**박연진 | 자연어처리 텀프로젝트**

## 실행 순서
1. 런타임 > 런타임 유형 변경 > **T4 GPU** 선택
2. **셀 1** 실행 → 런타임 다시 시작
3. **셀 2 ~ 8** 순서대로 실행
4. 최종 출력: `outputs/cls_output.json`

| 라벨 | 카테고리 |
|:---:|---|
| 0 | 졸업요건 |
| 1 | 학교공지사항 |
| 2 | 학사일정 |
| 3 | 식단안내 |
| 4 | 통학/셔틀버스 |\
"""

CELL_INSTALL = """\
# [셀 1] 패키지 설치 — 처음 1회 실행 후 런타임 재시작
!pip install -q transformers datasets scikit-learn tqdm
print('설치 완료 — 런타임 > 런타임 다시 시작 후 셀 2 부터 실행')\
"""

CELL_PATH = """\
# [셀 2] 프로젝트 경로 자동 탐색 — 필요할 때만 PROJECT_OVERRIDE 지정
import os, sys
from pathlib import Path

PROJECT_OVERRIDE = None  # 예: '/content/drive/MyDrive/NLP_Term_Project'

def is_project_root(path):
    path = Path(path)
    return (path / 'data' / 'train.json').is_file() and (path / 'src').is_dir()

def find_project_root():
    if PROJECT_OVERRIDE:
        override = Path(PROJECT_OVERRIDE).expanduser().resolve()
        if not is_project_root(override):
            raise FileNotFoundError(f'PROJECT_OVERRIDE가 프로젝트 루트가 아닙니다: {override}')
        return override

    starts = [Path.cwd()]
    for candidate in ('/content', '/content/drive/MyDrive'):
        path = Path(candidate)
        if path.exists():
            starts.append(path)

    checked = set()
    for start in starts:
        for candidate in (start, *start.parents):
            resolved = candidate.resolve()
            if resolved not in checked and is_project_root(resolved):
                return resolved
            checked.add(resolved)
        if start.is_dir():
            for child in start.iterdir():
                if child.is_dir() and is_project_root(child):
                    return child.resolve()
    raise FileNotFoundError(
        '프로젝트 루트를 찾지 못했습니다. PROJECT_OVERRIDE에 프로젝트 경로를 지정하세요.'
    )

try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    pass

PROJECT = find_project_root()
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))
print(f'프로젝트 루트: {PROJECT}')
print(f'작업 디렉토리: {os.getcwd()}')\
"""

CELL_IMPORTS = """\
# [셀 3] 라이브러리 + 하이퍼파라미터
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from collections import Counter
from tqdm import tqdm

MODEL_NAME = 'klue/bert-base'
NUM_LABELS = 5
MAX_LEN    = 128
BATCH_SIZE = 16
EPOCHS     = 5
LR         = 2e-5
SEED       = 42
FORCE_RETRAIN = True  # clean-run 증거 확보용. 저장 모델 재사용 시 False
MODEL_SAVE = 'model/cls_model'
LABELS     = {0: '졸업요건', 1: '학교공지사항', 2: '학사일정', 3: '식단안내', 4: '셔틀버스'}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {gb:.1f} GB')
else:
    print('CPU 환경 (학습이 느릴 수 있습니다)')\
"""

CELL_DATA = """\
# [셀 4] 데이터 로드 + 누수 없는 검증셋 구성
with open('data/train.json', encoding='utf-8') as f:
    all_data = json.load(f)

def normalized_question(item):
    return ' '.join(item['question'].split()).casefold()

unique_by_question = {}
for item in all_data:
    key = normalized_question(item)
    if key in unique_by_question and unique_by_question[key]['label'] != item['label']:
        raise ValueError(f'같은 질문에 서로 다른 라벨이 있습니다: {item["question"]}')
    unique_by_question.setdefault(key, item)
all_data = list(unique_by_question.values())

use_external_valid = False
if os.path.exists('data/valid.json'):
    with open('data/valid.json', encoding='utf-8') as f:
        candidate_valid = json.load(f)
    train_questions = {normalized_question(d) for d in all_data}
    overlap = sum(normalized_question(d) in train_questions for d in candidate_valid)
    use_external_valid = overlap == 0
    if use_external_valid:
        valid_data = candidate_valid
        print('valid.json 로드 완료 (train과 중복 없음)')
    else:
        print(f'주의: valid.json 중 {overlap}건이 train과 중복되어 자동 분리를 사용합니다.')

if use_external_valid:
    train_data = all_data
else:
    strat = [d['label'] for d in all_data]
    train_data, valid_data = train_test_split(
        all_data, test_size=0.2, random_state=SEED, stratify=strat
    )
    print('train 데이터에서 검증셋 20%를 고정 시드로 분리했습니다.')

assert not (
    {normalized_question(d) for d in train_data}
    & {normalized_question(d) for d in valid_data}
), '학습셋과 검증셋 질문이 중복됩니다.'

print(f'학습: {len(train_data)}건 | 검증: {len(valid_data)}건')
print('카테고리 분포 (학습셋):')
for lbl, cnt in sorted(Counter(d['label'] for d in train_data).items()):
    print(f'  {lbl} {LABELS[lbl]}: {cnt}건')\
"""

CELL_DATASET = """\
# [셀 5] Dataset + DataLoader
class CampusDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data, self.tokenizer, self.max_len = data, tokenizer, max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc = self.tokenizer(
            item['question'],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'token_type_ids': enc.get(
                'token_type_ids',
                torch.zeros(self.max_len, dtype=torch.long)
            ).squeeze(),
            'labels': torch.tensor(item['label'], dtype=torch.long),
        }

tokenizer    = AutoTokenizer.from_pretrained(MODEL_NAME)
loader_generator = torch.Generator().manual_seed(SEED)
train_loader = DataLoader(CampusDataset(train_data, tokenizer, MAX_LEN),
                          batch_size=BATCH_SIZE, shuffle=True,
                          generator=loader_generator)
valid_loader = DataLoader(CampusDataset(valid_data, tokenizer, MAX_LEN),
                          batch_size=BATCH_SIZE)
print(f'DataLoader: train {len(train_loader)}배치 | valid {len(valid_loader)}배치')\
"""

CELL_TRAIN = """\
# [셀 6] 학습 (FORCE_RETRAIN=False이고 저장 모델이 있을 때만 재사용)
model_config = os.path.join(MODEL_SAVE, 'config.json')
if not FORCE_RETRAIN and os.path.exists(model_config):
    print(f'기존 모델 로드: {MODEL_SAVE}')
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE).to(device)
    print('로드 완료')
else:
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    ).to(device)

    optimizer   = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )
    best_f1, best_state = -1.0, None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch}/{EPOCHS}'):
            optimizer.zero_grad()
            out = model(
                input_ids      = batch['input_ids'].to(device),
                attention_mask = batch['attention_mask'].to(device),
                token_type_ids = batch['token_type_ids'].to(device),
                labels         = batch['labels'].to(device),
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            total_loss += out.loss.item()

        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for batch in valid_loader:
                out = model(
                    input_ids      = batch['input_ids'].to(device),
                    attention_mask = batch['attention_mask'].to(device),
                    token_type_ids = batch['token_type_ids'].to(device),
                )
                preds.extend(out.logits.argmax(-1).cpu().numpy())
                truths.extend(batch['labels'].numpy())

        f1 = f1_score(truths, preds, average='macro')
        print(f'Epoch {epoch}/{EPOCHS} | loss {total_loss/len(train_loader):.4f} | val_F1 {f1:.4f}')
        if f1 > best_f1:
            best_f1, best_state = f1, {k: v.clone() for k, v in model.state_dict().items()}
            print(f'  베스트 갱신: {best_f1:.4f}')

    model.load_state_dict(best_state)
    os.makedirs(MODEL_SAVE, exist_ok=True)
    model.save_pretrained(MODEL_SAVE)
    tokenizer.save_pretrained(MODEL_SAVE)
    print(f'저장: {MODEL_SAVE} | 베스트 val_F1: {best_f1:.4f}')\
"""

CELL_REPORT = """\
# [셀 7] 검증셋 분류 리포트
model.eval()
preds, truths = [], []
with torch.no_grad():
    for batch in valid_loader:
        out = model(
            input_ids      = batch['input_ids'].to(device),
            attention_mask = batch['attention_mask'].to(device),
            token_type_ids = batch['token_type_ids'].to(device),
        )
        preds.extend(out.logits.argmax(-1).cpu().numpy())
        truths.extend(batch['labels'].numpy())

macro_f1 = f1_score(truths, preds, average='macro')
print(f'검증셋 F1 (macro): {macro_f1:.4f}')
print()
print(classification_report(truths, preds,
      target_names=[LABELS[i] for i in range(NUM_LABELS)]))\
"""

CELL_INFER = """\
# [셀 8] test_cls.json 추론 → outputs/cls_output.json 저장
TEST_PATH   = 'data/test_cls.json'
OUTPUT_PATH = 'outputs/cls_output.json'

if not os.path.exists(TEST_PATH):
    print(f'테스트 파일 없음: {TEST_PATH}')
    print('조교 제공 test_cls.json 을 data/ 에 넣은 후 이 셀을 다시 실행하세요.')
else:
    with open(TEST_PATH, encoding='utf-8') as f:
        test_data = json.load(f)

    model.eval()
    results = []
    for item in tqdm(test_data, desc='추론'):
        enc = tokenizer(
            item['question'],
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        with torch.no_grad():
            out = model(
                input_ids      = enc['input_ids'].to(device),
                attention_mask = enc['attention_mask'].to(device),
                token_type_ids = enc.get(
                    'token_type_ids',
                    torch.zeros(1, MAX_LEN, dtype=torch.long)
                ).to(device),
            )
        pred = out.logits.argmax(-1).item()
        results.append({'question': item['question'], 'label': pred})

    os.makedirs('outputs', exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f'저장 완료: {OUTPUT_PATH} ({len(results)}건)')
    print()
    for r in results[:5]:
        q   = r['question'][:50]
        lbl = r['label']
        print(f'  Q: {q:<50} -> {lbl} ({LABELS[lbl]})')\
"""

# ── 노트북 조립 ───────────────────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.12"},
        "colab": {"provenance": [], "gpuType": "T4"},
        "accelerator": "GPU",
    },
    "cells": [
        md_cell("md-title",    CELL_MD),
        code_cell("c-install", CELL_INSTALL),
        code_cell("c-path",    CELL_PATH),
        code_cell("c-imports", CELL_IMPORTS),
        code_cell("c-data",    CELL_DATA),
        code_cell("c-dataset", CELL_DATASET),
        code_cell("c-train",   CELL_TRAIN),
        code_cell("c-report",  CELL_REPORT),
        code_cell("c-infer",   CELL_INFER),
    ],
}

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

# 검증
with open(OUT_PATH, encoding="utf-8") as f:
    nb = json.load(f)
print(f"notebook 생성 완료: {OUT_PATH}")
print(f"  cells: {len(nb['cells'])}")
for i, c in enumerate(nb["cells"]):
    src_len = len(c["source"])
    print(f"  [{i}] {c['cell_type']:8} | source {src_len}자")
