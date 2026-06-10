# CNU Campus ChatBot

충남대학교 재학생을 위한 AI 챗봇 — 자연어처리 텀프로젝트
**박연진**

---

## 평가 실행 방법 (조교 확인용)

### Task 1 — 질문 유형 분류기 (40점)

```
1. Colab에서 classifier_박연진.ipynb 열기
2. 런타임 > T4 GPU 선택
3. 셀 1 실행 → 런타임 다시 시작
4. 셀 2~8 순서대로 실행
5. 결과: outputs/cls_output.json 자동 생성
```

- 셀 2는 `data/train.json`과 `src/`를 기준으로 프로젝트 루트를 자동 탐색합니다.
- `valid.json`이 train과 겹치면 누수를 경고하고 고정 시드로 검증셋을 다시 분리합니다.
- clean-run 증거 확보를 위해 기본값은 `FORCE_RETRAIN=True`입니다.

### Task 2 — 챗봇 실행 (60점)

**평가용 (추론 + UI):**
```bash
bash chatbot.sh
```

**데모용 (데이터 갱신 + UI만 바로 실행):**
```bash
bash run_ui.sh           # 식단/셔틀/장학 TTL 갱신 후 UI 실행
bash run_ui.sh --skip    # 데이터 갱신 건너뛰고 UI 바로 실행
```

**데이터만 선택 갱신:**
```bash
python scripts/refresh_data.py           # 전체 크롤링
python scripts/refresh_data.py --meal    # 식단만
python scripts/refresh_data.py --shuttle # 셔틀만
python scripts/refresh_data.py --notice  # 공지/장학만
```

Windows:
```cmd
chatbot.bat
```

실행 흐름:
1. FAQ 반영 후 벡터 DB 정합성 검사, 불일치 시 자동 전체 재구축 (~10분)
2. data/test_chat.json → outputs/chat_output.json 생성
3. Gradio UI 실행 → http://localhost:7860

### Optional — 실시간 정보 반영 (30점)

```bash
python src/realtime_model.py --input data/test_realtime.json --output outputs/realtime_output.json
```

---

## 환경

- Python 3.10.12 / torch 2.5.1
- T4 GPU 권장 (VRAM 15GB)
- 첫 실행 시 HuggingFace에서 모델 자동 다운로드 (~4GB, 10~20분)

```bash
pip install -r requirements.txt
```

### 모델 전달 및 다운로드

- 분류 모델 원본: `klue/bert-base` (노트북 셀 6에서 fine-tuning 후 `model/cls_model` 저장)
- 기본 생성 모델: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
- 검색 임베딩 모델: https://huggingface.co/nlpai-lab/KURE-v1
- 첫 실행에는 Hugging Face 다운로드가 필요하므로 네트워크 연결이 필요합니다.

---

## 구현 내용

| Task | 방법 |
|---|---|
| 질문 분류기 | klue/bert-base fine-tuning (5-class, macro F1) |
| 응답 생성 | Qwen/Qwen2.5-3B-Instruct 기본 (4-bit NF4), 7B는 실험용 선택지 |
| 문서 검색 | BM25(Kiwi 형태소) + KURE-v1 Dense Retrieval + RRF |
| 벡터 DB | ChromaDB (KURE-v1 1024d) |
| UI | Gradio ChatInterface |
| 실시간 정보 | 공지/학사일정: RAG, 식단: meal_crawler.py 크롤링 → MealHandler, 셔틀버스: ShuttleHandler |

---

## 라벨 체계

| 라벨 | 카테고리 | 예시 |
|:---:|---|---|
| 0 | 졸업요건 | 졸업하려면 몇 학점 필요해? |
| 1 | 학교공지사항 | 최근 공지사항 알려줘 |
| 2 | 학사일정 | 수강신청 기간이 언제야? |
| 3 | 식단안내 | 오늘 학식 뭐 나와요? |
| 4 | 통학/셔틀버스 | 셔틀버스 시간표 알려줘 |

---

## 학습 데이터

- `data/train.json`: 219개 (라벨별 41~45개 균등, 라우팅 충돌 제거)
- `data/valid.json`: 30개 (현재 train 중복이 있어 노트북이 누수 없는 자동 분리를 사용)
- `data/test_cls.json`: 조교 제공 파일 (현재 placeholder)
- `data/test_chat.json`: 조교 제공 파일 (현재 placeholder)
