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

### Task 2 — 챗봇 실행 (60점)

Linux / Mac:
```bash
bash chatbot.sh
```

Windows:
```cmd
chatbot.bat
```

실행 흐름:
1. 첫 실행: 벡터 DB 자동 구축 (~10분, Qwen2.5-3B 다운로드 포함)
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

---

## 구현 내용

| Task | 방법 |
|---|---|
| 질문 분류기 | klue/bert-base fine-tuning (5-class, macro F1) |
| 응답 생성 | Qwen/Qwen2.5-3B-Instruct (4-bit NF4 양자화) |
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

- `data/train.json`: 130개 (라벨 0~2 각 30개, 라벨 3~4 각 20개)
- `data/valid.json`: 26개 (train에서 20% stratified split)
- `data/test_cls.json`: 조교 제공 파일 (현재 placeholder)
- `data/test_chat.json`: 조교 제공 파일 (현재 placeholder)
