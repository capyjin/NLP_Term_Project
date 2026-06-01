# PROJECT_STATE — CNU Campus ChatBot
**박연진 | 자연어처리 텀프로젝트 | 마지막 업데이트: 2026-06-01**

---

## 📅 일정

| 항목 | 날짜 | 남은 시간 |
|---|---|---|
| 코드 제출 기한 | 2026-06-12(금) 자정 | **11일** |
| 텀프 발표 | 2026-06-16(화) | **15일** |

---

## ✅ 완료된 기능

### Task 1 — 질문 유형 분류기 (40점)
- [x] `data/train.json` — 130개 학습 데이터 (5 카테고리)
  - 라벨 0 졸업요건: 30개
  - 라벨 1 학교공지사항: 30개
  - 라벨 2 학사일정: 30개
  - 라벨 3 식단안내: 20개
  - 라벨 4 통학/셔틀버스: 20개
- [x] `data/valid.json` — 26개 (train 20% stratified split)
- [x] `classifier_박연진.ipynb` — klue/bert-base fine-tuning (9 셀, 유효 JSON)
  - Colab T4에서 ~10분 학습
  - `data/test_cls.json` 읽어 `outputs/cls_output.json` 자동 생성
  - 기존 모델 있으면 skip → 즉시 추론

### Task 2 — 챗봇 + UI (60점)
- [x] `src/chatbot_model.py` — RAGPipeline CLI 래퍼
  - `data/test_chat.json` → `outputs/chat_output.json`
  - argparse `--input` / `--output` 지원
- [x] `src/chatbot_ui.py` — Gradio ChatInterface
  - `share=True`, `server_name="0.0.0.0"`
  - 예시 질문 5개 포함
- [x] `chatbot.sh` — 평가 실행 스크립트 (Linux/Mac)
  - chroma_db 없으면 자동 구축
  - chat_output.json 생성
  - test_realtime.json 있으면 realtime 처리 (Optional 자동 포함)
  - Gradio UI 실행
- [x] `chatbot.bat` — Windows 버전

### Optional — 실시간 정보 반영 (30점)
- [x] `src/realtime_model.py`
  - 공지사항/학사일정: 기존 RAGPipeline으로 처리 ✅
  - 식단/셔틀버스: fallback 응답 (포털 안내) ⚠️
  - `data/test_realtime.json` → `outputs/realtime_output.json`

### RAG 파이프라인 (기존 구현, 재사용)
- [x] `src/rag/pipeline.py` — Qwen2.5-3B-Instruct (4-bit NF4)
- [x] `src/rag/retriever.py` — BM25(Kiwi) + KURE-v1 Dense + RRF
  - Phase2: search_text 인덱싱, subcategory boost/penalty
- [x] `src/vectordb/chroma_store.py` + `build_db.py`
- [x] `src/embedding/embedder.py` — KURE-v1 (1024d)
- [x] `data/processed/chunks.json` — 181개 (크롤링 158 + FAQ 23)
- [x] `data/faq/faq_manual.json` — 23개 수동 FAQ
- [x] `scripts/inject_faq.py` — FAQ → chunks.json 삽입
- [x] `scripts/make_notebook.py` — 노트북 재생성 스크립트

### 제출 구조 파일
- [x] `README.md`
- [x] `requirements.txt` (수정 필요 — 아래 참조)
- [x] `outputs/` 폴더 생성
- [x] `model/cls_model/` 폴더 생성
- [x] `data/test_cls.json` — placeholder (조교 파일로 교체)
- [x] `data/test_chat.json` — placeholder (조교 파일로 교체)
- [x] `data/test_realtime.json` — placeholder

---

## 🔴 남은 작업

### 필수 (점수 직결)

| 우선순위 | 작업 | 예상 시간 | 점수 |
|:---:|---|:---:|:---:|
| 1 | **Colab에서 classifier 실제 학습 + 출력 확인** | 1시간 | 40점 |
| 2 | **chatbot.sh 로컬/Colab 실제 실행 테스트** | 1시간 | 60점 |
| 3 | **chroma_db Colab 재구축** (FAQ 23개 포함) | 30분 | 필수 |
| 4 | **UI 작동 영상 녹화** (2분 내외) | 30분 | 발표 |
| 5 | **발표 자료 작성** (5분 분량) | 2시간 | 발표 |
| 6 | **최종 zip 압축 + 사이버캠퍼스 제출** | 30분 | 제출 |

### 권장 (점수 향상)

| 작업 | 예상 시간 | 점수 향상 |
|---|:---:|:---:|
| train.json 추가 데이터 보강 (현재 130개 → 200개) | 1시간 | F1 +0.05 |
| 식단/셔틀버스 실시간 크롤러 구현 | 3~5시간 | Optional +5~10점 |
| BGE-Reranker 추가 (RAG 품질 향상) | 2시간 | 응답 품질 |

### 선택 (시간 여유 시)
- Colab 노트북 통합 정리 (classifier + chatbot 흐름 통합)
- requirements.txt 버전 정리

---

## ⚠️ 주의사항

### 조교 실행 환경
```
classifier_박연진.ipynb 실행 시:
  - data/test_cls.json  ← 조교가 교체 (현재 placeholder)
  - model/cls_model/    ← 없으면 자동 학습 (~10분)
  - outputs/cls_output.json ← 자동 생성 확인

chatbot.sh 실행 시:
  - data/test_chat.json ← 조교가 교체 (현재 placeholder)
  - 첫 실행: Qwen2.5-3B + KURE-v1 다운로드 (~10~20분)
  - chroma_db 없으면 자동 구축 (~10분)
  - outputs/chat_output.json ← 자동 생성
```

### 알려진 한계
| 항목 | 상태 | 비고 |
|---|---|---|
| 식단 실시간 크롤링 | ❌ 미구현 | fallback 응답으로 처리 |
| 셔틀버스 실시간 크롤링 | ❌ 미구현 | fallback 응답으로 처리 |
| Colab 실행 시 Drive 폴더명 | ⚠️ 확인 필요 | `classifier_박연진.ipynb` 셀 2 경로 수정 |
| Windows에서 `chatbot.sh` | ⚠️ `.bat` 사용 | Linux/Mac에서는 `.sh` 사용 |
| 모델 다운로드 시간 | ⚠️ ~20분 | 첫 실행 시 README 참조 |

### 파일 경로 고정 사항 (변경 금지)
```
data/test_cls.json        ← classifier 노트북에서 하드코딩
data/test_chat.json       ← chatbot_model.py 기본값
data/test_realtime.json   ← realtime_model.py 기본값
outputs/cls_output.json   ← 조교 평가 기준
outputs/chat_output.json  ← 조교 평가 기준
outputs/realtime_output.json ← 조교 평가 기준
```

---

## 🚀 실행 방법

### Task 1 — 질문 분류기 (Colab)

```
1. Colab 접속 → classifier_박연진.ipynb 열기
2. 런타임 > 런타임 유형 변경 > T4 GPU
3. 셀 1: 패키지 설치 → 런타임 다시 시작
4. 셀 2: 폴더명 확인 후 PROJECT 경로 수정
   (기본값: /content/drive/MyDrive/CNU-QA-chatbot)
5. 셀 3~8 순서대로 실행
6. 결과: outputs/cls_output.json 확인
```

### Task 2 — 챗봇 (Linux/Mac)

```bash
git clone ... # 또는 Drive에서 폴더 접근
cd Termproject_박연진
bash chatbot.sh
# → outputs/chat_output.json 생성
# → http://localhost:7860 에서 UI 확인
```

### Task 2 — 챗봇 (Windows)

```cmd
cd Termproject_박연진
chatbot.bat
```

### Optional — 실시간 정보 (별도 실행)

```bash
python src/realtime_model.py \
  --input  data/test_realtime.json \
  --output outputs/realtime_output.json
```

### chroma_db 재구축 필요 시 (FAQ 반영)

```bash
python scripts/inject_faq.py        # FAQ → chunks.json 삽입
python src/vectordb/build_db.py     # chroma_db 재구축
```

---

## 📁 최종 제출 구조

```
Termproject_박연진/
├── classifier_박연진.ipynb   ← 조교 실행 (Task 1)
├── chatbot.sh                ← 조교 실행 (Task 2, Linux/Mac)
├── chatbot.bat               ← 조교 실행 (Task 2, Windows)
├── README.md
├── requirements.txt
├── PROJECT_STATE.md          ← 이 파일
│
├── data/
│   ├── train.json            (130개)
│   ├── valid.json            (26개)
│   ├── test_cls.json         ← 조교 교체
│   ├── test_chat.json        ← 조교 교체
│   ├── test_realtime.json    ← 조교 교체
│   └── processed/chunks.json (181개)
│
├── src/
│   ├── chatbot_model.py      (RAGPipeline CLI)
│   ├── chatbot_ui.py         (Gradio UI)
│   ├── realtime_model.py     (Optional)
│   ├── rag/                  (pipeline.py, retriever.py)
│   ├── vectordb/             (chroma_store.py, build_db.py)
│   ├── embedding/            (embedder.py — KURE-v1)
│   ├── crawling/             (cnu_crawler.py)
│   └── preprocessing/        (preprocess.py)
│
├── model/cls_model/          (학습 후 자동 생성)
├── outputs/                  (실행 후 자동 생성)
│   ├── cls_output.json
│   ├── chat_output.json
│   └── realtime_output.json
│
├── chroma_db/                (build_db.py 실행 후 생성)
└── scripts/
    ├── inject_faq.py
    └── make_notebook.py
```

---

## 📊 예상 점수

| Task | 만점 | 예상 | 근거 |
|---|:---:|:---:|---|
| Task 1 분류기 F1 | 40 | 32~38 | klue/bert-base + 130개, F1 0.80~0.95 기대 |
| Task 2 UI 구동 | 10 | 10 | Gradio 정상 실행 |
| Task 2 Chat Interface | 10 | 10 | ChatInterface 형식 ✅ |
| Task 2 응답 맥락 | 40 | 30~36 | RAG 기반 맥락 응답 ✅ |
| Optional 실시간 | 30 | 15~22 | 공지/학사 RAG ✅, 식단/셔틀 fallback ⚠️ |
| **합계** | **130** | **97~116** | |
