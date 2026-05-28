# 충남대학교 Q&A 챗봇

LLM 기반 RAG(Retrieval-Augmented Generation) Q&A 시스템.  
충남대학교 공식 공지사항을 크롤링하여 학생 질문에 정확하게 답변합니다.

## 프로젝트 구조

```
Termproject/
├── data/
│   ├── raw/          # 크롤링 원본 데이터
│   └── processed/    # 전처리된 청크 데이터
├── src/
│   ├── crawling/     # 크롤러 (Selenium)
│   ├── preprocessing/# 전처리 & 청킹
│   ├── embedding/    # 한국어 임베딩 모델
│   ├── vectordb/     # ChromaDB 구축 & 검색
│   ├── rag/          # RAG 파이프라인
│   ├── api/          # FastAPI 서버
│   └── ui/           # Gradio UI
├── notebooks/
│   └── colab_main.ipynb  # Colab 실행 노트북
├── requirements.txt
└── README.md
```

## 환경 설치

Python 3.11 권장. conda 환경 사용 시:

```bash
conda create -n nlp_project python=3.11 -y
conda activate nlp_project
pip install -r requirements.txt
```

> **Colab 실행 시**: `notebooks/colab_main.ipynb` 파일을 Google Colab에서 열어 순서대로 실행하세요.  
> 런타임 유형은 **T4 GPU** 선택 필수.

---

## 실행 순서 (로컬)

### 1. 크롤링

충남대 공식 포털(plus.cnu.ac.kr)에서 공지사항 수집.

```bash
python src/crawling/cnu_crawler.py
```

- 출력: `data/raw/all_docs.json` (5개 카테고리, 약 100건)
- 소요 시간: 약 10분

### 2. 전처리 & 청킹

노이즈 제거 후 300~400자 단위로 분할.

```bash
python src/preprocessing/preprocess.py
```

- 출력: `data/processed/chunks.json`

### 3. 벡터 DB 구축

한국어 임베딩 모델(`jhgan/ko-sroberta-multitask`)로 ChromaDB에 색인.

```bash
python src/vectordb/build_db.py
```

- 출력: `chroma_db/` 디렉토리
- 최초 실행 시 모델 자동 다운로드 (~500MB)

### 4. 검색 테스트 (선택)

벡터 검색이 정상 동작하는지 확인.

```bash
python test_search.py
```

---

## API 서버 실행

```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```

### 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/health` | 서버 상태 확인 |
| POST | `/ask` | 단건 질문 |
| POST | `/batch` | 질문 리스트 일괄 처리 |

**단건 질문 예시:**

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "장학금 신청은 어떻게 하나요?"}'
```

---

## Gradio UI 실행

```bash
python src/ui/app.py
```

- 로컬: `http://localhost:7860`
- Colab에서 실행 시 `share=True`로 공개 URL 자동 생성

---

## 평가용 답변 생성 (교수님 제출용)

교수님이 제공하는 `questions.json` 파일을 프로젝트 루트에 놓고 실행:

```bash
python src/evaluate.py --questions questions.json --output answers.json
```

**questions.json 형식** (둘 다 지원):

```json
["질문1", "질문2"]
```

또는

```json
[{"question": "질문1"}, {"question": "질문2"}]
```

- 출력: `answers.json` — `[{"question": "...", "answer": "..."}]` 형식

---

## 모델 스펙

| 항목 | 내용 |
|------|------|
| LLM | Qwen/Qwen2.5-3B-Instruct (4-bit QLoRA) |
| 임베딩 | jhgan/ko-sroberta-multitask |
| 벡터DB | ChromaDB |
| VRAM | ~3GB (T4 기준 여유 충분) |
| 할루시네이션 방지 | 유사도 임계값 0.40 미달 시 모름 답변 |
