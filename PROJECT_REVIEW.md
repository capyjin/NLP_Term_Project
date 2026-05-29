# Term Project 코드/파일 리뷰

작성일: 2026-05-28  
분석 대상: `2_term_project_QA.pdf`, `src/`, `notebooks/`, `data/`, `chroma_db/`, `requirements.txt`, `test_search.py`

## 1. 과제 요구사항 요약

`2_term_project_QA.pdf`에서 확인한 핵심 요구사항은 다음과 같습니다.

- LLM을 이용한 Q/A 시스템 구현을 이해할 것
- RAG(Retrieval-Augmented Generation) 또는 LLM 기반 QA 시스템을 구현할 것
- Machine Learning 분야에 적용 가능한 실용적인 Q/A 시스템을 개발할 것
- 문서 → 임베딩 모델 → 벡터 DB → LLM → 챗봇/Web App/API 형태의 동작 시스템을 만들 것
- 질문을 입력하면 답변을 반환하는 챗봇 및 API가 동작해야 함

현재 프로젝트는 충남대학교 공지 데이터를 지식원으로 사용해 RAG 기반 학생 Q&A 챗봇을 구현하는 방향으로 구성되어 있습니다.

## 2. 현재 프로젝트 구조 요약

```text
Termproject/
- 2_term_project_QA.pdf
- requirements.txt
- test_search.py
- data/
  - raw/all_docs.json, crawled.json
  - processed/chunks.json
  - qa_pairs/                # 현재 비어 있음
- chroma_db/                 # ChromaDB 저장소
- models/                    # 현재 비어 있음
- notebooks/colab_main.py
- src/
  - crawling/                # 충남대 공지 크롤링, PDF 보강
  - preprocessing/           # 본문 정제 및 청킹
  - embedding/               # 한국어 임베딩 래퍼
  - vectordb/                # ChromaDB 구축/검색
  - rag/                     # RAG 파이프라인
  - llm/                     # LoRA 파인튜닝 스크립트
  - api/                     # FastAPI 서버
  - ui/                      # Gradio UI
  - evaluate.py              # 질문 파일 → 답변 파일 생성
```

확인한 데이터 수치:

| 항목 | 확인 결과 |
|---|---:|
| `data/raw/crawled.json` | 100개 문서 |
| `data/raw/all_docs.json` | 100개 문서 |
| `data/processed/chunks.json` | 158개 청크 |
| ChromaDB 컬렉션 | 1개 |
| ChromaDB embedding 수 | 158개 |
| 원본 카테고리 분포 | 학사/장학/일반/취업/행사 각 20개 |
| 처리 후 청크 분포 | 학사 18, 장학 33, 일반 77, 취업 26, 행사 4 |

## 3. 잘한 점

### 3.1 과제 요구사항과 구조가 잘 맞음

PDF의 요구 흐름인 `Documents → Embedding Model → Vector Database → LLM → Chat Bot/Web App/API` 구조가 코드에 거의 그대로 반영되어 있습니다.

- 문서 수집: `src/crawling/cnu_crawler.py`
- 전처리/청킹: `src/preprocessing/preprocess.py`
- 임베딩: `src/embedding/embedder.py`
- 벡터 DB: `src/vectordb/chroma_store.py`, `src/vectordb/build_db.py`
- RAG 답변 생성: `src/rag/pipeline.py`
- API: `src/api/server.py`
- UI: `src/ui/app.py`

### 3.2 RAG 파이프라인이 모듈별로 분리되어 있음

크롤링, 전처리, 임베딩, 벡터 저장소, LLM 생성, API/UI가 각각 별도 파일로 나뉘어 있어 전체 흐름을 이해하기 쉽습니다. 과제 발표나 보고서에서 시스템 아키텍처를 설명하기 좋은 구조입니다.

### 3.3 한국어 환경을 고려한 모델 선택

`jhgan/ko-sroberta-multitask`를 임베딩 모델로 사용하고, 충남대 공지처럼 한국어 문서가 중심인 데이터에 맞춰 설계한 점이 좋습니다. 생성 모델도 `Qwen/Qwen2.5-3B-Instruct`를 기본값으로 두어 Colab/개인 GPU 환경에서 현실적으로 실행 가능한 방향입니다.

### 3.4 환각 방지 로직이 있음

`src/rag/pipeline.py`에서 유사도 임계값(`SIMILARITY_THRESHOLD = 0.40`)보다 낮으면 “해당 정보를 찾을 수 없습니다...” 형태로 답하도록 되어 있습니다. 참고 자료에 없는 내용을 억지로 생성하지 않도록 설계한 점은 Q/A 시스템에서 중요한 장점입니다.

### 3.5 API와 UI를 모두 제공함

- `FastAPI` 기반 `/ask`, `/batch`, `/health` 엔드포인트가 있음
- `Gradio` 기반 간단한 챗봇 UI가 있음
- `src/evaluate.py`로 질문 파일을 받아 답변 파일을 생성하는 평가 흐름도 있음

즉, 단순 모델 코드가 아니라 실제 사용/평가 가능한 인터페이스까지 고려했습니다.

### 3.6 데이터 구축 결과물이 존재함

`data/processed/chunks.json`와 `chroma_db/`가 이미 생성되어 있어, 단순 코드 작성에서 끝나지 않고 실제 문서 색인 단계까지 진행된 흔적이 있습니다. ChromaDB에도 158개 임베딩이 저장되어 있음을 확인했습니다.

## 4. 부족한 점

### 4.1 실행 방법 문서가 없음

현재 루트에 `README.md`나 실행 가이드가 없습니다. 과제 제출/평가 관점에서는 다음 정보가 반드시 필요합니다.

- 환경 설치 방법
- 크롤링 실행 순서
- 전처리 실행 순서
- 벡터 DB 구축 방법
- API 실행 명령어
- Gradio UI 실행 명령어
- 평가용 질문 파일을 넣고 답변을 생성하는 방법

코드는 나뉘어 있지만, 처음 보는 사람이 어떤 순서로 실행해야 하는지 파악하기 어렵습니다.

### 4.2 `requirements.txt`에 누락된 패키지가 있음

코드에서는 사용하지만 `requirements.txt`에 없는 패키지가 있습니다.

| 코드에서 사용 | 누락된 패키지명 |
|---|---|
| `import pdfplumber` | `pdfplumber` |
| `from webdriver_manager.chrome import ChromeDriverManager` | `webdriver-manager` |

이 상태에서는 크롤링/PDF 보강 코드를 새 환경에서 바로 실행할 때 실패할 수 있습니다.

### 4.3 `notebooks/colab_main.py`는 일반 Python 파일로 실행 불가

`notebooks/colab_main.py`에는 다음과 같은 Colab 전용 셸 명령이 포함되어 있습니다.

```python
!pip install -q ...
!git clone ...
```

그래서 `python -m py_compile notebooks/colab_main.py` 기준으로는 `SyntaxError`가 발생합니다. Colab용 문서라면 `.ipynb`로 관리하거나, 일반 `.py` 파일로 유지하려면 셸 명령을 주석/함수화하는 편이 좋습니다.

### 4.4 데이터 증강/파인튜닝 흐름이 완성되지 않음

`src/crawling/data_augment.py`는 `TODO`와 `NotImplementedError`가 남아 있습니다. 또한 `data/qa_pairs/`와 `models/` 폴더가 비어 있어 QA pair 생성과 LoRA 파인튜닝 결과까지는 실제로 완성되지 않은 상태로 보입니다.

RAG만으로 과제 요구사항은 충족 가능하지만, 코드상 파인튜닝 경로를 제시했다면 최소한 “미사용/선택 기능”임을 README나 보고서에서 명확히 설명하는 것이 좋습니다.

### 4.5 데이터 품질 이슈가 있음

원본 JSON 100개 문서의 `title` 필드가 모두 비어 있습니다. 전처리 단계에서 본문 첫 줄로 제목을 보완하고 있지만, 원본 수집 단계의 품질은 개선 여지가 있습니다.

또한 처리 후 청크 분포가 `일반공지 77개`, `행사안내 4개`로 많이 치우쳐 있습니다. 일부 카테고리에 대해 질문이 들어오면 검색 품질이 낮아질 가능성이 있습니다.

### 4.6 예외 처리가 너무 조용함

크롤러 코드에 `except:` 또는 넓은 예외 처리 후 빈 문자열을 반환하는 부분이 있습니다. 실패 로그가 충분하지 않으면 다음 문제가 생깁니다.

- 어떤 URL/PDF에서 실패했는지 추적하기 어려움
- 데이터 누락이 조용히 발생함
- 재현성과 디버깅이 어려움

과제 평가 시에는 “왜 이 데이터만 들어갔는지” 설명하기 어려울 수 있습니다.

### 4.7 테스트와 정량 평가가 부족함

`test_search.py`는 검색 결과를 눈으로 확인하는 스크립트에 가깝고, 자동화된 단위 테스트/평가 지표는 부족합니다.

부족한 예:

- 전처리 함수 테스트
- 벡터 검색 top-k 결과 테스트
- 근거 없는 질문에 대한 fallback 테스트
- API `/ask`, `/batch` 테스트
- 샘플 질문/정답 기반 정확도 또는 수동 평가표

### 4.8 경로와 실행 환경이 일부 하드코딩되어 있음

`test_search.py`에는 절대 경로가 들어 있습니다.

```python
C:\충남대학교\3-1\자연어처리\Termproject
```

다른 PC나 제출 환경에서는 바로 실행되지 않을 수 있으므로 `Path(__file__)` 기준 상대 경로로 바꾸는 것이 좋습니다.

### 4.9 모델 로딩 비용에 대한 안내가 부족함

`RAGPipeline()` 생성 시 ChromaDB, 임베딩 모델, LLM이 바로 로드됩니다. API 서버나 Gradio UI 실행 시 GPU/VRAM이 부족하면 실행이 실패하거나 매우 느릴 수 있습니다. README에 권장 환경과 CPU/GPU 실행 방법을 적는 것이 좋습니다.

## 5. 개선 우선순위 제안

### 우선순위 1: 제출 안정성

1. `README.md` 작성
2. `requirements.txt`에 `pdfplumber`, `webdriver-manager` 추가
3. `test_search.py`의 절대 경로 제거
4. Colab 전용 파일을 `.ipynb`로 분리하거나 일반 Python 문법으로 정리

### 우선순위 2: 평가 대응력

1. 샘플 질문 10~20개와 답변 결과 저장
2. 근거 문서 출처가 답변에 포함되도록 개선
3. `/ask`, `/batch`, `evaluate.py` 실행 예시 추가
4. 검색 실패/낮은 유사도 질문 예시도 포함

### 우선순위 3: 데이터 품질

1. 크롤링 단계에서 제목 추출 보강
2. 카테고리별 문서/청크 수 균형 개선
3. PDF 추출 실패 로그 저장
4. 중복 문서/짧은 문서 제거 기준을 보고서에 명시

### 우선순위 4: 코드 품질

1. 넓은 `except:`를 구체적인 예외 처리와 로그로 변경
2. 설정값을 `.env` 또는 config 파일로 분리
3. `SIMILARITY_THRESHOLD`를 실험 결과 기반으로 조정
4. 간단한 pytest 테스트 추가

## 6. 종합 평가

전체적으로 과제의 핵심 요구사항인 RAG 기반 Q/A 시스템 구조는 잘 갖추어져 있습니다. 특히 문서 수집부터 전처리, 벡터 DB, 검색, LLM 답변 생성, API, UI까지 이어지는 전체 파이프라인이 구현되어 있다는 점이 가장 큰 장점입니다.

다만 제출 완성도 관점에서는 실행 문서, 의존성 정리, 테스트/평가 결과, 데이터 품질 설명이 부족합니다. 기능 구현 자체는 과제 방향과 잘 맞지만, 평가자가 새 환경에서 재현하고 결과를 확인하기 쉽도록 README와 실행 예시를 보강하는 것이 가장 중요합니다.

## 7. 확인한 검증 결과

- `python -m compileall -q src test_search.py` 통과
- `notebooks/colab_main.py`는 Colab 셸 명령 때문에 일반 Python 컴파일 실패 확인
- `requirements.txt`와 import 목록 대조 결과 `pdfplumber`, `webdriver-manager` 누락 확인
- `data/processed/chunks.json` 158개 청크 확인
- `chroma_db/chroma.sqlite3` 내 embedding 158개 확인
