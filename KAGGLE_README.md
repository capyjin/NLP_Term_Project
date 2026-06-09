# Kaggle 실행 가이드 — NLP Term Project

> **과제 지정 환경: Python 3.10.12 / torch 2.5.1**

---

## 사전 준비 (처음 1회)

### 1. Kaggle 전화번호 인증 (필수)
GPU와 인터넷을 사용하려면 인증이 필요합니다.

1. [kaggle.com](https://www.kaggle.com) → 로그인
2. 우측 상단 프로필 → **Settings**
3. **Phone Verification** → 번호 입력 → 인증 완료

---

## Notebook 생성 및 설정

### 2. 새 Notebook 만들기

1. Kaggle 좌측 메뉴 **Code** 클릭
2. 우측 상단 **+ New Notebook** 클릭
3. 제목: `NLP_Term_Project`

### 3. GPU 설정 (필수)

우측 패널 **Session Options**:

```
Accelerator: GPU T4 x2  ← 선택 (VRAM 30GB, 무료)
```

> T4 x2 없으면 P100 (16GB) 선택.

### 4. Internet 설정 (필수)

같은 **Session Options** 패널:

```
Internet: ON  ← 반드시 켜기
```

> 꺼져 있으면 GitHub clone, HuggingFace 다운로드 불가.

---

## 노트북 로드 방법

### 방법 A: 파일 업로드

1. `notebooks/kaggle_main.ipynb` 파일을 로컬에 저장
2. Kaggle에서 **File > Import Notebook** 선택
3. 파일 업로드

### 방법 B: GitHub에서 직접 (권장)

Kaggle Notebook 첫 셀에 붙여넣기:

```python
import os
os.system("git clone https://github.com/본인아이디/NLP_Term_Project.git /kaggle/working/NLP_Term_Project")
os.chdir("/kaggle/working/NLP_Term_Project")
```

그 다음 `notebooks/kaggle_main.ipynb` 셀을 하나씩 복붙하거나, 파일을 열어서 실행.

---

## 실행 순서

| 셀 | 내용 | 예상 시간 | 비고 |
|---|---|---|---|
| Cell 1 | 환경 확인 | 30초 | Python/GPU/torch 상태 확인 |
| Cell 2 | torch 2.5.1 설치 | 2~3분 | **실행 후 커널 재시작 필수** |
| — | **커널 재시작** | — | Run > Restart Session |
| Cell 3 | 패키지 설치 | 3~5분 | 재시작 후 여기서 시작 |
| Cell 4 | 프로젝트 경로 설정 | 30초 | GitHub clone 포함 |
| Cell 5 | 데이터 파일 확인 | 10초 | |
| Cell 6 | refresh_data | 1~2분 | 식단·셔틀·학사일정 갱신 |
| Cell 7 | ChromaDB 구축 | 2~5분 | 이미 있으면 자동 건너뜀 |
| Cell 8 | Task 2 실행 | 3~5분 | chat_output.json |
| Cell 9 | Optional 실행 | 1~2분 | realtime_output.json |
| Cell 10 | Task 1 실행 | 10~30분 | cls_output.json (KLUE-BERT 학습) |
| Cell 11 | Gradio UI | — | 공개 URL 출력 (gradio.live) |
| Cell 12 | Output 검증 | 10초 | |
| Cell 13 | requirements.txt | 30초 | pip freeze + git push |

---

## Python 3.10.12 / torch 2.5.1 확인 방법

Cell 1 실행 후 출력 예시:

```
Python : 3.10.x (OK) 또는 3.12.x (WARN)
torch  : 2.5.1+cu121  ← 이게 목표
CUDA   : 12.1
GPU OK : True
```

Cell 2 실행 후 `torch 2.5.1` 확인:

```python
import torch
print(torch.__version__)  # 2.5.1+cu121
print(torch.cuda.is_available())  # True
```

---

## 커맨드라인 스크립트 사용법

`notebooks/kaggle_main.ipynb` 대신 스크립트로 실행할 수도 있습니다.

```bash
# 환경 확인
python scripts/kaggle_setup_and_run.py --mode check

# torch 2.5.1 + 패키지 설치
python scripts/kaggle_setup_and_run.py --mode install

# 전체 실행 (refresh → build_db → chat → realtime → 검증)
python scripts/kaggle_setup_and_run.py --mode all

# Task 2만 실행
python scripts/kaggle_setup_and_run.py --mode chat

# Optional만 실행
python scripts/kaggle_setup_and_run.py --mode realtime

# Gradio UI
python scripts/kaggle_setup_and_run.py --mode ui
```

---

## Output 파일 위치

| 파일 | 경로 | 생성 셀 |
|---|---|---|
| Task 1 | `outputs/cls_output.json` | Cell 10 |
| Task 2 | `outputs/chat_output.json` | Cell 8 |
| Optional | `outputs/realtime_output.json` | Cell 9 |
| requirements | `requirements.txt` | Cell 13 |

### Kaggle에서 파일 다운로드

1. 우측 패널 **Output** 탭
2. `outputs/` 폴더 찾기
3. 파일 선택 → 다운로드

또는 Cell 13 실행 시 자동으로 `git push`됩니다.

---

## 오류 해결

### torch CUDA 버전 불일치
```
증상: torch.cuda.is_available() == False  또는  설치 중 오류
원인: CUDA 12.1용 wheel을 설치했는데 Kaggle GPU가 다른 버전
해결: nvidia-smi 출력에서 CUDA 버전 확인 후 cu118/cu121/cu124 선택
```

```python
# CUDA 버전 확인
import subprocess
r = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
print(r.stdout[:300])
```

### bitsandbytes 오류
```
증상: CUDA Setup failed!
원인: bitsandbytes가 GPU를 인식 못함
해결: pip install bitsandbytes --upgrade 후 커널 재시작
```

### ChromaDB 구축 실패
```
증상: build_db.py 오류
원인: chunks.json 없거나 경로 오류
해결: Cell 4(경로 설정) 재실행 → Cell 5(파일 확인) → Cell 7 재실행
```

### MOCK 응답 포함
```
증상: chat_output.json에 [MOCK] 텍스트
원인: ChromaDB가 구축되지 않음
해결: Cell 7 재실행 (chroma_db 폴더 삭제 후)
```

### Gradio URL 미출력
```
증상: gradio.live URL이 안 나옴
원인: share=False 또는 네트워크 제한
해결: Cell 11이 share=True를 자동 적용함 — 모델 로드 완료까지 3분 대기
```

### 파일 경로 오류
```
증상: FileNotFoundError
원인: cwd가 프로젝트 루트가 아님
해결: os.chdir("/kaggle/working/NLP_Term_Project") 실행 후 재시도
```

### torch가 다른 패키지에 덮어써짐
```
증상: Cell 3 후 torch 버전이 2.5.1이 아님
원인: sentence-transformers 등이 torch를 업그레이드
해결: Cell 2 재실행 → 커널 재시작 → Cell 3 재실행
```

---

## Colab으로 다시 돌아올 때

Kaggle에서 작업 후 Colab으로 복귀 시 확인 사항:

| 항목 | 조치 |
|---|---|
| 코드 변경 사항 | `git pull origin main` |
| `chroma_db` | 매 세션 rebuild (경로 무관) |
| `outputs/` | git push 되어 있으면 자동 반영 |
| `requirements.txt` | Colab 환경에서 `pip freeze > requirements.txt` 재생성 |
| 모델 Drive 캐시 | `/content/drive/MyDrive/models/` 그대로 유지 |
| shell script | `bash chatbot.sh` 동일하게 작동 |

---

## 오늘 반드시 해야 할 것

```
1. Kaggle 전화인증 (아직 안 했다면)
2. GPU T4 x2 + Internet ON 설정
3. Cell 1~9 순서대로 실행
4. chat_output.json / realtime_output.json MOCK 없이 생성 확인
5. Cell 13: pip freeze > requirements.txt → git push
```

## 나중에 해도 되는 것

```
- Task 1 (Cell 10): KLUE-BERT 학습 시간 있을 때
- Gradio UI 화면 녹화 (발표 준비)
- Kaggle Dataset에 모델 캐시 업로드 (속도 최적화)
```

## 절대 건드리면 안 되는 것

```
- data/processed/chunks.json  (RAG 데이터 원본)
- data/train.json, data/valid.json  (학습 데이터)
- src/rag/pipeline.py의 DEFAULT_MODEL 경로  (Drive fallback 로직)
- .gitignore의 chroma_db/ 항목
```
