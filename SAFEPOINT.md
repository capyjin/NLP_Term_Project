# 세이프포인트 — CNU Campus ChatBot 안정 동작 상태

> 생성 시각: 2026-06-09  
> 브랜치: `safe/cnu-chatbot-working-20260609`  
> 기준 커밋: `50aa03b51bc612d63c387bb2ead1bee7d6671f3c`

---

## 실행 성공 조건

| 항목 | 값 |
|---|---|
| Python | 3.12.x (Colab 기본값, 과제 요구 3.10.12는 Colab 고정값) |
| torch | 2.5.1+cu124 |
| transformers | 4.47.1 |
| tokenizers | 0.21.0 |
| bitsandbytes | 0.43.3 |
| chromadb | 0.5.5 |
| sentence-transformers | 3.0.1 |
| pytorch-lightning | 2.4.0 |
| GPU | Tesla T4 (VRAM 15.6 GB) |

---

## Cell 실행 순서 (매 세션)

```
[최초 1회]
  셀 A  → 패키지 설치
        → [런타임 다시 시작]
  셀 B  → git clone (Drive에 없을 때만)

[매 세션]
  셀 0  → Drive 마운트 + git pull + 패키지 확인
  셀 1  → 데이터 갱신 (식단/셔틀/공지 크롤링) — 선택적
  셀 2  → ChromaDB 구축 (chroma_db 없을 때만)
  셀 3  → Qwen2.5-3B Drive 캐시 확인 (없으면 HF 다운로드 ~5분)
  셀 4  → 실행 전 체크리스트 확인
  셀 5  → chatbot.sh 실행 (추론 + UI)
  셀 6  → chat_output.json 결과 확인
```

---

## 정상 동작 확인된 질문/답변 예시

| 질문 | 라우팅 | 정상 응답 여부 |
|---|---|---|
| 졸업 학점이 몇 점인가요? | direct_handler (학사일정) | ✓ |
| 장학금 신청은 어디서 해? | rag_pipeline | ✓ |
| 수강신청 변경 기간은 언제야? | direct_handler | ✓ |
| 오늘 학식 뭐 나와? | MealHandler (cat=3) | ✓ |
| 셔틀버스 시간표 알려줘 | ShuttleHandler (cat=4) | ✓ |
| 최근 장학공지 보여줘 | ScholarshipHandler (cat=5) | ✓ |

---

## 경로 정보

| 항목 | 경로 | 삭제 여부 |
|---|---|---|
| 프로젝트 루트 | `/content/drive/MyDrive/NLP_Term_Project` | **삭제 금지** |
| Qwen2.5-3B Drive 캐시 | `/content/drive/MyDrive/models/qwen2.5-3b-4bit` | **삭제 금지** (재다운로드 ~5분) |
| Qwen2.5-7B Drive 캐시 | `/content/drive/MyDrive/models/qwen2.5-7b-4bit` | 삭제 금지 (실험용) |
| ChromaDB | `[프로젝트루트]/chroma_db` | 재구축 가능 (셀 2) |
| 청크 데이터 | `data/processed/chunks.json` | **삭제 금지** |
| 식단 데이터 | `data/raw/meal_menu.json` | 재크롤링 가능 |
| 셔틀 데이터 | `data/raw/shuttle_bus.json` | 재크롤링 가능 |
| 추론 결과 | `outputs/chat_output.json` | 재생성 가능 |

---

## 이 상태로 되돌리는 방법

### 방법 A — safe 브랜치로 전환 (코드만 복구)

```bash
git fetch origin
git checkout safe/cnu-chatbot-working-20260609
```

### 방법 B — main을 이 커밋으로 되돌리기

```bash
git checkout main
git reset --hard 50aa03b51bc612d63c387bb2ead1bee7d6671f3c
git push origin main --force
```

> ⚠️ 방법 B는 이후 커밋이 모두 사라집니다. 신중하게 사용하세요.

### 방법 C — 특정 파일만 복구

```bash
# 예: pipeline.py만 이 시점으로 복구
git checkout 50aa03b51bc612d63c387bb2ead1bee7d6671f3c -- src/rag/pipeline.py
```

---

## 절대 먼저 하면 안 되는 조치

| 조치 | 이유 |
|---|---|
| Drive 모델 캐시 삭제 | 재다운로드 5~30분, 세션 낭비 |
| tokenizers 버전 0.19.x 이하로 다운그레이드 | Qwen2.5 tokenizer.json 파싱 불가 |
| transformers 4.45.0~4.46.x 사용 | extra_special_tokens list→dict 버그 |
| chroma_db 삭제 후 chromadb 버전 변경 | 버전 불일치로 재구축 실패 |
| LoRA fine-tuning / Cross-Encoder reranker 추가 | 구조 복잡도 급증, 안정성 저하 |

---

## 주요 버전 고정 근거 요약

```
transformers==4.47.1  → 4.47.0에서 extra_special_tokens list→dict 버그 공식 수정
tokenizers==0.21.0    → Qwen2.5 tokenizer.json 포맷 호환 (0.19.x는 파싱 불가)
sentence-transformers==3.0.1 → 3.4+의 torchcodec import 충돌 방지
chromadb==0.5.5       → 현재 chroma_db와 동일 버전으로 구축됨
torch==2.5.1          → 과제 지정 버전
pytorch-lightning==2.4.0 → 과제 지정 버전
```
