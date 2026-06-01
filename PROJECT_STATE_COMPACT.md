# PROJECT_STATE_COMPACT — CNU Campus ChatBot
**박연진 | 자연어처리 텀프로젝트 | 2026-06-01**

---

## 📌 핵심 현황

| 항목 | 상태 | 비고 |
|---|:---:|---|
| 코드 제출 기한 | D-11 | 2026-06-12(금) 자정 |
| 텀프 발표 | D-15 | 2026-06-16(화) |
| Task 1 분류기 | ✅ 완료 | classifier_박연진.ipynb, klue/bert-base |
| Task 2 챗봇 RAG | ✅ 완료 | Qwen2.5-3B + BM25+KURE+RRF |
| Task 2 UI | ✅ 완료 | Gradio ChatInterface |
| Optional 식단 실시간 | ✅ 완료 | meal_crawler.py → MealHandler |
| Optional 셔틀버스 | ✅ 완료 | ShuttleHandler + known_data fallback |

---

## 🗂 핵심 파일 구조

```
chatbot.sh / chatbot.bat          ← 조교 실행 진입점
classifier_박연진.ipynb            ← Task 1 (Colab T4)
src/
  chatbot_model.py                ← test_chat.json → chat_output.json
  chatbot_ui.py                   ← Gradio UI (port 7860)
  realtime_model.py               ← test_realtime.json → realtime_output.json
  chatbot_router.py               ← CNUChatRouter + detect_category()
  rag/pipeline.py                 ← RAGPipeline (Qwen2.5-3B NF4)
  rag/retriever.py                ← BM25(Kiwi) + KURE-v1 + RRF + boost
  handlers/meal_handler.py        ← 식단 응답 (meal_menu.json)
  handlers/shuttle_handler.py     ← 셔틀버스 응답 (shuttle_bus.json)
  crawling/meal_crawler.py        ← 식단 크롤러 (mobileadmin.cnu.ac.kr)
data/processed/chunks.json        ← 181청크 (크롤링 158 + FAQ 23)
```

---

## 🔀 라우팅 흐름

```
질문 입력
  ↓ detect_category(q)
  ├─ cat=3 (식단)    → MealHandler.answer(q)      [Qwen 미사용]
  ├─ cat=4 (셔틀)    → ShuttleHandler.answer(q)   [Qwen 미사용]
  └─ cat=-1 (RAG)   → CNUChatRouter.chat(q)
                         ↓ HybridRetriever.search()
                           BM25(Kiwi형태소) + KURE-v1 ChromaDB
                           → RRF + subcategory boost/penalty
                         ↓ best_embed_score ≥ 0.40 ?
                           YES → Qwen2.5-3B-Instruct 생성
                           NO  → "찾을 수 없습니다" (할루시네이션 방지)
```

---

## ✅ RAG 검증 결과 (2026-06-01, BM25 로컬 검증)

| 질문 | BM25 Top-1 청크 | Top-1 subcategory | 판정 |
|---|---|:---:|:---:|
| 수강신청 변경 기간은 언제야? | 수강신청 방법 (BM25=9.99) | 수강신청 | ✅ 적절 |
| 졸업하려면 몇 학점 필요해? | 졸업학점 및 졸업 요건 (BM25=6.03) | 졸업요건 | ✅ 적절 |
| 장학금 신청은 어디서 해? | 장학금 신청은 어디서 하나요? (BM25=15.88) | 장학FAQ | ✅ 완벽 |

> **참고**: 로컬 검증은 공백 토크나이저 사용. Colab 실행 시 Kiwi 형태소 분석 + subcategory boost 적용으로 정확도 추가 향상.
> "졸업" 쿼리: Kiwi 환경에서 GRAD_TRIGGERS boost(2.5x)로 졸업요건 FAQ가 Top-1으로 상승 예상.

---

## 📝 chatbot.sh 실행 흐름

```bash
# 1. 식단 크롤링 (실패해도 계속)
python src/crawling/meal_crawler.py    || echo "⚠ 식단 크롤링 실패 — 수동 파일 사용"
python src/crawling/shuttle_crawler.py || echo "⚠ 셔틀버스 크롤링 실패 — known_data 사용"

# 2. chroma_db 없으면 자동 구축
if [ ! -d "chroma_db" ]; then python src/vectordb/build_db.py; fi

# 3. 챗봇 응답 생성
python src/chatbot_model.py   # → outputs/chat_output.json

# 4. 실시간 정보 (선택)
if [ -f "data/test_realtime.json" ]; then
  python src/realtime_model.py
fi

# 5. UI 실행
python src/chatbot_ui.py
```

---

## ⚠️ Colab 실행 전 체크리스트

- [ ] classifier_박연진.ipynb 셀 2: PROJECT 경로 수정 (Drive 폴더명 확인)
- [ ] chatbot.sh: 실제 T4 GPU에서 전체 실행 테스트
- [ ] chroma_db 재구축: `python scripts/inject_faq.py && python src/vectordb/build_db.py`
- [ ] outputs/ 폴더: cls_output.json, chat_output.json 생성 확인
- [ ] 조교 파일 교체: test_cls.json, test_chat.json, test_realtime.json

---

## 🔴 남은 필수 작업

| 우선순위 | 작업 | 예상시간 |
|:---:|---|:---:|
| 1 | Colab classifier 학습 + cls_output.json 확인 | 1h |
| 2 | Colab chatbot.sh 전체 실행 + chat_output.json 확인 | 1h |
| 3 | chroma_db FAQ 포함 재구축 | 30min |
| 4 | 발표 자료 작성 | 2h |
| 5 | 최종 zip 압축 + 사이버캠퍼스 제출 | 30min |

---

## 📊 예상 점수

| Task | 만점 | 예상 |
|---|:---:|:---:|
| Task 1 분류기 F1 | 40 | 32~38 |
| Task 2 UI + ChatInterface | 20 | 20 |
| Task 2 응답 맥락 | 40 | 30~36 |
| Optional 실시간 | 30 | 20~25 |
| **합계** | **130** | **102~119** |
