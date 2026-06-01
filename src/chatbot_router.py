"""
CNU Campus ChatBot 라우터 — 질문 유형별 핸들러 분기
──────────────────────────────────────────────────
분기 로직:
  식단 키워드   → MealHandler   → (실패) 공식 URL
  셔틀 키워드   → ShuttleHandler → (실패) 공식 URL / known_data
  그 외         → RAGPipeline   (졸업요건/공지/학사일정)

chatbot_model.py 와 chatbot_ui.py 에서 공유 사용.
직접 실행 시 인터랙티브 테스트 가능.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.handlers.meal_handler    import MealHandler
from src.handlers.shuttle_handler import ShuttleHandler

# ── 키워드 세트 ───────────────────────────────────────────────────────────────
# 셔틀을 먼저 확인 (중의적 "버스" 키워드 처리)
_SHUTTLE_KW = frozenset({
    "셔틀", "셔틀버스", "통학버스", "스쿨버스",
    "버스시간", "버스 시간", "정류장", "운행시간",
    "첫차", "막차", "통학",
})

_MEAL_KW = frozenset({
    "학식", "식단", "메뉴", "밥", "점심",
    "저녁", "아침", "조식", "중식", "석식",
    "식당", "구내식당", "학생식당", "오늘밥",
    "뭐나와", "뭐 나와", "뭐 먹", "뭐먹",
})


def detect_category(question: str) -> int:
    """
    키워드 기반 카테고리 감지 (classifier 모델 불필요, 런타임 경량).

    Returns:
      3 — 식단 안내
      4 — 통학/셔틀버스
     -1 — RAG 처리 (졸업요건/공지/학사일정)
    """
    nq = question.replace(" ", "")
    # 셔틀 먼저: "버스" 단어는 shuttlekw 포함
    if any(k in nq for k in _SHUTTLE_KW):
        return 4
    if any(k in nq for k in _MEAL_KW):
        return 3
    return -1


class CNUChatRouter:
    """
    라우터.

    chat(question)    → (answer, source_tag)
    generate(question) → answer (source 없이)

    source_tag 값:
      "meal_handler"       — 식단 핸들러 (데이터 있음)
      "meal_official"      — 식단 핸들러 (데이터 없음, 공식 URL 안내)
      "shuttle_handler"    — 셔틀버스 핸들러 (크롤링/파일)
      "shuttle_known"      — 셔틀버스 핸들러 (known_data fallback)
      "shuttle_official"   — 셔틀버스 (모든 데이터 없음)
      "rag_pipeline"       — RAG 정상 응답
      "rag_threshold_miss" — RAG 임계값 미달 (관련 문서 없음)
    """

    def __init__(self, pipeline, base_dir: Path = BASE_DIR):
        self.pipeline        = pipeline
        self._meal    = MealHandler(base_dir)
        self._shuttle = ShuttleHandler(base_dir)

    def chat(self, question: str) -> tuple[str, str]:
        """질문 → (응답 텍스트, source_tag)"""
        cat = detect_category(question)

        if cat == 3:
            return self._meal.answer(question)

        if cat == 4:
            return self._shuttle.answer(question)

        # 졸업요건 / 공지사항 / 학사일정 → RAG
        answer = self.pipeline.generate(question)
        source = (
            "rag_threshold_miss"
            if "찾을 수 없습니다" in answer
            else "rag_pipeline"
        )
        return answer, source

    def generate(self, question: str) -> str:
        answer, _ = self.chat(question)
        return answer


# ── 인터랙티브 테스트 ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_qs = [
        "오늘 학식 뭐 나와요?",
        "다음주 학식 알려줘",
        "학생식당 메뉴 어디서 봐?",
        "셔틀버스 시간표 알려줘",
        "셔틀버스 정상 운행하나요?",
        "셔틀버스 정류장은 어디야?",
    ]

    print("─" * 60)
    print("[chatbot_router] 키워드 분기 테스트 (모델 로드 없이)")
    print("─" * 60)
    for q in test_qs:
        cat = detect_category(q)
        tag = {3: "meal_handler", 4: "shuttle_handler", -1: "rag_pipeline"}[cat]
        print(f"  Q: {q[:40]:<40}  → [{tag}]")

    print()
    print("[chatbot_router] 핸들러 응답 테스트 (모델 로드 없이)")
    print("─" * 60)

    meal_h    = MealHandler(BASE_DIR)
    shuttle_h = ShuttleHandler(BASE_DIR)

    for q in test_qs:
        cat = detect_category(q)
        if cat == 3:
            ans, src = meal_h.answer(q)
        elif cat == 4:
            ans, src = shuttle_h.answer(q)
        else:
            ans, src = "(RAG 필요 — 모델 미로드)", "rag_pipeline"

        print(f"\n  Q: {q}")
        print(f"  경로: [{src}]")
        print(f"  A: {ans[:120].replace(chr(10), ' | ')}{'...' if len(ans) > 120 else ''}")
