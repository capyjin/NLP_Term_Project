"""
CNU Campus ChatBot 라우터 — 질문 유형별 핸들러 분기
──────────────────────────────────────────────────
분기 로직:
  식단 키워드   → MealHandler
  셔틀 키워드   → ShuttleHandler
  둘 다 감지   → 각각 호출 후 합산 (복합 질문)
  그 외         → RAGPipeline (졸업요건/공지/학사일정)

chatbot_model.py 와 chatbot_ui.py 에서 공유 사용.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.handlers.meal_handler    import MealHandler
from src.handlers.shuttle_handler import ShuttleHandler

# ── 셔틀버스 키워드 ────────────────────────────────────────────────────────────
_SHUTTLE_KW = frozenset({
    "셔틀", "셔틀버스", "셧틀", "셔틀뻐스",          # 오타 포함
    "통학버스", "스쿨버스", "통학",
    "버스시간", "버스 시간", "정류장", "운행시간",
    "첫차", "막차",
    "버스운행", "버스취소", "버스정상",
    "버스있어", "버스있나", "버스몇시", "버스언제",
    "캠퍼스순환", "교내순환", "대덕", "보운",          # 노선명
})

# "버스" 단독 → 충남대 맥락에서 셔틀버스로 간주
_SHUTTLE_BUS_KW = frozenset({"버스"})

# ── 식단 키워드 ────────────────────────────────────────────────────────────────
_MEAL_KW = frozenset({
    "학식", "식단", "메뉴", "밥", "점심",
    "저녁", "아침", "조식", "중식", "석식",
    "식당", "구내식당", "학생식당", "오늘밥",
    "뭐나와", "뭐 나와", "뭐 먹", "뭐먹",
    "나오나요", "나오나", "나왔나", "나왔어",
    "뭐나왔", "뭐나오", "학식메뉴",
})


def _has_shuttle(nq: str) -> bool:
    return any(k in nq for k in _SHUTTLE_KW) or any(k in nq for k in _SHUTTLE_BUS_KW)


def _has_meal(nq: str) -> bool:
    return any(k in nq for k in _MEAL_KW)


def detect_category(question: str) -> int:
    """
    키워드 기반 단일 카테고리 감지.
    복합 질문은 detect_all_categories() 사용.

    Returns: 3(식단) | 4(셔틀) | -1(RAG)
    """
    nq = question.replace(" ", "")
    if _has_shuttle(nq):
        return 4
    if _has_meal(nq):
        return 3
    return -1


def detect_all_categories(question: str) -> list[int]:
    """
    복합 질문 감지 — 감지된 카테고리 전체 반환.
    "오늘 학식이랑 셔틀버스 시간 알려줘" → [4, 3]
    단일 질문은 기존 detect_category()와 동일하게 동작.
    """
    nq = question.replace(" ", "")
    has_s = _has_shuttle(nq)
    has_m = _has_meal(nq)

    if has_s and has_m:
        return [4, 3]   # 셔틀 먼저 표시
    if has_s:
        return [4]
    if has_m:
        return [3]
    return [-1]


class CNUChatRouter:
    """
    라우터.

    chat(question)     → (answer, source_tag)
    generate(question) → answer (source 없이)

    source_tag 값:
      "meal_handler"       — 식단 핸들러
      "meal_official"      — 식단 핸들러 (fallback)
      "shuttle_handler"    — 셔틀버스 핸들러
      "shuttle_known"      — 셔틀버스 핸들러 (known_data)
      "multi_handler"      — 복합 질문 (식단+셔틀)
      "rag_pipeline"       — RAG 정상 응답
      "rag_threshold_miss" — RAG 임계값 미달 (카테고리별 안내 반환)
    """

    def __init__(self, pipeline, base_dir: Path = BASE_DIR):
        self.pipeline = pipeline
        self._meal    = MealHandler(base_dir)
        self._shuttle = ShuttleHandler(base_dir)

    def chat(self, question: str) -> tuple[str, str]:
        """질문 → (응답 텍스트, source_tag)"""
        cats = detect_all_categories(question)

        # ── 복합 질문: 식단 + 셔틀 ──────────────────────────────────────
        if cats == [4, 3]:
            shuttle_ans, _ = self._shuttle.answer(question)
            meal_ans,    _ = self._meal.answer(question)
            combined = f"[셔틀버스 안내]\n{shuttle_ans}\n\n[식단 안내]\n{meal_ans}"
            return combined, "multi_handler"

        # ── 단일 카테고리 ────────────────────────────────────────────────
        if cats == [3]:
            return self._meal.answer(question)

        if cats == [4]:
            return self._shuttle.answer(question)

        # ── RAG (졸업요건/공지/학사일정) ─────────────────────────────────
        answer = self.pipeline.generate(question)
        # 카테고리별 fallback 메시지는 "확인 경로:" 포함 → threshold miss 판별
        source = (
            "rag_threshold_miss"
            if "확인 경로:" in answer or "관련 정보를 저장된 자료에서 찾지 못했습니다" in answer
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
        "셔틀버스 시간표 알려줘",
        "오늘 학식이랑 셔틀버스 시간 알려줘",   # 복합
        "버스 몇 시에 있어?",
        "뭐 나오나요?",
        "졸업학점 몇 점이에요?",
        "장학금 신청 어디서 해?",
    ]

    print("-" * 60)
    print("[chatbot_router] 라우팅 테스트 (모델 로드 없이)")
    print("-" * 60)

    meal_h    = MealHandler(BASE_DIR)
    shuttle_h = ShuttleHandler(BASE_DIR)

    for q in test_qs:
        cats = detect_all_categories(q)
        if cats == [4, 3]:
            tag = "multi(shuttle+meal)"
        elif cats == [3]:
            tag = "meal_handler"
        elif cats == [4]:
            tag = "shuttle_handler"
        else:
            tag = "rag_pipeline"
        print(f"  [{tag}] {q}")
