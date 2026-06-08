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

from src.handlers.meal_handler        import MealHandler
from src.handlers.shuttle_handler     import ShuttleHandler
from src.handlers.scholarship_handler import ScholarshipHandler

# ── 장학공지 리스트 키워드 ────────────────────────────────────────────────────────
# "리스트/목록/뭐있어/보여줘/공지/최근" 의도 → ScholarshipHandler
# 설명형("신청방법", "얼마") → RAG/FAQ 유지
_SCHOLARSHIP_LIST_KW = frozenset({
    "장학금리스트", "장학공지", "장학금목록", "장학금종류",
    "장학금뭐있어", "장학금있어", "장학있어", "장학금있나",
    "최근장학금", "장학금보여", "장학공지보여", "장학금신청가능",
    "장학금공지", "장학리스트",
})
# "장학" + 리스트 의도 트리거 — 짧은 질문에서 조합 감지
_SCHOLARSHIP_TRIGGER = frozenset({"리스트", "목록", "뭐있어", "뭐있나", "보여줘", "알려줘"})

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

# ── 졸업요건 직접 반환 키워드 ────────────────────────────────────────────────────
# Qwen 완전 우회: 해당 키워드 감지 시 사전 작성 답변 즉시 반환 (~1ms)
_GRAD_DIRECT_KW = frozenset({
    "졸업학점", "전공학점", "교양학점", "이수학점",
    "졸업요건", "졸업조건", "졸업인증", "조기졸업",
    "졸업기준", "학생편람",
})

# ── 수강신청/학사일정 직접 반환 키워드 ──────────────────────────────────────────
_COURSE_DIRECT_KW = frozenset({
    "수강신청", "수강정정", "수강변경",
    "정정기간", "수강신청기간",
    "계절학기", "하기계절", "동기계절",
    "개강", "종강",
    "중간고사", "기말고사", "시험기간",
    "성적공시", "성적발표", "성적이의",
    "등록금납부", "등록금기간",
})


def _has_shuttle(nq: str) -> bool:
    return any(k in nq for k in _SHUTTLE_KW) or any(k in nq for k in _SHUTTLE_BUS_KW)


def _has_meal(nq: str) -> bool:
    return any(k in nq for k in _MEAL_KW)


def _has_scholarship_list(nq: str) -> bool:
    """
    장학금 리스트/목록 의도 감지.
    "장학" 포함 + 리스트 트리거 OR 명시적 복합 키워드.
    설명형("신청방법", "얼마") 은 False → RAG/FAQ 유지.
    """
    # 명시적 복합 키워드 직접 매칭
    if any(k in nq for k in _SCHOLARSHIP_LIST_KW):
        return True
    # "장학" + 리스트 트리거 조합
    if "장학" in nq and any(t in nq for t in _SCHOLARSHIP_TRIGGER):
        # 설명형 제외: "신청방법", "얼마", "어떻게", "어디서"는 RAG로
        if not any(ex in nq for ex in ("신청방법", "얼마", "어떻게", "어디서", "어떤", "방법")):
            return True
    return False


def _has_grad_direct(nq: str) -> bool:
    """졸업요건 직접 반환 대상 여부. 'nq'는 공백 제거된 질문 문자열."""
    if any(k in nq for k in _GRAD_DIRECT_KW):
        return True
    # "졸업하려면 몇 학점 필요해?" 같은 복합 표현 포착
    if "졸업" in nq and "학점" in nq:
        return True
    return False


def _has_course_direct(nq: str) -> bool:
    """수강신청/학사일정 직접 반환 대상 여부."""
    return any(k in nq for k in _COURSE_DIRECT_KW)


def _build_grad_answer(question: str) -> str:
    """졸업요건 관련 직접 답변 — 키워드에 따라 세분화."""
    nq = question.replace(" ", "")

    if "졸업인증" in nq:
        return (
            "졸업인증 요건은 학과와 학번에 따라 다르며, 일반적으로 아래 항목들을 요구합니다.\n\n"
            "주요 졸업인증 항목 (학과·학번별 상이):\n"
            "  • 영어 능력 인증: TOEIC·TOEFL·OPIC 등 공인 점수 또는 교내 인증 시험\n"
            "  • 봉사활동 시간: 일정 시간 이상 사회봉사 이수\n"
            "  • SW/AI 인증: 소프트웨어 기초 교육 이수 (일부 학과 대상)\n"
            "  • 전공 역량 인증: 캡스톤 디자인, 포트폴리오 등 학과별 추가 요건\n\n"
            "확인 경로:\n"
            "  충남대학교 포털(plus.cnu.ac.kr) → 학사서비스 → 졸업요건 조회\n"
            "  또는 소속 학과 사무실·학과 홈페이지 교육과정 안내"
        )

    if "조기졸업" in nq:
        return (
            "조기졸업은 정규 수업 연한보다 일찍 졸업하는 제도로, 일정 요건을 갖춰야 합니다.\n\n"
            "일반적인 조기졸업 요건:\n"
            "  • 전체 졸업학점 이수 완료\n"
            "  • 성적 기준 충족 (학과·학칙에 따라 GPA 기준 다름)\n"
            "  • 졸업인증 요건 완료\n"
            "  • 지도교수 및 학과 승인\n\n"
            "신청 방법:\n"
            "  해당 학기 초 포털 학사서비스에서 신청 → 학과 심의 → 승인\n\n"
            "확인 경로:\n"
            "  충남대학교 포털(plus.cnu.ac.kr) → 학사서비스 → 졸업 관련\n"
            "  또는 학과 사무실 직접 문의"
        )

    if "전공학점" in nq:
        return (
            "전공 이수학점 기준은 학과와 입학연도에 따라 다릅니다.\n\n"
            "일반적인 전공학점 구성:\n"
            "  • 전공필수: 반드시 이수해야 하는 핵심 과목 (미이수 시 졸업 불가)\n"
            "  • 전공선택: 일정 학점 이상 자유 선택 이수\n"
            "  • 복수전공·부전공: 추가 이수학점 별도 요구\n\n"
            "정확한 기준 확인:\n"
            "  충남대학교 포털(plus.cnu.ac.kr) → 학사서비스 → 졸업요건 조회\n"
            "  (본인 학과·학번 기준 이수 현황 자동 조회 가능)\n"
            "  또는 소속 학과 홈페이지 교육과정표 참고"
        )

    if "교양학점" in nq:
        return (
            "교양 이수학점 기준은 입학연도와 학과에 따라 다릅니다.\n\n"
            "일반적인 교양학점 구성:\n"
            "  • 기초교양: 글쓰기·영어·수학 등 필수 기초 과목\n"
            "  • 균형교양: 인문·사회·자연과학·예술 등 영역별 균형 이수\n"
            "  • 자유교양: 자유롭게 선택하여 이수\n\n"
            "확인 경로:\n"
            "  충남대학교 포털(plus.cnu.ac.kr) → 학사서비스 → 졸업요건 조회\n"
            "  또는 교양교육원 홈페이지(ce.cnu.ac.kr) 교양교육과정 안내"
        )

    # 기본 졸업요건 종합 답변
    return (
        "충남대학교 졸업요건은 학과, 입학연도, 전공(단일/복수/부전공) 여부에 따라 달라집니다.\n\n"
        "일반적인 졸업 요건 구성:\n"
        "  • 총 이수학점: 130~140학점 내외 (학과별 상이)\n"
        "  • 전공 이수: 전공필수 전부 + 전공선택 일정 학점 이상\n"
        "  • 교양 이수: 기초교양·균형교양·자유교양 포함\n"
        "  • 졸업인증 요건: 영어·봉사활동·SW 인증 등 (학과·학번별 상이)\n"
        "  • 전공필수 과목 전부 이수\n\n"
        "정확한 내 기준은 포털에서 직접 확인하세요:\n"
        "  1. 충남대학교 포털(plus.cnu.ac.kr) → 학사서비스 → 졸업요건 조회\n"
        "     (본인 학과·학번 기준 이수 현황 자동 확인 가능)\n"
        "  2. 소속 학과 홈페이지 → 교육과정 또는 졸업요건 안내\n"
        "  3. 학과 사무실 또는 지도교수 상담"
    )


def _build_course_answer(question: str) -> str:
    """수강신청/학사일정 관련 직접 답변 — 키워드에 따라 세분화."""
    nq = question.replace(" ", "")

    if any(k in nq for k in ("계절학기", "하기계절", "동기계절")):
        return (
            "계절학기(하기·동기)는 정규 학기 종료 후 별도로 운영됩니다.\n\n"
            "계절학기 주요 안내:\n"
            "  • 하기계절학기: 1학기 종강 후 ~ 2학기 개강 전 (7~8월 중)\n"
            "  • 동기계절학기: 2학기 종강 후 ~ 다음 학기 개강 전 (1~2월 중)\n"
            "  • 최대 이수학점: 6학점 이내 (학칙에 따라 다를 수 있음)\n"
            "  • 수강료: 별도 납부 (학점당 요율 적용)\n"
            "  • 성적: 졸업학점에 포함, F 학점 취득 시 불이익 가능\n\n"
            "신청 방법:\n"
            "  충남대학교 포털(plus.cnu.ac.kr) → 학사서비스 → 수강신청 → 계절학기\n\n"
            "정확한 신청 기간은 포털 학사공지에서 반드시 확인하세요."
        )

    if any(k in nq for k in ("중간고사", "기말고사", "시험기간")):
        return (
            "시험 기간은 매 학기 학사일정에 따라 다르게 운영됩니다.\n\n"
            "일반적인 시험 일정:\n"
            "  • 중간고사: 학기 7~9주차 (강의 주수 기준, 학기별 상이)\n"
            "  • 기말고사: 학기 15~16주차 또는 종강 직전\n"
            "  • 시험 방식 및 범위: 각 교수자 재량 (강의계획서 확인 필수)\n\n"
            "확인 방법:\n"
            "  충남대학교 포털(plus.cnu.ac.kr) → 학사서비스 → 학사일정\n"
            "  또는 각 강의 강의계획서에서 해당 과목 시험 일정 확인"
        )

    if any(k in nq for k in ("성적공시", "성적발표", "성적이의")):
        return (
            "성적 공시 및 이의신청은 학기 종료 후 정해진 기간 내에만 가능합니다.\n\n"
            "주요 안내:\n"
            "  • 성적 입력: 기말고사 종료 후 담당 교수자가 포털에 입력\n"
            "  • 성적 공시: 포털 → 학사서비스 → 성적 조회에서 확인 가능\n"
            "  • 성적 이의신청: 공시 기간 내 담당 교수자에게 직접 문의\n"
            "  • 학점 취득 포기(W): 별도 신청 기간에 포털에서 신청 가능\n\n"
            "확인 경로:\n"
            "  충남대학교 포털(plus.cnu.ac.kr) → 학사서비스 → 성적 조회·이의신청"
        )

    if "등록금납부" in nq or ("등록금" in nq and "납부" in nq):
        return (
            "등록금 납부는 매 학기 초 포털 공지를 통해 기간과 방법이 안내됩니다.\n\n"
            "납부 방법:\n"
            "  1. 포털(plus.cnu.ac.kr) 로그인 → 학사서비스 → 등록금 납부\n"
            "  2. 본인 가상계좌 확인 후 은행 이체 또는 인터넷뱅킹으로 납부\n"
            "  3. 기한 내 미납 시 수강 취소 등 불이익 발생 가능\n\n"
            "장학금 수혜 시:\n"
            "  장학금 입금 확인 후 차액만 납부 (입금 전 납부 시 환불 절차 복잡)\n\n"
            "확인 경로:\n"
            "  충남대학교 포털(plus.cnu.ac.kr) → 학사서비스 → 등록금 납부"
        )

    if any(k in nq for k in ("개강", "종강")):
        return (
            "개강·종강 일정은 매 학년도 초에 학사일정으로 공지됩니다.\n\n"
            "일반적인 학기 일정 (연도별로 다를 수 있음):\n"
            "  • 1학기: 3월 초 개강 ~ 6월 말 종강\n"
            "  • 2학기: 9월 초 개강 ~ 12월 말 종강\n"
            "  • 방학: 1학기 종강 후 하계방학, 2학기 종강 후 동계방학\n\n"
            "정확한 날짜 확인:\n"
            "  충남대학교 포털(plus.cnu.ac.kr) → 학사서비스 → 학사일정\n"
            "  (연도별 정확한 날짜는 반드시 포털에서 확인하세요)"
        )

    if any(k in nq for k in ("수강정정", "수강변경", "정정기간")):
        return (
            "수강정정(수강변경)은 개강 직후 일정 기간 동안 진행됩니다.\n\n"
            "주요 안내:\n"
            "  • 수강정정 기간: 일반적으로 개강 후 1~2주 이내\n"
            "  • 가능한 작업: 수강 추가·삭제·변경 (인원 여유 있는 강좌만 가능)\n"
            "  • 재수강 신청: 별도 기간 진행 (성적 C+ 이하 과목 대상)\n"
            "  • 수강 포기(W): 정정 기간 이후 별도 신청 기간 있음\n\n"
            "확인 방법:\n"
            "  충남대학교 포털(plus.cnu.ac.kr) → 학사서비스 → 학사공지 → 수강신청 관련\n"
            "  (정확한 정정 기간은 해당 학기 학사공지에서 반드시 확인)"
        )

    # 기본 수강신청 종합 답변
    return (
        "수강신청 및 관련 학사일정은 학기마다 다르게 운영됩니다.\n\n"
        "수강신청 주요 안내:\n"
        "  • 예비 수강신청: 학기 시작 2~3주 전 (희망 과목 선점)\n"
        "  • 본 수강신청: 학기 시작 1~2주 전 (확정)\n"
        "  • 수강정정: 개강 직후 1~2주 이내\n"
        "  • 수강 포기(W): 별도 기간 신청 가능\n"
        "  • 계절학기: 정규학기 종료 후 별도 운영\n\n"
        "신청 방법:\n"
        "  충남대학교 포털(plus.cnu.ac.kr) → 학사서비스 → 수강신청\n\n"
        "정확한 일정은 해당 학기 학사공지에서 반드시 확인하세요.\n"
        "  확인 경로: 포털 → 공지사항 → 학사공지"
    )


def detect_category(question: str) -> int:
    """
    키워드 기반 단일 카테고리 감지.
    복합 질문은 detect_all_categories() 사용.

    Returns: 3(식단) | 4(셔틀) | 5(장학리스트) | -1(RAG)
    """
    nq = question.replace(" ", "")
    if _has_shuttle(nq):
        return 4
    if _has_meal(nq):
        return 3
    if _has_scholarship_list(nq):
        return 5
    return -1


def detect_all_categories(question: str) -> list[int]:
    """
    복합 질문 감지 — 감지된 카테고리 전체 반환.
    "오늘 학식이랑 셔틀버스 시간 알려줘" → [4, 3]
    단일 질문은 기존 detect_category()와 동일하게 동작.
    """
    nq = question.replace(" ", "")
    has_s  = _has_shuttle(nq)
    has_m  = _has_meal(nq)
    has_sc = _has_scholarship_list(nq)

    cats = []
    if has_s:  cats.append(4)
    if has_m:  cats.append(3)
    if has_sc: cats.append(5)
    return cats if cats else [-1]


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
      "direct_handler"     — 졸업요건/수강신청 직접 반환 (Qwen 완전 우회, ~1ms)
    """

    def __init__(self, pipeline, base_dir: Path = BASE_DIR):
        self.pipeline     = pipeline
        self._meal        = MealHandler(base_dir)
        self._shuttle     = ShuttleHandler(base_dir)
        self._scholarship = ScholarshipHandler(base_dir)

    def chat(self, question: str) -> tuple[str, str]:
        """질문 → (응답 텍스트, source_tag)"""
        cats = detect_all_categories(question)

        # ── 복합 질문: 식단 + 셔틀 ──────────────────────────────────────
        if 4 in cats and 3 in cats:
            shuttle_ans, _ = self._shuttle.answer(question)
            meal_ans,    _ = self._meal.answer(question)
            combined = f"[셔틀버스 안내]\n{shuttle_ans}\n\n[식단 안내]\n{meal_ans}"
            return combined, "multi_handler"

        # ── 단일 카테고리 ────────────────────────────────────────────────
        if cats == [3]:
            return self._meal.answer(question)

        if cats == [4]:
            return self._shuttle.answer(question)

        if cats == [5]:
            return self._scholarship.answer(question)

        # ── 졸업요건/수강신청 직접 반환 (Qwen 완전 우회, ~1ms) ─────────────
        nq = question.replace(" ", "")
        if _has_grad_direct(nq):
            return _build_grad_answer(question), "direct_handler"
        if _has_course_direct(nq):
            return _build_course_answer(question), "direct_handler"

        # ── RAG (공지/장학설명형/기타) ──────────────────────────────────
        answer = self.pipeline.generate(question)
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
