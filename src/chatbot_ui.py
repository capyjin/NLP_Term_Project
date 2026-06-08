"""
충남대 Campus ChatBot — Gradio 웹 UI
──────────────────────────────────────
실행:
  python src/chatbot_ui.py           # 로컬 (http://localhost:7860)
  chatbot.sh 에서 자동 호출          # share=True → 공개 URL (Colab)

UI 구성:
  - gr.ChatInterface: 대화형 채팅 인터페이스
  - 질문 입력 → CNUChatRouter → 응답 + 처리 경로(source) 표시
  - 예시 질문 6개 제공 (식단/셔틀/장학/학사/졸업/수강)
  - source 뱃지: 응답 하단에 처리 경로 표시 (평가자 확인용)
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

import gradio as gr
from src.rag.pipeline   import RAGPipeline
from src.chatbot_router import CNUChatRouter

# ── source 뱃지 매핑 ──────────────────────────────────────────────────────────
_SOURCE_LABELS = {
    "meal_handler":        "🍱 식단 DB",
    "meal_official":       "🍱 식단 DB (공식)",
    "shuttle_handler":     "🚌 셔틀 DB",
    "shuttle_known":       "🚌 셔틀 DB (시간표)",
    "multi_handler":       "🍱🚌 식단+셔틀",
    "scholarship_handler": "🎓 장학공지 DB",
    "scholarship_no_data": "🎓 장학공지 (데이터 없음)",
    "rag_pipeline":        "🔍 RAG 검색",
    "rag_threshold_miss":  "💡 일반 안내",
    "direct_handler":      "⚡ 직접 안내",
    "notice_handler":      "📋 공지사항 DB",
}


# ── 초기화 ────────────────────────────────────────────────────────────────────
print("=" * 50)
print("  충남대학교 Campus ChatBot 로딩 중...")
print("=" * 50)
_pipeline = RAGPipeline()
_router   = CNUChatRouter(_pipeline, BASE_DIR)
print("  ✓ 로드 완료\n")


def chat(message: str, history: list) -> str:
    """
    사용자 메시지 → CNUChatRouter 분기 → 응답 + source 뱃지.

    처리 경로:
      식단/셔틀 질문  → 전용 핸들러 (RAGPipeline 호출 없음)
      장학 리스트     → ScholarshipHandler
      학사/졸업/공지  → RAGPipeline (BM25 + Dense + RRF)
    """
    if not message.strip():
        return "질문을 입력해주세요."

    answer, source = _router.chat(message)
    badge = _SOURCE_LABELS.get(source, source)

    # 응답 하단에 처리 경로 뱃지 표시 (Markdown 소문자 텍스트)
    return f"{answer}\n\n---\n*처리 경로: {badge}*"


# ── Gradio UI ─────────────────────────────────────────────────────────────────
demo = gr.ChatInterface(
    fn=chat,
    title="🎓 충남대학교 Campus ChatBot",
    description=(
        "충남대학교 재학생을 위한 AI 챗봇입니다.\n"
        "졸업요건 · 학사일정 · 장학금 · 식단 · 셔틀버스 관련 질문을 입력하세요.\n"
        "아래 예시 버튼을 클릭하면 바로 질문할 수 있습니다."
    ),
    examples=[
        "오늘 학식 뭐 나와요?",
        "이번주 식단 알려줘",
        "셔틀버스 시간표 알려줘",
        "최근 장학금 공지 보여줘",
        "졸업학점 몇 점이에요?",
        "수강신청 정정기간 언제예요?",
    ],
    theme=gr.themes.Soft(),
    # Gradio 4.x: show_copy_button 기본 활성화
)

if __name__ == "__main__":
    # share=True: Gradio 공개 URL (Colab 환경 필수)
    # server_name="0.0.0.0": 외부 접근 허용
    demo.launch(share=True, server_name="0.0.0.0")
