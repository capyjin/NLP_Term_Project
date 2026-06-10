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
    "notice_handler":              "📋 공지사항 DB",
    "academic_calendar_handler":   "📅 학사일정 DB",
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


# ── 마스코트 이미지 base64 인코딩 ───────────────────────────────────────────────
import base64 as _b64

_mascot_path = BASE_DIR / "src" / "ui" / "chacha.png"
_MASCOT_HTML = ""
if _mascot_path.exists():
    with open(_mascot_path, "rb") as _f:
        _b64_str = _b64.b64encode(_f.read()).decode()
    _MASCOT_HTML = (
        '<div id="cnu-mascot">'
        f'<img src="data:image/png;base64,{_b64_str}" alt="차차" />'
        '</div>'
    )


# ── CSS ───────────────────────────────────────────────────────────────────────
_CSS = """
/* ── 전체 배경 ── */
body, .gradio-container {
    background: #f2f5fa !important;
    font-family: 'Noto Sans KR', 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif !important;
}

/* ── 헤더 카드 ── */
.cnu-header {
    background: linear-gradient(135deg, #002d72 0%, #0047b3 100%);
    color: #ffffff;
    padding: 26px 32px 20px;
    border-radius: 14px;
    margin-bottom: 14px;
    box-shadow: 0 4px 20px rgba(0, 45, 114, 0.20);
}
.cnu-header h1 {
    margin: 0 0 6px 0;
    font-size: 1.80rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    line-height: 1.25;
}
.cnu-header p {
    margin: 0;
    font-size: 0.91rem;
    opacity: 0.85;
    line-height: 1.55;
}

/* ── 채팅 영역 감싸는 카드 ── */
.chat-card {
    background: #ffffff;
    border-radius: 14px;
    box-shadow: 0 2px 16px rgba(0, 0, 0, 0.07);
    overflow: hidden;
    padding: 4px;
}

/* ── Chatbot 메시지 창 높이 ── */
#cnu-chatbot {
    height: 530px !important;
    min-height: 530px !important;
}

/* ── 사용자 메시지 말풍선 ── */
.message.user > div,
div[data-testid="user"] .prose {
    background: #002d72 !important;
    color: #ffffff !important;
    border-radius: 18px 18px 4px 18px !important;
}

/* ── 봇 메시지 말풍선 ── */
.message.bot > div,
div[data-testid="bot"] .prose {
    background: #eef3fb !important;
    color: #1a1f36 !important;
    border-radius: 18px 18px 18px 4px !important;
}

/* ── 입력창 ── */
#cnu-input textarea {
    border: 1.5px solid #c2cfe0 !important;
    border-radius: 10px !important;
    font-size: 0.94rem !important;
    padding: 10px 14px !important;
    background: #fafcff !important;
    transition: border-color 0.2s, box-shadow 0.2s;
    resize: none !important;
}
#cnu-input textarea:focus {
    border-color: #002d72 !important;
    box-shadow: 0 0 0 3px rgba(0, 45, 114, 0.09) !important;
    outline: none !important;
}

/* ── 전송 버튼 ── */
#cnu-submit {
    background: #002d72 !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 0.94rem !important;
    transition: background 0.18s, transform 0.12s;
}
#cnu-submit:hover {
    background: #0047b3 !important;
    transform: translateY(-1px);
}
#cnu-submit:active {
    transform: translateY(0px);
}

/* ── 예시 질문 버튼 ── */
.gr-button-secondary,
button.secondary {
    background: #ffffff !important;
    color: #002d72 !important;
    border: 1.5px solid #b8cce8 !important;
    border-radius: 20px !important;
    padding: 7px 15px !important;
    font-size: 0.87rem !important;
    font-weight: 500 !important;
    box-shadow: 0 2px 7px rgba(0, 45, 114, 0.09) !important;
    transition: all 0.18s ease !important;
}
.gr-button-secondary:hover,
button.secondary:hover {
    background: #eef3fb !important;
    border-color: #002d72 !important;
    box-shadow: 0 5px 14px rgba(0, 45, 114, 0.15) !important;
    transform: translateY(-2px) !important;
}

/* ── 마스코트 고정 오버레이 ── */
#cnu-mascot {
    position: fixed;
    bottom: 30px;
    right: 30px;
    z-index: 9999;
    pointer-events: none;
    filter: drop-shadow(0 6px 14px rgba(0, 0, 0, 0.16));
    animation: mascot-float 3.2s ease-in-out infinite;
}
#cnu-mascot img {
    width: 108px;
    height: auto;
    display: block;
}
@keyframes mascot-float {
    0%   { transform: translateY(0px);  }
    50%  { transform: translateY(-8px); }
    100% { transform: translateY(0px);  }
}

/* ── 푸터 ── */
.cnu-footer {
    text-align: center;
    color: #8a9ab8;
    font-size: 0.76rem;
    padding: 10px 0 2px;
    letter-spacing: 0.2px;
}

/* ── 반응형 (모바일) ── */
@media (max-width: 640px) {
    .cnu-header h1  { font-size: 1.30rem; }
    #cnu-chatbot    { height: 380px !important; min-height: 380px !important; }
    #cnu-mascot img { width: 78px; }
}
"""

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(css=_CSS, theme=gr.themes.Base(), title="CNU Campus ChatBot") as demo:

    # 마스코트 오버레이 (base64 인라인 이미지)
    if _MASCOT_HTML:
        gr.HTML(_MASCOT_HTML)

    # 헤더
    gr.HTML("""
    <div class="cnu-header">
      <h1 style="color:#ffffff;">🎓 CNU Campus ChatBot</h1>
      <p>충남대학교 학사정보 · 공지사항 · 식단 · 셔틀버스 안내 AI 챗봇</p>
    </div>
    """)

    # 채팅 인터페이스 (fn=chat 및 내부 로직 변경 없음)
    with gr.Column(elem_classes=["chat-card"]):
        gr.ChatInterface(
            fn=chat,
            chatbot=gr.Chatbot(
                elem_id="cnu-chatbot",
                show_label=False,
                bubble_full_width=False,
            ),
            textbox=gr.Textbox(
                elem_id="cnu-input",
                placeholder="질문을 입력하세요  예) 오늘 학식 뭐 나와요?",
                show_label=False,
                lines=1,
                max_lines=4,
            ),
            submit_btn="전송 ▶",
            examples=[
                "🎓  졸업 학점이 몇 점인가요?",
                "📢  최근 공지사항 알려줘",
                "📅  이번 학기 수강신청 일정 알려줘",
                "🍱  오늘 학식 뭐 나와요?",
                "🚌  셔틀버스 시간표 알려줘",
                "💰  장학금 신청은 어디서 해?",
            ],
            cache_examples=False,
        )

    # 푸터
    gr.HTML('<div class="cnu-footer">Chungnam National University · AI ChatBot · 자연어처리 텀프로젝트</div>')


if __name__ == "__main__":
    # share=True: Gradio 공개 URL (Colab 환경 필수)
    # server_name="0.0.0.0": 외부 접근 허용
    demo.launch(share=True, server_name="0.0.0.0")
