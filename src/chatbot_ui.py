"""
충남대 Campus ChatBot — Gradio 웹 UI
──────────────────────────────────────
실행:
  python src/chatbot_ui.py           # 로컬 (http://localhost:7860)
  chatbot.sh 에서 자동 호출          # share=True → 공개 URL (Colab)

UI 구성:
  - gr.ChatInterface: 대화형 채팅 인터페이스
  - 질문 입력 → RAGPipeline.generate() → 응답 표시
  - 예시 질문 5개 제공
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

import gradio as gr
from src.rag.pipeline   import RAGPipeline
from src.chatbot_router import CNUChatRouter

# 라우터 초기화 (RAGPipeline + MealHandler + ShuttleHandler)
print("Campus ChatBot 로딩 중...")
_pipeline = RAGPipeline()
_router   = CNUChatRouter(_pipeline, BASE_DIR)
print("Campus ChatBot 로드 완료")


def chat(message: str, history: list) -> str:
    """
    사용자 메시지 → 라우터 분기 → 응답.
    식단/셔틀 질문: 전용 핸들러
    그 외: RAGPipeline
    """
    if not message.strip():
        return "질문을 입력해주세요."
    return _router.generate(message)


demo = gr.ChatInterface(
    fn=chat,
    title="충남대학교 Campus ChatBot",
    description=(
        "충남대학교 재학생을 위한 AI 챗봇입니다.\n"
        "장학금, 수강신청, 졸업요건, 식단, 셔틀버스 등 학교 관련 질문을 입력하세요."
    ),
    examples=[
        "오늘 학식 뭐 나와요?",
        "셔틀버스 시간표 알려줘",
        "장학금 신청은 어떻게 하나요?",
        "수강신청 정정 기간이 언제예요?",
        "졸업하려면 몇 학점이 필요해요?",
    ],
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    # share=True: Gradio 공개 URL (Colab 환경 필수)
    # server_name="0.0.0.0": 외부 접근 허용
    demo.launch(share=True, server_name="0.0.0.0")
