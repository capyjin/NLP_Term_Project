"""
Gradio 웹 UI — 충남대 Q&A 챗봇 인터페이스
──────────────────────────────────────────
실행:
  python src/ui/app.py           # 로컬 (http://localhost:7860)
  Colab 노트북 Cell 9에서 실행   # share=True → 공개 URL 자동 생성 (72시간)

UI 구조:
  - Chatbot: 대화 이력 표시 (height=450px)
  - Textbox: 질문 입력 (Enter 또는 전송 버튼)
  - btn.click / txt.submit: answer() 호출

answer() 흐름:
  질문 입력 → pipeline.generate() → 응답 → chatbot history 업데이트
"""

import gradio as gr
from src.rag.pipeline import RAGPipeline

# RAGPipeline 초기화: CNUVectorStore + HybridRetriever + Qwen2.5-3B 로드
pipeline = RAGPipeline()


def answer(question: str, history: list):
    """질문을 RAGPipeline에 전달하고 응답을 chatbot history에 추가."""
    if not question.strip():
        return "", history
    response = pipeline.generate(question)
    history.append((question, response))
    return "", history


with gr.Blocks(title="충남대 Q&A 챗봇") as demo:
    gr.Markdown("## 충남대학교 학생 Q&A 챗봇")
    chatbot = gr.Chatbot(height=450)
    with gr.Row():
        txt = gr.Textbox(placeholder="질문을 입력하세요...", scale=9, show_label=False)
        btn = gr.Button("전송", scale=1)
    # Enter 키와 전송 버튼 모두 answer() 트리거
    btn.click(answer, [txt, chatbot], [txt, chatbot])
    txt.submit(answer, [txt, chatbot], [txt, chatbot])

if __name__ == "__main__":
    demo.launch(share=True)   # share=True: Gradio 공개 URL 생성 (Colab 필수)
