"""
Gradio 웹 UI — Colab에서 share=True 로 ngrok 없이 공개 URL 생성 가능
"""

import gradio as gr
from src.rag.pipeline import RAGPipeline

pipeline = RAGPipeline()


def answer(question: str, history: list):
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
    btn.click(answer, [txt, chatbot], [txt, chatbot])
    txt.submit(answer, [txt, chatbot], [txt, chatbot])

if __name__ == "__main__":
    demo.launch(share=True)
