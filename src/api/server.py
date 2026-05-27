"""
FastAPI 서버 — 교수님/조교가 질문 리스트를 보내면 답변 일괄 반환
POST /ask         : 단건 질문
POST /batch       : 질문 리스트 파일(JSON) → 답변 리스트 반환
GET  /health      : 서버 상태 확인
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import json

from src.rag.pipeline import RAGPipeline

app = FastAPI(title="CNU QA API")
pipeline: Optional[RAGPipeline] = None


@app.on_event("startup")
async def startup():
    global pipeline
    pipeline = RAGPipeline()


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    answer: str


class BatchRequest(BaseModel):
    questions: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if pipeline is None:
        raise HTTPException(503, "모델 로딩 중입니다.")
    answer = pipeline.generate(req.question)
    return AskResponse(question=req.question, answer=answer)


@app.post("/batch")
def batch(req: BatchRequest):
    if pipeline is None:
        raise HTTPException(503, "모델 로딩 중입니다.")
    results = []
    for q in req.questions:
        answer = pipeline.generate(q)
        results.append({"question": q, "answer": answer})
    return {"results": results}
