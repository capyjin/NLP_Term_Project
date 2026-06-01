"""
FastAPI 서버 — HTTP API 엔드포인트
────────────────────────────────────
엔드포인트:
  POST /ask    : 단건 질문 → 단건 답변
  POST /batch  : 질문 리스트 → 답변 리스트 (평가용)
  GET  /health : 서버·모델 상태 확인

사용:
  uvicorn src.api.server:app --host 0.0.0.0 --port 8000

평가 흐름:
  교수님 questions.json → POST /batch → answers 목록 반환
  또는 evaluate.py 직접 실행 (API 서버 없이 로컬 처리)

⚠️ startup 이벤트: 서버 시작 시 RAGPipeline 로드 (LLM 4-bit 로딩 약 1~3분)
   /health 엔드포인트로 모델 로드 완료 여부 확인 가능
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
    """서버 시작 시 RAGPipeline 로드 (HybridRetriever + Qwen2.5-3B 4-bit)."""
    global pipeline
    pipeline = RAGPipeline()


# ── 요청/응답 스키마 ──────────────────────────────────────────

class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    answer:   str


class BatchRequest(BaseModel):
    questions: list[str]


# ── 엔드포인트 ────────────────────────────────────────────────

@app.get("/health")
def health():
    """모델 로드 완료 여부 확인. model_loaded=False이면 startup 진행 중."""
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """단건 질문 → RAGPipeline.generate() → 답변 반환."""
    if pipeline is None:
        raise HTTPException(503, "모델 로딩 중입니다. /health 로 확인 후 재시도하세요.")
    answer = pipeline.generate(req.question)
    return AskResponse(question=req.question, answer=answer)


@app.post("/batch")
def batch(req: BatchRequest):
    """
    질문 리스트 일괄 처리 → 답변 리스트 반환.
    평가용: 교수님 제공 질문 목록을 한 번에 처리.
    """
    if pipeline is None:
        raise HTTPException(503, "모델 로딩 중입니다. /health 로 확인 후 재시도하세요.")
    results = []
    for q in req.questions:
        answer = pipeline.generate(q)
        results.append({"question": q, "answer": answer})
    return {"results": results}
