"""
REST API route definitions for status, completions, memory, and diagnostics.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import time


router = APIRouter(prefix="/v1")


class ChatCompletionRequest(BaseModel):
    message: str
    session_id: str = "default"
    max_tokens: int = 128
    temperature: float = 0.7


class ChatCompletionResponse(BaseModel):
    session_id: str
    reply: str
    latency_ms: float
    tokens_generated: int


@router.get("/healthz")
async def health_check():
    return {"status": "ok", "timestamp": time.time()}


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completion(req: ChatCompletionRequest):
    t0 = time.perf_counter()
    # Simulated execution harness
    return ChatCompletionResponse(
        session_id=req.session_id,
        reply="Cortex inference executed deterministically.",
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        tokens_generated=8
    )
