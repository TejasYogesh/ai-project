"""Health check and configuration info."""
from fastapi import APIRouter

from backend.core.config import settings

router = APIRouter(tags=["system"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/system/info")
def system_info() -> dict:
    return {
        "llm_model": settings.llm_model,
        "embed_model": settings.embed_model,
        "similarity_cutoff": settings.similarity_cutoff,
        "retrieval_top_k": settings.retrieval_top_k,
        "notes": [
            "Gemini calls retry on 429/503 with exponential backoff.",
            "First briefing of a paper takes ~30-60s (download, embed, summarize); "
            "repeat requests for the same arXiv ID reuse the saved briefing.",
        ],
    }