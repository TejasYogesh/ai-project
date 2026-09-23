"""FastAPI app: HTTP layer over the paper agent, topic explorer and QA agent."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from backend.api import conversations, explore, papers, reports, system
from backend.core.exceptions import (AgentError, ArxivAPIError, BriefingError, ConversationNotFound,
                                     PaperNotFound, PaperNotIndexed, PDFDownloadError, PDFParseError,
                                     PipelineError, QAError, ReportNotFound)
from backend.core.llm import init_llm
from backend.models import database as db

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
for noisy in ("httpx", "chromadb"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.ERROR)

# Which HTTP status each project error becomes. More specific classes first.
STATUS_BY_ERROR: dict[type[AgentError], int] = {
    PaperNotFound: 404,
    ReportNotFound: 404,
    ConversationNotFound: 404,
    PaperNotIndexed: 404,
    PipelineError: 422,          # the graph could not produce a briefing (e.g. no papers for a topic)
    PDFParseError: 422,
    ArxivAPIError: 502,          # an upstream service failed
    PDFDownloadError: 502,
    BriefingError: 502,
    QAError: 502,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once at startup: configure Gemini and create database tables."""
    init_llm()
    db.init_db()
    yield


app = FastAPI(
    title="arXiv Paper Digest & QA Agent",
    description="Brief an arXiv paper from an ID or a topic, then ask grounded questions about it.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(AgentError)
async def agent_error_handler(request: Request, exc: AgentError) -> JSONResponse:
    """Turn any project error into a JSON response with a sensible status code."""
    status = next((code for cls, code in STATUS_BY_ERROR.items() if isinstance(exc, cls)), 500)
    return JSONResponse(status_code=status,
                        content={"error": type(exc).__name__, "detail": str(exc)})


for module in (system, explore, papers, reports, conversations):
    app.include_router(module.router)