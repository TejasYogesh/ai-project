"""Create and read executive briefings."""
from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from backend.agents import session
from backend.core.exceptions import ReportNotFound
from backend.models import database as db
from backend.models.schemas import Briefing, ReportRequest, ReportSession, ReportSummary
from backend.services.summarizer import briefing_to_markdown

router = APIRouter(prefix="/reports", tags=["reports"])


def _load(report_id: str) -> Briefing:
    briefing = db.get_report(report_id)
    if briefing is None:
        raise ReportNotFound(f"No report with ID {report_id}.")
    return briefing


@router.post("", response_model=ReportSession)
def create_report(request: ReportRequest) -> ReportSession:
    """Brief a paper from an arXiv ID, URL or topic. Also starts a conversation for QA."""
    return session.create_report(request.input, refresh=request.refresh)


@router.get("", response_model=list[ReportSummary])
def list_reports() -> list[ReportSummary]:
    return [ReportSummary(**r) for r in db.list_reports()]


@router.get("/{report_id}", response_model=Briefing)
def get_report(report_id: str) -> Briefing:
    return _load(report_id)


@router.get("/{report_id}/markdown", response_class=PlainTextResponse)
def get_report_markdown(report_id: str) -> str:
    return briefing_to_markdown(_load(report_id))