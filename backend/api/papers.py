"""Information about papers the system has already processed."""
from fastapi import APIRouter

from backend.core.exceptions import PaperNotFound
from backend.models import database as db
from backend.models.schemas import PaperMeta, PaperView
from backend.services import vector_store

router = APIRouter(prefix="/papers", tags=["papers"])


@router.get("/{arxiv_id}", response_model=PaperView)
def get_paper(arxiv_id: str) -> PaperView:
    row = db.get_paper(arxiv_id)
    if row is None:
        raise PaperNotFound(f"Paper {arxiv_id} has not been processed yet. POST /reports to brief it.")
    return PaperView(paper=PaperMeta.model_validate_json(row.meta_json),
                     parse_quality=row.parse_quality,
                     indexed=vector_store.is_indexed(arxiv_id),
                     report_ids=db.report_ids_for(arxiv_id))