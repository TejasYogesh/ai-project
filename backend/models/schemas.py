# backend/models/schemas.py
from pydantic import BaseModel


class PaperMeta(BaseModel):
    """Metadata for one arXiv paper, as returned by the arXiv API."""
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str
    pdf_url: str
    abs_url: str
    categories: list[str]
    comment: str | None = None