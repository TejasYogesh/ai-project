"""Pydantic schemas shared across the project."""
from typing import Literal

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


class Section(BaseModel):
    """One section of a parsed paper."""
    title: str
    text: str
    page_start: int


class ParsedPaper(BaseModel):
    """Result of parsing a paper's PDF."""
    arxiv_id: str
    sections: list[Section]
    references: str = ""
    num_pages: int
    parse_quality: Literal["full", "partial", "abstract_only"]
    warnings: list[str] = []