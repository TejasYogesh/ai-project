"""Pydantic schemas shared across the project."""
from typing import Literal

from pydantic import BaseModel, Field

class Citation(BaseModel):
    """A chunk the answer relied on."""
    number: int
    section: str
    page: int


class QAAnswer(BaseModel):
    """Final answer returned to the user."""
    question: str
    answer: str
    found: bool                         # False = the paper does not address this
    citations: list[Citation] = []
    top_score: float | None = None      # best retrieval score, useful for tuning the cutoff


class GradeResult(BaseModel):
    """LLM output: do the excerpts contain the answer?"""
    relevant_numbers: list[int] = Field(description="Excerpt numbers that contain information needed to answer")
    sufficient: bool = Field(description="True only if the excerpts contain the specific information asked for")
    reason: str = Field(description="One short sentence explaining the decision")


class AnswerDraft(BaseModel):
    """LLM output: the answer and which excerpts it used."""
    answerable: bool = Field(description="False if the excerpts do not contain the answer")
    answer: str = Field(description="Concise answer with citations like [1] or [2][3]")
    cited_numbers: list[int] = Field(description="Excerpt numbers cited in the answer")
    
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