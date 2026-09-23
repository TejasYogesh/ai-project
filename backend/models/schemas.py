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
    
# ---------- Briefing (Stage 6) ----------

class Limitation(BaseModel):
    """One limitation, labelled by where it comes from."""
    text: str = Field(description="The limitation, in one sentence")
    source: Literal["stated", "inferred"] = Field(
        description="'stated' if the authors say it in the paper, 'inferred' if you deduced it")


class BriefingContent(BaseModel):
    """The part of the briefing the LLM writes."""
    why_it_matters: str = Field(
        description="One plain-English paragraph (3-5 sentences) for a smart non-specialist: "
                    "what the paper does and why it matters, based only on the paper's own claims")
    problem_statement: str = Field(description="The problem the paper addresses, in 2-3 sentences")
    method: list[str] = Field(description="3-6 bullet points describing the approach")
    key_results: list[str] = Field(
        description="3-6 key results or claims, including the specific numbers reported in the paper")
    limitations: list[Limitation] = Field(
        min_length=1, description="At least one limitation; include stated ones and inferred ones")
    suggested_questions: list[str] = Field(
        description="3-5 follow-up questions a reader might ask that the paper can answer")


class Briefing(BriefingContent):
    """The full briefing: LLM-written content plus metadata taken directly from arXiv."""
    title: str
    authors: list[str]
    arxiv_id: str
    published: str
    link: str
    parse_quality: str
    warnings: list[str] = []