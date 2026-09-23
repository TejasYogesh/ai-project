"""Shared state for the paper graph."""
import operator
from typing import Annotated, Literal, TypedDict

from backend.models.schemas import Briefing, PaperMeta, ParsedPaper, RankedPaper


class PaperState(TypedDict, total=False):
    """Everything the paper graph knows, filled in step by step by the nodes."""
    # input
    user_input: str
    # understand
    intent: Literal["paper_id", "topic"]
    arxiv_id: str
    query: str
    # fetch / explore
    paper: PaperMeta
    candidates: list[RankedPaper]      # shortlist from the topic explorer
    # parse
    parsed: ParsedPaper
    # index
    collection_name: str
    # summarize
    briefing: Briefing
    # errors
    error: str
    failed_node: str
    # accumulated across nodes (reducer: lists are joined, not replaced)
    warnings: Annotated[list[str], operator.add]
    steps: Annotated[list[str], operator.add]