"""Grounded question answering over one indexed paper.

Building blocks used by the QA agent, plus answer_question(), a simple linear version.
"""
import logging

from llama_index.core.schema import NodeWithScore, TextNode

from backend.core.config import settings
from backend.core.llm import structured_llm_call
from backend.models import database as db
from backend.models.schemas import AnswerDraft, Citation, GradeResult, PaperMeta, QAAnswer
from backend.services import vector_store
from backend.services.summarizer import metadata_text
from backend.services.prompts import ANSWER_PROMPT, GRADE_PROMPT

logger = logging.getLogger(__name__)

NOT_FOUND = "The paper does not appear to address this question."
MAX_HISTORY_TURNS = 3


# ---------- formatting ----------

def format_context(nodes: list[NodeWithScore]) -> str:
    """Number the chunks so the LLM can cite them as [1], [2], ..."""
    return "\n\n".join(
        f"[{i}] (Section: {n.metadata['section']}, p.{n.metadata['page']})\n{n.text}"
        for i, n in enumerate(nodes, start=1)
    )


def format_history(history: list[QAAnswer]) -> str:
    """Last few answered turns, for resolving follow-up questions."""
    recent = [h for h in history if h.found][-MAX_HISTORY_TURNS:]
    if not recent:
        return "(none)"
    return "\n".join(f"Q: {h.question}\nA: {h.answer}" for h in recent)


# ---------- building blocks ----------

def not_found_answer(question: str, top_score: float | None, reason: str) -> QAAnswer:
    logger.info("Not answering %r: %s", question, reason)
    return QAAnswer(question=question, answer=NOT_FOUND, found=False, top_score=top_score)


def paper_details_node(arxiv_id: str) -> NodeWithScore | None:
    """Title/authors/date from arXiv metadata. The PDF chunks rarely carry these reliably,
    so questions like 'what is the title?' need this pseudo-chunk."""
    row = db.get_paper(arxiv_id)
    if row is None:
        return None
    text = metadata_text(PaperMeta.model_validate_json(row.meta_json))
    return NodeWithScore(node=TextNode(text=text, metadata={"section": "Paper details", "page": 1}))


def retrieve_relevant(arxiv_id: str, query: str) -> tuple[list[NodeWithScore], float | None]:
    """Layer 1: retrieve top-k chunks and keep those above the similarity cutoff, plus
    the paper-details chunk when chunks survive. Returns (kept_chunks, best_score)."""
    results = vector_store.retrieve(arxiv_id, query)
    top_score = results[0].score if results else None
    kept = [r for r in results if r.score is not None and r.score >= settings.similarity_cutoff]
    details = paper_details_node(arxiv_id) if kept else None
    return ([details] if details else []) + kept, top_score


def grade_context(question: str, nodes: list[NodeWithScore]) -> GradeResult:
    """Layer 2: ask the LLM whether the chunks actually contain the answer."""
    return structured_llm_call(
        GradeResult, GRADE_PROMPT, question=question, context=format_context(nodes))


def generate_answer(question: str, nodes: list[NodeWithScore],
                    history: list[QAAnswer]) -> AnswerDraft:
    """Write an answer that cites the numbered chunks."""
    return structured_llm_call(
        AnswerDraft, ANSWER_PROMPT, question=question,
        context=format_context(nodes), history=format_history(history))


def finalize_answer(question: str, nodes: list[NodeWithScore], draft: AnswerDraft,
                    top_score: float | None) -> QAAnswer:
    """Layer 3: accept the answer only if it cites chunks we actually provided."""
    valid = sorted({n for n in draft.cited_numbers if 1 <= n <= len(nodes)})
    if not draft.answerable or not valid:
        return not_found_answer(question, top_score, "answer was not grounded in cited chunks")
    citations = [Citation(number=n, section=nodes[n - 1].metadata["section"],
                          page=nodes[n - 1].metadata["page"]) for n in valid]
    return QAAnswer(question=question, answer=draft.answer, found=True,
                    citations=citations, top_score=top_score)


def relevant_subset(nodes: list[NodeWithScore], grade: GradeResult) -> list[NodeWithScore]:
    """Keep only the chunks the grader marked relevant (all of them if it marked none)."""
    chosen = [nodes[i - 1] for i in grade.relevant_numbers if 1 <= i <= len(nodes)]
    return chosen or nodes


# ---------- simple linear pipeline (Stage 5) ----------

def answer_question(arxiv_id: str, question: str,
                    history: list[QAAnswer] | None = None) -> QAAnswer:
    """Retrieve, filter, grade, answer and verify citations, with no retries."""
    history = history or []
    nodes, top_score = retrieve_relevant(arxiv_id, question)
    if not nodes:
        return not_found_answer(question, top_score,
                                f"no chunk above cutoff {settings.similarity_cutoff}")
    grade = grade_context(question, nodes)
    if not grade.sufficient:
        return not_found_answer(question, top_score, f"grader: {grade.reason}")
    nodes = relevant_subset(nodes, grade)
    draft = generate_answer(question, nodes, history)
    return finalize_answer(question, nodes, draft, top_score)