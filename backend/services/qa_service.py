"""Grounded question answering over one indexed paper."""
import logging

from llama_index.core import Settings
from llama_index.core.schema import NodeWithScore

from backend.core.config import settings
from backend.models.schemas import AnswerDraft, Citation, GradeResult, QAAnswer
from backend.services import vector_store
from backend.services.prompts import ANSWER_PROMPT, GRADE_PROMPT

logger = logging.getLogger(__name__)

NOT_FOUND = "The paper does not appear to address this question."
MAX_HISTORY_TURNS = 3


def _format_context(nodes: list[NodeWithScore]) -> str:
    """Number the chunks so the LLM can cite them as [1], [2], ..."""
    return "\n\n".join(
        f"[{i}] (Section: {n.metadata['section']}, p.{n.metadata['page']})\n{n.text}"
        for i, n in enumerate(nodes, start=1)
    )


def _format_history(history: list[QAAnswer]) -> str:
    """Last few turns, so follow-ups like 'what about its limitations?' work."""
    recent = [h for h in history if h.found][-MAX_HISTORY_TURNS:]
    if not recent:
        return "(none)"
    return "\n".join(f"Q: {h.question}\nA: {h.answer}" for h in recent)


def _not_found(question: str, top_score: float | None, reason: str) -> QAAnswer:
    logger.info("Not answering %r: %s", question, reason)
    return QAAnswer(question=question, answer=NOT_FOUND, found=False, top_score=top_score)


def grade_context(question: str, nodes: list[NodeWithScore]) -> GradeResult:
    """Layer 2: ask the LLM whether the chunks actually contain the answer."""
    return Settings.llm.structured_predict(
        GradeResult, GRADE_PROMPT, question=question, context=_format_context(nodes))


def generate_answer(question: str, nodes: list[NodeWithScore],
                    history: list[QAAnswer]) -> AnswerDraft:
    """Write an answer that cites the numbered chunks."""
    return Settings.llm.structured_predict(
        AnswerDraft, ANSWER_PROMPT, question=question,
        context=_format_context(nodes), history=_format_history(history))


def answer_question(arxiv_id: str, question: str,
                    history: list[QAAnswer] | None = None) -> QAAnswer:
    """Retrieve, filter, grade, answer and verify citations."""
    history = history or []
    results = vector_store.retrieve(arxiv_id, question)
    top_score = results[0].score if results else None

    # Layer 1: similarity cutoff (no LLM call)
    nodes = [r for r in results if r.score is not None and r.score >= settings.similarity_cutoff]
    if not nodes:
        return _not_found(question, top_score, f"no chunk above cutoff {settings.similarity_cutoff}")

    # Layer 2: LLM grader
    grade = grade_context(question, nodes)
    if not grade.sufficient:
        return _not_found(question, top_score, f"grader: {grade.reason}")
    relevant = [nodes[i - 1] for i in grade.relevant_numbers if 1 <= i <= len(nodes)]
    nodes = relevant or nodes

    # Answer
    draft = generate_answer(question, nodes, history)

    # Layer 3: citations must point at chunks we actually provided
    valid = sorted({n for n in draft.cited_numbers if 1 <= n <= len(nodes)})
    if not draft.answerable or not valid:
        return _not_found(question, top_score, "answer was not grounded in cited chunks")

    citations = [Citation(number=n, section=nodes[n - 1].metadata["section"],
                          page=nodes[n - 1].metadata["page"]) for n in valid]
    return QAAnswer(question=question, answer=draft.answer, found=True,
                    citations=citations, top_score=top_score)