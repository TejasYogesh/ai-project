"""AGENT: grounded QA with question condensing and one retry via query rewriting."""
import logging
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from llama_index.core.schema import NodeWithScore

from backend.agents.prompts import CONDENSE_PROMPT, REWRITE_PROMPT
from backend.agents.utils import graph_node
from backend.core.exceptions import QAError
from backend.core.llm import structured_llm_call
from backend.models.schemas import GradeResult, QAAnswer, RewrittenQuery, StandaloneQuestion
from backend.services import qa_service

logger = logging.getLogger(__name__)

MAX_RETRIEVALS = 2          # first search + one retry


class QAState(TypedDict, total=False):
    arxiv_id: str
    question: str                  # exactly what the user typed
    history: list[QAAnswer]
    standalone_question: str       # follow-up resolved using history (used for grading/answering)
    search_query: str              # what we send to Chroma (may be rewritten on retry)
    attempts: int
    nodes: list[NodeWithScore]
    top_score: float | None
    grade: GradeResult
    answer: QAAnswer
    error: str
    failed_node: str
    queries: Annotated[list[str], operator.add]
    steps: Annotated[list[str], operator.add]


# ---------- nodes ----------

@graph_node("condense")
def condense(state: QAState) -> dict:
    """Rewrite a follow-up into a standalone question (no LLM call without history)."""
    answered = [h for h in state.get("history", []) if h.found]
    if not answered:
        question = state["question"]
    else:
        question = structured_llm_call(
            StandaloneQuestion, CONDENSE_PROMPT,
            history=qa_service.format_history(answered), question=state["question"]).question
    logger.info("Standalone question: %r", question)
    return {"standalone_question": question, "search_query": question, "attempts": 0}


@graph_node("retrieve")
def retrieve(state: QAState) -> dict:
    nodes, top = qa_service.retrieve_relevant(state["arxiv_id"], state["search_query"])
    previous = state.get("top_score")
    best = top if previous is None or (top is not None and top > previous) else previous
    return {"nodes": nodes, "top_score": best, "attempts": state["attempts"] + 1,
            "queries": [state["search_query"]]}


@graph_node("grade")
def grade(state: QAState) -> dict:
    result = qa_service.grade_context(state["standalone_question"], state["nodes"])
    logger.info("Grade: sufficient=%s reason=%s", result.sufficient, result.reason)
    return {"grade": result}


@graph_node("rewrite")
def rewrite(state: QAState) -> dict:
    new_query = structured_llm_call(
        RewrittenQuery, REWRITE_PROMPT, question=state["standalone_question"],
        tried=", ".join(repr(q) for q in state.get("queries", [])),
        reason=state["grade"].reason).query
    logger.info("Rewritten search query: %r", new_query)
    return {"search_query": new_query}


@graph_node("answer")
def answer(state: QAState) -> dict:
    nodes = qa_service.relevant_subset(state["nodes"], state["grade"])
    draft = qa_service.generate_answer(state["standalone_question"], nodes, state.get("history", []))
    return {"answer": qa_service.finalize_answer(state["question"], nodes, draft,
                                                 state.get("top_score"))}


@graph_node("not_found")
def not_found(state: QAState) -> dict:
    grade_result = state.get("grade")
    reason = (f"grader: {grade_result.reason}" if grade_result and not grade_result.sufficient
              else "no chunk above the similarity cutoff")
    return {"answer": qa_service.not_found_answer(state["question"], state.get("top_score"), reason)}


# ---------- routers ----------

def ok_or_error(state: QAState) -> str:
    return "error" if state.get("error") else "ok"


def after_retrieve(state: QAState) -> str:
    if state.get("error"):
        return "error"
    return "grade" if state["nodes"] else "not_found"     # off-topic: no retry


def after_grade(state: QAState) -> str:
    if state.get("error"):
        return "error"
    if state["grade"].sufficient:
        return "answer"
    if state["attempts"] < MAX_RETRIEVALS:
        return "rewrite"
    return "not_found"


# ---------- graph ----------

def build_qa_graph():
    graph = StateGraph(QAState)
    for name, fn in [("condense", condense), ("retrieve", retrieve), ("grade", grade),
                     ("rewrite", rewrite), ("answer", answer), ("not_found", not_found)]:
        graph.add_node(name, fn)

    graph.add_edge(START, "condense")
    graph.add_conditional_edges("condense", ok_or_error, {"ok": "retrieve", "error": END})
    graph.add_conditional_edges("retrieve", after_retrieve,
                                {"grade": "grade", "not_found": "not_found", "error": END})
    graph.add_conditional_edges("grade", after_grade,
                                {"answer": "answer", "rewrite": "rewrite",
                                 "not_found": "not_found", "error": END})
    graph.add_conditional_edges("rewrite", ok_or_error, {"ok": "retrieve", "error": END})
    graph.add_edge("answer", END)
    graph.add_edge("not_found", END)
    return graph.compile()


qa_graph = build_qa_graph()


def ask(arxiv_id: str, question: str, history: list[QAAnswer] | None = None) -> QAAnswer:
    """Answer one question about an indexed paper. Raises QAError if the agent fails."""
    state = qa_graph.invoke({"arxiv_id": arxiv_id, "question": question,
                             "history": history or []})
    if state.get("error"):
        raise QAError(f"QA failed in {state.get('failed_node')}: {state['error']}")
    return state["answer"]