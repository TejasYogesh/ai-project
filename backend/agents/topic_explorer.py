"""AGENT: search arXiv for a topic, judge the results, and refine the query if needed."""
import logging
import operator
from typing import Annotated, TypedDict
from backend.core.exceptions import ArxivQueryError
from langgraph.graph import END, START, StateGraph
from llama_index.core import Settings

from backend.agents.prompts import JUDGE_PROMPT, PLAN_PROMPT
from backend.agents.utils import graph_node
from backend.core.config import settings
from backend.models.schemas import Judgment, PaperMeta, RankedPaper, SearchPlan
from backend.services import arxiv_client, ranker

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3


class ExplorerState(TypedDict, total=False):
    topic: str
    query: str
    wants_recent: bool
    attempts: int
    tried_queries: Annotated[list[str], operator.add]
    candidates: list[RankedPaper]
    judgment: Judgment
    shortlist: list[RankedPaper]
    paper: PaperMeta
    error: str
    failed_node: str
    warnings: Annotated[list[str], operator.add]
    steps: Annotated[list[str], operator.add]


def _format_candidates(candidates: list[RankedPaper]) -> str:
    if not candidates:
        return "(no results)"
    return "\n".join(
        f"{i}. [{c.paper.arxiv_id}] ({c.paper.published}) similarity={c.score:.2f}\n"
        f"   {c.paper.title}\n   {c.paper.abstract[:350]}..."
        for i, c in enumerate(candidates, start=1)
    )


# ---------- nodes ----------

@graph_node("plan_search")
def plan_search(state: ExplorerState) -> dict:
    plan = Settings.llm.structured_predict(SearchPlan, PLAN_PROMPT, topic=state["topic"])
    return {"query": plan.keywords, "wants_recent": plan.wants_recent, "attempts": 0}

@graph_node("search")
def search(state: ExplorerState) -> dict:
    sort_by = "submittedDate" if state.get("wants_recent") else "relevance"
    try:
        papers = arxiv_client.search(query=state["query"], max_results=settings.arxiv_max_results,
                                     sort_by=sort_by)
    except ArxivQueryError as e:
        # an invalid query written by the LLM is a failed attempt, not a fatal error
        logger.warning("Query rejected by arXiv, treating it as 0 results: %s", e)
        papers = []
    logger.info("Query %r (%s) -> %d results", state["query"], sort_by, len(papers))
    return {"candidates": papers, "attempts": state["attempts"] + 1,
            "tried_queries": [state["query"]]}

@graph_node("rank")
def rank(state: ExplorerState) -> dict:
    return {"candidates": ranker.rank_by_similarity(state["topic"], state["candidates"])}


@graph_node("judge")
def judge(state: ExplorerState) -> dict:
    judgment = Settings.llm.structured_predict(
        Judgment, JUDGE_PROMPT, topic=state["topic"],
        tried=", ".join(state.get("tried_queries", [])),
        candidates=_format_candidates(state["candidates"]))
    logger.info("Judge: good_enough=%s best=%s reason=%s",
                judgment.good_enough, judgment.best_id, judgment.reason)
    return {"judgment": judgment}


@graph_node("refine")
def refine(state: ExplorerState) -> dict:
    return {"query": state["judgment"].refined_query or state["topic"]}


@graph_node("select")
def select(state: ExplorerState) -> dict:
    judgment = state["judgment"]
    by_id = {c.paper.arxiv_id: c for c in state["candidates"]}
    best = by_id[arxiv_client.extract_arxiv_id(judgment.best_id)]
    relevant = {arxiv_client.extract_arxiv_id(i) for i in judgment.relevant_ids}
    others = [c for c in state["candidates"] if c.paper.arxiv_id in relevant and c is not best]
    update = {"paper": best.paper, "shortlist": [best] + others[: settings.top_k_papers - 1]}
    if not judgment.good_enough:
        update["warnings"] = [f"No clearly on-topic paper found after {state['attempts']} searches; "
                              f"using the closest match: {best.paper.title}"]
    return update


@graph_node("give_up")
def give_up(state: ExplorerState) -> dict:
    tried = ", ".join(repr(q) for q in state.get("tried_queries", []))
    return {"error": f"No relevant arXiv papers found for this topic. Tried: {tried}."}


# ---------- router ----------

def ok_or_error(state: ExplorerState) -> str:
    return "error" if state.get("error") else "ok"


def route_after_judge(state: ExplorerState) -> str:
    if state.get("error"):
        return "error"
    judgment = state["judgment"]
    ids = {c.paper.arxiv_id for c in state["candidates"]}
    best_is_valid = bool(judgment.best_id) and arxiv_client.extract_arxiv_id(judgment.best_id) in ids
    out_of_tries = state["attempts"] >= MAX_ATTEMPTS

    if best_is_valid and (judgment.good_enough or out_of_tries):
        return "select"
    if out_of_tries:
        return "give_up"
    return "refine"


# ---------- graph ----------

def build_explorer_graph():
    graph = StateGraph(ExplorerState)
    for name, fn in [("plan_search", plan_search), ("search", search), ("rank", rank),
                     ("judge", judge), ("refine", refine), ("select", select),
                     ("give_up", give_up)]:
        graph.add_node(name, fn)

    graph.add_edge(START, "plan_search")
    # any failed node ends the run with `error` set, like the paper graph
    graph.add_conditional_edges("plan_search", ok_or_error, {"ok": "search", "error": END})
    graph.add_conditional_edges("search", ok_or_error, {"ok": "rank", "error": END})
    graph.add_conditional_edges("rank", ok_or_error, {"ok": "judge", "error": END})
    graph.add_conditional_edges("judge", route_after_judge,
                                {"select": "select", "refine": "refine", "give_up": "give_up",
                                 "error": END})
    graph.add_edge("refine", "search")          # the agent loop
    graph.add_edge("select", END)
    graph.add_edge("give_up", END)
    return graph.compile()


explorer_graph = build_explorer_graph()