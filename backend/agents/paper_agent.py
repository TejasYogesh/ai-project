"""LangGraph state graph: arXiv input -> paper -> parsed -> indexed -> briefing."""
import logging

from langgraph.graph import END, START, StateGraph

from backend.agents.state import PaperState
from backend.agents.topic_explorer import explorer_graph
from backend.agents.utils import graph_node
from backend.core.exceptions import PaperNotFound
from backend.services import arxiv_client, pdf_parser, summarizer, vector_store

logger = logging.getLogger(__name__)


# ---------- nodes ----------

@graph_node("understand")
def understand(state: PaperState) -> dict:
    """Decide whether the input is an arXiv ID/URL or a topic."""
    arxiv_id = arxiv_client.extract_arxiv_id(state["user_input"])
    if arxiv_id:
        return {"intent": "paper_id", "arxiv_id": arxiv_id}
    return {"intent": "topic", "query": state["user_input"].strip()}


@graph_node("topic_explorer")
def topic_explorer(state: PaperState) -> dict:
    """Run the explorer agent; its contract is to put a `paper` in the state."""
    result = explorer_graph.invoke({"topic": state["query"]})
    sub_steps = [f"explorer.{s}" for s in result.get("steps", [])]
    if result.get("error"):
        return {"error": result["error"], "steps": sub_steps}
    return {"paper": result["paper"], "candidates": result["shortlist"],
            "warnings": result.get("warnings", []), "steps": sub_steps}


@graph_node("fetch_paper")
def fetch_paper(state: PaperState) -> dict:
    """Look up the paper's metadata on arXiv."""
    papers = arxiv_client.search(id_list=state["arxiv_id"])
    if not papers:
        raise PaperNotFound(f"No arXiv paper found with ID {state['arxiv_id']}.")
    return {"paper": papers[0]}


@graph_node("parse")
def parse(state: PaperState) -> dict:
    """Download and parse the PDF into sections."""
    parsed = pdf_parser.parse_pdf(state["paper"])
    return {"parsed": parsed, "warnings": parsed.warnings}


@graph_node("index")
def index(state: PaperState) -> dict:
    """Chunk, embed and store in Chroma (skipped if already indexed)."""
    name = vector_store.build_index(state["paper"], state["parsed"])
    return {"collection_name": name}


@graph_node("summarize")
def summarize(state: PaperState) -> dict:
    """Generate the executive briefing."""
    return {"briefing": summarizer.make_briefing(state["paper"], state["parsed"])}


# ---------- routers (used by conditional edges) ----------

def route_intent(state: PaperState) -> str:
    return state["intent"]


def ok_or_error(state: PaperState) -> str:
    return "error" if state.get("error") else "ok"


# ---------- graph ----------

def build_paper_graph():
    graph = StateGraph(PaperState)

    graph.add_node("understand", understand)
    graph.add_node("topic_explorer", topic_explorer)
    graph.add_node("fetch_paper", fetch_paper)
    graph.add_node("parse", parse)
    graph.add_node("index", index)
    graph.add_node("summarize", summarize)

    graph.add_edge(START, "understand")
    graph.add_conditional_edges("understand", route_intent,
                                {"paper_id": "fetch_paper", "topic": "topic_explorer"})
    graph.add_conditional_edges("topic_explorer", ok_or_error, {"ok": "parse", "error": END})
    graph.add_conditional_edges("fetch_paper", ok_or_error, {"ok": "parse", "error": END})
    graph.add_conditional_edges("parse", ok_or_error, {"ok": "index", "error": END})
    graph.add_conditional_edges("index", ok_or_error, {"ok": "summarize", "error": END})
    graph.add_edge("summarize", END)

    return graph.compile()


paper_graph = build_paper_graph()


def run_paper_graph(user_input: str) -> PaperState:
    """Run the whole pipeline for one input. Requires init_llm() to have been called."""
    return paper_graph.invoke({"user_input": user_input})