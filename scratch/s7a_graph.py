"""Stage 7a: run the paper graph on a valid ID, an invalid ID, and a topic."""
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.ERROR)

from backend.agents.paper_agent import paper_graph, run_paper_graph
from backend.core.llm import init_llm

init_llm()

print("Graph diagram (Mermaid):\n")
print(paper_graph.get_graph().draw_mermaid())

for user_input in ["https://arxiv.org/abs/1706.03762", "9999.99999", "kv cache compression"]:
    print(f"\n{'=' * 70}\nINPUT: {user_input!r}")
    state = run_paper_graph(user_input)
    print(f"steps:    {' -> '.join(state.get('steps', []))}")
    if state.get("error"):
        print(f"ERROR in {state.get('failed_node')}: {state['error']}")
        continue
    briefing = state["briefing"]
    print(f"title:    {briefing.title}")
    print(f"index:    {state['collection_name']}")
    print(f"warnings: {state.get('warnings', [])}")
    print(f"summary:  {briefing.why_it_matters[:200]}...")