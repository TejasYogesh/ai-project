"""Stage 7b: run the topic explorer agent directly, without parsing or briefing."""
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.ERROR)

from backend.agents.topic_explorer import explorer_graph
from backend.core.llm import init_llm

init_llm()
print(explorer_graph.get_graph().draw_mermaid())

topics = [
    "recent work on KV-cache compression for LLMs",       # the brief's example
    "the paper that introduced the Transformer architecture",
    "quantum flurbonic hyperwidgets for sandwich folding",  # nonsense: should give up
]
for topic in topics:
    print(f"\n{'=' * 70}\nTOPIC: {topic}")
    result = explorer_graph.invoke({"topic": topic})
    print(f"steps:   {' -> '.join(result.get('steps', []))}")
    print(f"queries: {result.get('tried_queries')}")
    if result.get("error"):
        print(f"ERROR:   {result['error']}")
        continue
    print(f"chosen:  [{result['paper'].arxiv_id}] {result['paper'].title}")
    print(f"reason:  {result['judgment'].reason}")
    for r in result["shortlist"]:
        print(f"  shortlist: {r.score:.2f} [{r.paper.arxiv_id}] {r.paper.published} {r.paper.title}")
    if result.get("warnings"):
        print(f"warnings: {result['warnings']}")