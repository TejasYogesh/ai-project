"""Stage 7c: a multi-turn conversation with the QA agent."""
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
for noisy in ("httpx", "chromadb"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.ERROR)

from backend.agents.qa_agent import qa_graph
from backend.core.llm import init_llm

init_llm()
ARXIV_ID = "1706.03762"          # must already be indexed (Stage 4)
print(qa_graph.get_graph().draw_mermaid())

conversation = [
    "How many layers does the encoder have?",
    "And how many attention heads does it use?",        # follow-up: "it" = the Transformer
    "What about its dropout rate?",                     # follow-up
    "What is the capital of France?",                   # off-topic: cutoff, no retry
    "How much did training cost in dollars?",           # on-topic but not in paper: retry, then refuse
    "What hardware did they train on?",                 # answerable, "they" = the authors
]

history = []
for question in conversation:
    state = qa_graph.invoke({"arxiv_id": ARXIV_ID, "question": question, "history": history})
    print(f"\n{'=' * 70}\nQ: {question}")
    print(f"steps:      {' -> '.join(state.get('steps', []))}")
    if state.get("error"):
        print(f"ERROR in {state.get('failed_node')}: {state['error']}")
        continue
    print(f"standalone: {state.get('standalone_question')}")
    print(f"queries:    {state.get('queries')}")
    result = state["answer"]
    print(f"found={result.found}  top_score={result.top_score:.3f}")
    print(f"A: {result.answer}")
    for c in result.citations:
        print(f"   [{c.number}] {c.section} (p.{c.page})")
    history.append(result)