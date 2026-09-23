"""Stage 4: index papers into ChromaDB once, then retrieve without re-embedding."""
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
for noisy in ("httpx", "chromadb", "google_genai"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

from backend.core.llm import init_llm
from backend.services import vector_store
from backend.services.arxiv_client import search
from backend.services.pdf_parser import parse_pdf

init_llm()
ARXIV_ID = "1706.03762"

# 1. Build (or reuse) the index
paper = search(id_list=ARXIV_ID)[0]
start = time.time()
if vector_store.is_indexed(ARXIV_ID):
    print("Already indexed, skipping parse and embedding.")
else:
    parsed = parse_pdf(paper)
    vector_store.build_index(paper, parsed)
print(f"Index ready in {time.time() - start:.1f}s\n")

# 2. Retrieve
questions = [
    "How many layers does the encoder have?",
    "What optimizer did they use for training?",
    "What BLEU score did the model get on English-to-German?",
    "What is the capital of France?",
]
for question in questions:
    print(f"Q: {question}")
    for result in vector_store.retrieve(ARXIV_ID, question):
        preview = result.text[:80].replace("\n", " ")
        print(f"  {result.score:.3f} | {result.metadata['section']} (p.{result.metadata['page']}) | {preview}")
    print()