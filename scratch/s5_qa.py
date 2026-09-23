"""Stage 5: grounded QA with refusal for questions the paper does not answer."""
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
for noisy in ("httpx", "chromadb", "google_genai"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

from backend.core.config import settings
from backend.core.llm import init_llm
from backend.services.qa_service import answer_question

init_llm()
ARXIV_ID = "1706.03762"          # indexed in Stage 4
print(f"similarity_cutoff = {settings.similarity_cutoff}\n")

questions = [
    # answerable
    "How many layers does the encoder have?",
    "What optimizer did they use and with what settings?",
    "What BLEU score did the big model get on English-to-German?",
    # off-topic: should be stopped by the cutoff
    "What is the capital of France?",
    # on-topic but NOT in the paper: should be stopped by the grader
    "How does the Transformer perform on image classification?",
    "How much did it cost in dollars to train the big model?",
    "What is the main contribution of this paper?",
    "What are the limitations of this approach?",
]

history = []
for question in questions:
    result = answer_question(ARXIV_ID, question, history)
    history.append(result)
    print(f"Q: {question}")
    print(f"   top_score={result.top_score:.3f}  found={result.found}")
    print(f"   A: {result.answer}")
    for c in result.citations:
        print(f"      [{c.number}] {c.section} (p.{c.page})")
    print()

# follow-up that depends on history
follow_up = answer_question(ARXIV_ID, "Why did they choose that optimizer's warmup schedule?", history)
print(f"Follow-up: found={follow_up.found}\n   A: {follow_up.answer}")