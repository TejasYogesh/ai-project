"""Prompt templates used by services."""
from llama_index.core import PromptTemplate

GRADE_PROMPT = PromptTemplate(
    "You are checking whether excerpts from a research paper contain information that "
    "answers a question.\n\n"
    "Question: {question}\n\n"
    "Excerpts:\n{context}\n\n"
    "List the excerpt numbers that are relevant, and decide whether they are sufficient.\n"
    "- sufficient = true if the excerpts contain information that directly answers the "
    "question, even if only partially or spread across sections (for example, limitations "
    "mentioned in the conclusion or as future work count).\n"
    "- sufficient = false if the excerpts are only about a related topic and the specific "
    "information asked for is absent (for example, the question asks for a cost in dollars "
    "but the paper only reports FLOPs)."
)

ANSWER_PROMPT = PromptTemplate(
    "Answer the question using ONLY the numbered excerpts from the paper below.\n\n"
    "Rules:\n"
    "- Every factual sentence must cite the excerpt(s) it comes from, like [1] or [2][3].\n"
    "- Do not use outside knowledge, even if you know the answer.\n"
    "- If the excerpts do not contain the answer, set answerable to false.\n"
    "- Be concise: 2 to 5 sentences.\n\n"
    "Recent conversation (only for resolving words like 'it' or 'that'):\n{history}\n\n"
    "Excerpts:\n{context}\n\n"
    "Question: {question}"
    "- If the excerpts answer only part of the question, answer that part and briefly say "
    "what the paper does not state.\n"
)

BRIEFING_PROMPT = PromptTemplate(
    "You are writing an executive briefing about a research paper for a busy engineer "
    "deciding whether to read it in full.\n\n"
    "Rules:\n"
    "- Use ONLY the paper text below. Do not use outside knowledge about this paper, its "
    "authors, or later work that built on it, even if you know about them.\n"
    "- Key results must use the exact numbers reported in the paper.\n"
    "- Limitations are required. Include limitations the authors state, and limitations you "
    "can reasonably infer from the paper (for example: what was not evaluated, assumptions "
    "made, compute requirements). Label each one correctly as 'stated' or 'inferred'.\n"
    "- Suggested questions must be answerable from the paper text.\n"
    "- Write for a smart non-specialist: plain English, no unexplained jargon.\n\n"
    "Paper metadata:\n{metadata}\n\n"
    "Paper text (sections in order):\n{paper_text}"
    "- Limitations are required. Include limitations the authors state, and limitations you "
    "can reasonably infer from the paper (for example: what was not evaluated, assumptions "
    "made, compute requirements). Label each one correctly as 'stated' or 'inferred'. "
    "Problems the authors name as future work count as 'stated'.\n"
)