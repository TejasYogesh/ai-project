"""Prompt templates used by services."""
from llama_index.core import PromptTemplate

GRADE_PROMPT = PromptTemplate(
    "You are checking whether excerpts from a research paper contain enough information "
    "to answer a question.\n\n"
    "Question: {question}\n\n"
    "Excerpts:\n{context}\n\n"
    "List the excerpt numbers that are relevant, and decide whether together they are "
    "sufficient to answer the question. Be strict: excerpts about a related topic are NOT "
    "sufficient; the specific information asked for must actually be present."
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
)