"""Prompt templates used only by agents."""
from llama_index.core import PromptTemplate

PLAN_PROMPT = PromptTemplate(
    "Turn this research topic into a short arXiv search query.\n\n"
    "Topic: {topic}\n\n"
    "Rules:\n"
    "- 2 to 6 keywords: the technical terms a paper's title or abstract would use.\n"
    "- Drop filler words like 'recent work on', 'papers about', 'for'.\n"
    "- Set wants_recent to true only if the topic asks for recent, latest, new or current work."
)

JUDGE_PROMPT = PromptTemplate(
    "A user wants a research paper on this topic: {topic}\n\n"
    "Search queries already tried: {tried}\n\n"
    "Candidate papers, ranked by similarity to the topic:\n{candidates}\n\n"
    "Decide:\n"
    "- relevant_ids: arXiv IDs of candidates that are genuinely about the topic (not just mentioning it).\n"
    "- best_id: the single best paper to brief. Prefer the most directly on-topic one; if the "
    "topic asks for recent work, prefer newer papers. Null if none are relevant.\n"
    "- good_enough: true only if best_id is clearly on-topic.\n"
    "- refined_query: if not good enough, a NEW arXiv query different from those already tried. "
    "Make it broader if there were no or few results, more specific if results were off-topic.\n"
    "- reason: one sentence."
    "- refined_query: if not good enough, a NEW arXiv query different from those already tried. "
    "Make it broader if there were no or few results, more specific if results were off-topic. "
    "Plain keywords are all required to match. To find a specific known paper, you may search "
    'the title or author field, e.g. ti:"exact title" or au:surname.\n'
)

CONDENSE_PROMPT = PromptTemplate(
    "Rewrite the user's latest question as a standalone question about the research paper, "
    "replacing words like 'it', 'that', 'they' or 'this' with what they refer to in the "
    "conversation. Keep the meaning exactly; do not answer it. If the question is already "
    "standalone, return it unchanged.\n\n"
    "Conversation:\n{history}\n\n"
    "Latest question: {question}"
)

REWRITE_PROMPT = PromptTemplate(
    "A search over chunks of a research paper did not find enough information to answer "
    "this question.\n\n"
    "Question: {question}\n"
    "Search queries already tried: {tried}\n"
    "Why the retrieved text was insufficient: {reason}\n\n"
    "Write ONE new search query that is more likely to match the paper's own wording: use the "
    "technical terms a paper would use, synonyms, or the kind of section likely to contain the "
    "answer. It must be different from the queries already tried."
)