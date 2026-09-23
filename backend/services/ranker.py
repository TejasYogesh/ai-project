"""Rank candidate papers by embedding similarity to the user's topic."""
import math

from llama_index.core import Settings

from backend.models.schemas import PaperMeta, RankedPaper


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def rank_by_similarity(topic: str, papers: list[PaperMeta]) -> list[RankedPaper]:
    """Most similar first. Uses one query embedding and one batch for the abstracts."""
    if not papers:
        return []
    embed = Settings.embed_model
    topic_vector = embed.get_query_embedding(topic)
    paper_vectors = embed.get_text_embedding_batch([f"{p.title}. {p.abstract}" for p in papers])
    ranked = [RankedPaper(paper=p, score=_cosine(topic_vector, v))
              for p, v in zip(papers, paper_vectors)]
    return sorted(ranked, key=lambda r: r.score, reverse=True)