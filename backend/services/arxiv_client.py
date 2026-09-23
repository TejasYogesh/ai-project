"""Search arXiv via its official Atom API and convert results into PaperMeta objects."""
import logging
import re
import time

import feedparser
import httpx

from backend.core.exceptions import ArxivAPIError, ArxivQueryError        # NEW
from backend.models.schemas import PaperMeta

logger = logging.getLogger(__name__)

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")
BOOLEAN_RE = re.compile(r'\b(AND|OR|ANDNOT)\b|["()]')
FIELD_PREFIX_RE = re.compile(r"\b(all|ti|au|abs|cat):", re.IGNORECASE)    # NEW
STOP_WORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from",  # NEW
              "in", "into", "is", "it", "of", "on", "or", "the", "to", "with"}
MIN_SECONDS_BETWEEN_CALLS = 3.0
_last_call = 0.0


def extract_arxiv_id(text: str) -> str | None:
    """Find an arXiv ID in text like '2401.12345', '2401.12345v2' or an arxiv.org URL."""
    match = ARXIV_ID_RE.search(text)
    return match.group(1) if match else None


def _respect_rate_limit() -> None:
    """arXiv asks for ~3 seconds between requests."""
    global _last_call
    wait = MIN_SECONDS_BETWEEN_CALLS - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.time()


def _build_search_query(query: str) -> str:
    """Turn a query into arXiv search syntax.

    - Already uses field prefixes (ti:, au:, abs:, all:, cat:) -> passed through unchanged
      e.g. ti:"Attention Is All You Need"
    - Uses AND / OR / quotes / parentheses -> searched in all fields as written
    - Plain keywords -> every non-stop-word must match:
      'Attention Is All You Need' -> 'all:Attention AND all:All AND all:You AND all:Need'
    """
    if FIELD_PREFIX_RE.search(query):                                      # NEW
        return query
    if BOOLEAN_RE.search(query):
        return f"all:{query}"
    words = [w for w in query.split() if w.lower() not in STOP_WORDS] or query.split()  # NEW
    return " AND ".join(f"all:{word}" for word in words)


def _entry_to_paper(entry) -> PaperMeta:
    """Convert one feedparser entry into a PaperMeta."""
    arxiv_id = extract_arxiv_id(entry.id)
    if not arxiv_id:
        raise ValueError(f"Could not extract arXiv ID from {entry.id!r}")

    pdf_url = next(
        (link.href for link in entry.links if link.get("type") == "application/pdf"),
        f"https://arxiv.org/pdf/{arxiv_id}",
    )

    return PaperMeta(
        arxiv_id=arxiv_id,
        title=" ".join(entry.title.split()),
        authors=[a["name"] for a in entry.get("authors", [])],
        abstract=" ".join(entry.summary.split()),
        published=entry.published[:10],
        pdf_url=pdf_url,
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
        categories=[t["term"] for t in entry.get("tags", [])],
        comment=entry.get("arxiv_comment"),
    )


def search(query: str | None = None, id_list: str | None = None,
           max_results: int = 10, sort_by: str = "relevance") -> list[PaperMeta]:
    """Search arXiv by topic (query) or by ID (id_list). Returns [] if nothing matches.
    sort_by: "relevance" or "submittedDate" (newest first)."""
    if not query and not id_list:
        raise ValueError("Provide either query or id_list")

    params = {"max_results": max_results}
    if id_list:
        params["id_list"] = id_list
    else:
        params["search_query"] = _build_search_query(query)
        params["sortBy"] = sort_by

    _respect_rate_limit()
    try:
        response = httpx.get(ARXIV_API, params=params, timeout=30)
        response.raise_for_status()
    except httpx.HTTPStatusError as e:                                     # NEW
        if e.response.status_code == 400:
            raise ArxivQueryError(
                f"arXiv rejected the query {params.get('search_query')!r}") from e
        raise ArxivAPIError(f"arXiv API request failed: {e}") from e
    except httpx.HTTPError as e:
        raise ArxivAPIError(f"arXiv API request failed: {e}") from e

    feed = feedparser.parse(response.text)
    papers = []
    for entry in feed.entries:
        try:
            papers.append(_entry_to_paper(entry))
        except Exception as e:
            logger.warning("Skipping malformed arXiv entry: %r", e)
    return papers