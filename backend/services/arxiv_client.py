# backend/services/arxiv_client.py
import logging
import re
import time

import feedparser
import httpx

from backend.core.exceptions import ArxivAPIError
from backend.models.schemas import PaperMeta

logger = logging.getLogger(__name__)

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")
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
           max_results: int = 10) -> list[PaperMeta]:
    """Search arXiv by topic (query) or by ID (id_list). Returns [] if nothing matches."""
    if not query and not id_list:
        raise ValueError("Provide either query or id_list")

    params = {"max_results": max_results}
    if id_list:
        params["id_list"] = id_list
    else:
        params["search_query"] = f"all:{query}"
        params["sortBy"] = "relevance"

    _respect_rate_limit()
    try:
        response = httpx.get(ARXIV_API, params=params, timeout=30)
        response.raise_for_status()
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