"""Generate a structured executive briefing for a parsed paper."""
import logging
import re
from llama_index.core import Settings
from pydantic import ValidationError

from backend.core.exceptions import BriefingError
from backend.models.schemas import Briefing, BriefingContent, PaperMeta, ParsedPaper
from backend.services.prompts import BRIEFING_PROMPT

logger = logging.getLogger(__name__)

MAX_PAPER_CHARS = 120_000       # ~30k tokens; keeps cost and latency bounded
MAX_ATTEMPTS = 2


def _paper_text(parsed: ParsedPaper, max_chars: int) -> tuple[str, bool]:
    """Join sections in order, stopping before max_chars. Returns (text, was_truncated)."""
    parts: list[str] = []
    total = 0
    for section in parsed.sections:
        block = f"## {section.title} (p.{section.page_start})\n{section.text}\n"
        if total + len(block) > max_chars:
            return "\n".join(parts), True
        parts.append(block)
        total += len(block)
    return "\n".join(parts), False


def metadata_text(paper: PaperMeta) -> str:
    lines = [f"Title: {paper.title}",
             f"Authors: {', '.join(paper.authors)}",
             f"Published: {paper.published}",
             f"Categories: {', '.join(paper.categories)}"]
    if paper.comment:
        lines.append(f"arXiv comment: {paper.comment}")
    return "\n".join(lines)


def _generate_content(metadata: str, paper_text: str) -> BriefingContent:
    """Ask Gemini for the briefing content, retrying once if the output is invalid."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return Settings.llm.structured_predict(
                BriefingContent, BRIEFING_PROMPT, metadata=metadata, paper_text=paper_text)
        except (ValidationError, ValueError) as e:
            logger.warning("Briefing attempt %d/%d invalid: %s", attempt, MAX_ATTEMPTS, e)
    raise BriefingError(f"Could not generate a valid briefing after {MAX_ATTEMPTS} attempts.")


NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")


def _unverified_numbers(content: BriefingContent, paper_text: str) -> list[str]:
    """Numbers in the results and method bullets that do not appear in the paper text.
    A cheap hallucination check: flags invented, derived, or reformatted numbers."""
    source = paper_text.replace(",", "")
    claims = " ".join(content.key_results + content.method)
    missing = set()
    for number in NUMBER_RE.findall(claims):
        clean = number.replace(",", "")
        if "." not in clean and len(clean) < 3:   # skip small integers like 4, 8, 12
            continue
        if clean not in source:
            missing.add(number)
    return sorted(missing)

def make_briefing(paper: PaperMeta, parsed: ParsedPaper) -> Briefing:
    """Build the full briefing: LLM content + metadata from arXiv + parsing warnings."""
    paper_text, truncated = _paper_text(parsed, MAX_PAPER_CHARS)

    warnings = list(parsed.warnings)
    if truncated:
        warnings.append(f"Paper text exceeded {MAX_PAPER_CHARS:,} characters; "
                        "the end (usually appendix) was not included in the briefing.")
    if parsed.parse_quality == "abstract_only":
        warnings.append("The PDF could not be read; this briefing is based on the abstract only.")

    content = _generate_content(metadata_text(paper), paper_text)
    missing = _unverified_numbers(content, paper_text)
    if missing:
        warnings.append("Numbers not found verbatim in the paper text (verify manually): "
                        + ", ".join(missing))

    return Briefing(
        **content.model_dump(),
        title=paper.title,
        authors=paper.authors,
        arxiv_id=paper.arxiv_id,
        published=paper.published,
        link=paper.abs_url,
        parse_quality=parsed.parse_quality,
        warnings=warnings,
    )


def briefing_to_markdown(b: Briefing) -> str:
    """Render a briefing as Markdown."""
    def bullets(items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items)

    limitations = "\n".join(f"- {lim.text} *({lim.source})*" for lim in b.limitations)
    authors = ", ".join(b.authors[:6]) + (" et al." if len(b.authors) > 6 else "")
    md = [
        f"# {b.title}",
        f"**Authors:** {authors}  ",
        f"**arXiv:** [{b.arxiv_id}]({b.link}) · **Published:** {b.published}",
        "",
        "## Why it matters", b.why_it_matters, "",
        "## Problem", b.problem_statement, "",
        "## Method", bullets(b.method), "",
        "## Key results", bullets(b.key_results), "",
        "## Limitations", limitations, "",
        "## Suggested questions", bullets(b.suggested_questions),
    ]
    if b.warnings:
        md += ["", "## Warnings", bullets(b.warnings)]
    return "\n".join(md)