"""Download arXiv PDFs and split them into sections using PyMuPDF."""
import logging
import re
from pathlib import Path

import httpx
import pymupdf

from backend.core.exceptions import PDFDownloadError, PDFParseError
from backend.models.schemas import PaperMeta, ParsedPaper, Section

logger = logging.getLogger(__name__)

PDF_CACHE_DIR = Path("data/pdfs")
MAX_PAGES = 60                 # cap for huge papers
MIN_CHARS_PER_PAGE = 200       # below this, assume scanned / image-only
MIN_SECTIONS = 3               # below this, heading detection is considered failed

# Body headings: "3.1" + next line "Encoder ...", or "3.1 Encoder ..." on one line
SECTION_NUMBER_RE = re.compile(r"^\d{1,2}(\.\d{1,2}){0,2}$")
INLINE_HEADING_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,2})\.?\s+([A-Z].{2,70})$")
# NEW: appendix headings: "A" + next line "Proofs", or "B.1 Setup" on one line, or "Appendix ..."
APPENDIX_NUMBER_RE = re.compile(r"^[A-Z](\.\d{1,2}){0,2}$")
APPENDIX_INLINE_RE = re.compile(r"^([A-Z](?:\.\d{1,2}){0,2})\.?\s+([A-Z].{2,70})$")
APPENDIX_WORD_RE = re.compile(r"^(appendix|appendices)\b", re.IGNORECASE)

NAMED_HEADINGS = {"abstract", "introduction", "related work", "background", "conclusion",
                  "conclusions", "discussion", "limitations", "acknowledgments",
                  "acknowledgements", "appendix"}
REFERENCES_HEADINGS = {"references", "bibliography"}


def download_pdf(paper: PaperMeta) -> bytes:
    """Download a paper's PDF, caching it locally so each paper is fetched only once."""
    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = PDF_CACHE_DIR / f"{paper.arxiv_id}.pdf"
    if cache_file.exists():
        return cache_file.read_bytes()
    try:
        response = httpx.get(paper.pdf_url, follow_redirects=True, timeout=60)
        response.raise_for_status()
    except httpx.HTTPError as e:
        raise PDFDownloadError(f"Could not download {paper.pdf_url}: {e}") from e
    cache_file.write_bytes(response.content)
    return response.content


def _extract_lines(pages: list[str]) -> list[tuple[int, str]]:
    """Flatten pages into (page_no, line) pairs, dropping blank lines and page numbers."""
    lines: list[tuple[int, str]] = []
    for page_no, text in enumerate(pages, start=1):
        page_lines = [line.strip() for line in text.splitlines() if line.strip()]
        if page_lines and page_lines[-1] == str(page_no):   # page number at the bottom
            page_lines = page_lines[:-1]
        lines.extend((page_no, line) for line in page_lines)
    return lines


def _looks_like_title(text: str) -> bool:
    """Heuristic: short, starts with a capital, no trailing period, mostly letters."""
    text = text.strip()
    if not (3 <= len(text) <= 70) or not text[0].isupper():
        return False
    if text.endswith((".", ",", ":")):
        return False
    letters = sum(c.isalpha() or c.isspace() for c in text)
    return letters / len(text) > 0.8


def _label_tuple(label: str) -> tuple[int, ...]:
    """NEW: turn a heading label into comparable numbers.
    '3.2.1' -> (3, 2, 1), 'B.2' -> (2, 2) because A=1, B=2, ..."""
    head, *rest = label.split(".")
    first = ord(head) - ord("A") + 1 if head.isalpha() else int(head)
    return (first, *(int(part) for part in rest))


def _is_next_section(last: tuple[int, ...], new: tuple[int, ...]) -> bool:
    """True if `new` is a plausible heading number right after `last`.

    After 3.2, plausible next headings are 3.2.1, 3.3 or 4. We allow skipping
    one number (e.g. 3.4 or 5) in case a heading was missed, but not jumps
    like 6.3 -> 91.7, which come from table values.
    """
    if len(new) == 1:                                   # top level: 4 after 3.x
        return 1 <= new[0] - last[0] <= 2
    parent = new[:-1]
    if parent == last[:len(parent)]:                    # child or sibling
        previous = last[len(parent)] if len(last) > len(parent) else 0
        return 1 <= new[-1] - previous <= 2
    # parent heading was missed, e.g. 4.1 right after 3.2: accept if 4 would be valid
    return new[-1] == 1 and _is_next_section(last, parent)


def _detect_heading(lines: list[tuple[int, str]], i: int, last: tuple[int, ...],
                    appendix: bool) -> tuple[str, int, tuple[int, ...]] | None:
    """NEW: check whether lines[i] starts a numbered (or lettered) heading.
    Returns (heading_text, lines_consumed, heading_number) or None."""
    line = lines[i][1]
    next_line = lines[i + 1][1] if i + 1 < len(lines) else ""
    number_re, inline_re = ((APPENDIX_NUMBER_RE, APPENDIX_INLINE_RE) if appendix
                            else (SECTION_NUMBER_RE, INLINE_HEADING_RE))

    # two-line form: "3.1" then "Encoder and Decoder Stacks"
    if number_re.match(line) and _looks_like_title(next_line):
        number = _label_tuple(line)
        if _is_next_section(last, number):
            return f"{line} {next_line}", 2, number

    # one-line form: "3.1 Encoder and Decoder Stacks"
    match = inline_re.match(line)
    if match and _looks_like_title(match.group(2)):
        number = _label_tuple(match.group(1))
        if _is_next_section(last, number):
            return line, 1, number
    return None

def _split_sections(lines: list[tuple[int, str]]) -> tuple[list[Section], str]:
    """Walk the lines, starting a new section at each detected heading.
    Text after 'References' goes into the references string, until an appendix starts."""
    sections: list[Section] = []
    title, buffer, start_page = "Front matter", [], 1
    references: list[str] = []
    in_references = False
    in_appendix = False
    appendix_start = 0              # NEW: index in `sections` where the appendix begins
    first_appendix_title = ""       # NEW: used to detect an appendix table of contents
    last_number: tuple[int, ...] = (0,)

    def close_section() -> None:
        if buffer:
            sections.append(Section(title=title, text="\n".join(buffer), page_start=start_page))

    i = 0
    while i < len(lines):
        page_no, line = lines[i]
        lower = line.lower().rstrip(":")

        # --- inside the reference list: collect lines, but watch for an appendix ---
        if in_references:
            found = None
            if lines[i - 1][1].endswith("."):             # previous reference entry has ended
                if APPENDIX_WORD_RE.match(line):
                    found = (line, 1, (0,))
                else:
                    found = _detect_heading(lines, i, (0,), appendix=True)
            if found:                                     # appendix starts, resume sections
                title, consumed, last_number = found
                buffer, start_page = [], page_no
                in_references, in_appendix = False, True
                appendix_start = len(sections)            # NEW
                first_appendix_title = title              # NEW
                i += consumed
                continue
            references.append(line)
            i += 1
            continue

        # --- start of the reference list ---
        if lower in REFERENCES_HEADINGS and not in_appendix:
            close_section()
            buffer = []                                   # prevents duplicate last section
            in_references = True
            i += 1
            continue

        # --- normal text: is this line a heading? ---
        heading, consumed = None, 1
        if lower in NAMED_HEADINGS:
            heading = line
        else:
            found = _detect_heading(lines, i, last_number, appendix=in_appendix)
            if found:
                heading, consumed, last_number = found

        # NEW: the first appendix heading appears a second time, so what we saw
        # before was a table of contents. Drop those sections and restart at "A".
        if heading is None and in_appendix:
            restart = _detect_heading(lines, i, (0,), appendix=True)
            if restart and restart[0] == first_appendix_title:
                del sections[appendix_start:]             # remove table-of-contents sections
                buffer = []                               # and the TOC text still in the buffer
                heading, consumed, last_number = restart

        if heading:
            close_section()
            title, buffer, start_page = heading, [], page_no
        else:
            buffer.append(line)
        i += consumed

    close_section()
    return sections, "\n".join(references)

def parse_pdf(paper: PaperMeta) -> ParsedPaper:
    """Download and parse a paper into sections, with fallbacks for problem PDFs."""
    pdf_bytes = download_pdf(paper)
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise PDFParseError(f"Could not open PDF for {paper.arxiv_id}: {e}") from e

    warnings: list[str] = []
    num_pages = len(doc)
    if num_pages > MAX_PAGES:
        warnings.append(f"Paper has {num_pages} pages; only the first {MAX_PAGES} were parsed.")
    pages = [doc[i].get_text() for i in range(min(num_pages, MAX_PAGES))]

    # Fallback 1: scanned / image-only PDF -> abstract only
    avg_chars = sum(len(p) for p in pages) / max(len(pages), 1)
    if avg_chars < MIN_CHARS_PER_PAGE:
        warnings.append("PDF has little extractable text (possibly scanned); using abstract only.")
        return ParsedPaper(arxiv_id=paper.arxiv_id, num_pages=num_pages,
                           parse_quality="abstract_only", warnings=warnings,
                           sections=[Section(title="Abstract", text=paper.abstract, page_start=1)])

    sections, references = _split_sections(_extract_lines(pages))

    # Fallback 2: heading detection failed -> one section per page
    if len(sections) < MIN_SECTIONS:
        warnings.append("Could not detect section headings; splitting by page instead.")
        sections = [Section(title=f"Page {i}", text=text.strip(), page_start=i)
                    for i, text in enumerate(pages, start=1) if text.strip()]
        return ParsedPaper(arxiv_id=paper.arxiv_id, sections=sections, num_pages=num_pages,
                           parse_quality="partial", warnings=warnings)

    if not references:
        warnings.append("No References section found.")
    return ParsedPaper(arxiv_id=paper.arxiv_id, sections=sections, references=references,
                       num_pages=num_pages, parse_quality="full", warnings=warnings)