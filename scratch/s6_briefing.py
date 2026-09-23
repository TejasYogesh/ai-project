"""Stage 6: generate executive briefings as JSON and Markdown."""
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.ERROR)

from backend.core.llm import init_llm
from backend.services.arxiv_client import search
from backend.services.pdf_parser import parse_pdf
from backend.services.summarizer import briefing_to_markdown, make_briefing

init_llm()
out_dir = Path("data/briefings")
out_dir.mkdir(parents=True, exist_ok=True)

for arxiv_id in ["1706.03762", "2607.01520"]:
    paper = search(id_list=arxiv_id)[0]
    parsed = parse_pdf(paper)

    start = time.time()
    briefing = make_briefing(paper, parsed)
    print(f"\n{'=' * 80}\nGenerated in {time.time() - start:.1f}s\n{'=' * 80}")

    markdown = briefing_to_markdown(briefing)
    print(markdown)

    (out_dir / f"{arxiv_id}.json").write_text(briefing.model_dump_json(indent=2), encoding="utf-8")
    (out_dir / f"{arxiv_id}.md").write_text(markdown, encoding="utf-8")
    print(f"\nSaved to {out_dir / arxiv_id}.json and .md")