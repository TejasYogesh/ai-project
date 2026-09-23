from backend.services.arxiv_client import search
from backend.services.pdf_parser import parse_pdf

for arxiv_id in ["1706.03762", "2607.01520"]:
    paper = search(id_list=arxiv_id)[0]
    parsed = parse_pdf(paper)
    print(f"\n=== {paper.title} ===")
    print(f"pages={parsed.num_pages} quality={parsed.parse_quality} warnings={parsed.warnings}")
    for s in parsed.sections:
        print(f"  p.{s.page_start:>2} | {len(s.text):>6} chars | {s.title}")
    print(f"  references: {len(parsed.references)} chars")
    print(f"  references start: {parsed.references[:150]!r}")