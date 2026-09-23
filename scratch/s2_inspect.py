# scratch/s2_inspect.py
import fitz  # PyMuPDF
import httpx

pdf_bytes = httpx.get("https://arxiv.org/pdf/1706.03762", follow_redirects=True, timeout=60).content
doc = fitz.open(stream=pdf_bytes, filetype="pdf")

print("Number of pages:", len(doc))

for page_no in [0, 2]:
    print(f"\n{'=' * 30} PAGE {page_no + 1} {'=' * 30}")
    print(doc[page_no].get_text()[:1500])

print(f"\n{'=' * 30} LINES THAT LOOK LIKE HEADINGS {'=' * 30}")
for page_no, page in enumerate(doc, start=1):
    for line in page.get_text().splitlines():
        stripped = line.strip()
        if stripped[:1].isdigit() and len(stripped) < 60:
            print(f"p.{page_no}: {stripped!r}")