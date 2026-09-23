import re

import pymupdf

from backend.services.pdf_parser import _extract_lines

doc = pymupdf.open("data/pdfs/2607.01520.pdf")
lines = _extract_lines([page.get_text() for page in doc])

# lines that start with "Appendix" or a letter label like A, B.1, C.2
pattern = re.compile(r"^(Appendix\b|[A-E](\.\d+)*\b)")

for idx, (page, line) in enumerate(lines):
    if 12 <= page <= 16 and pattern.match(line):
        next_line = lines[idx + 1][1] if idx + 1 < len(lines) else ""
        print(f"p.{page}: {line!r:60}  next: {next_line!r}")