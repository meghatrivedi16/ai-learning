"""
02_chunkingSplitters/MarkdownHeaderTextSplitter.py


Install:
    pip install pymupdf4llm langchain-text-splitters
"""

import os
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter

# ------------------------------------------------------------
# 1. Build an absolute path from the script's own location.
#    This works no matter which directory you run `python` from.
# ------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(SCRIPT_DIR, "Warranty_Returns_Policy_LuxeThreads.pdf")

# Fail loudly and clearly if the file genuinely isn't there --
# better than a stack trace three layers deep in a library.
if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(
        f"No PDF found at: {PDF_PATH}\n"
        f"Make sure '{os.path.basename(PDF_PATH)}' is in the same folder "
        f"as this script: {SCRIPT_DIR}"
    )

print(f"Loading PDF from: {PDF_PATH}")

# ------------------------------------------------------------
# 2. Convert PDF -> markdown (this is the step that actually
#    produces '#' / '##' headers for the splitter to use).
# ------------------------------------------------------------

md_text = pymupdf4llm.to_markdown(PDF_PATH)
print(f"Converted to markdown ({len(md_text)} chars).")

# ------------------------------------------------------------
# 3. Split on markdown headers.
# ------------------------------------------------------------

header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "H1"), ("##", "H2")]
)
header_chunks = header_splitter.split_text(md_text)

print(f"\n--- {len(header_chunks)} chunks after header split ---")
for c in header_chunks:
    print(f"metadata: {c.metadata}")
    print(f"text: {c.page_content[:80]}...\n")