from __future__ import annotations

import re
from pathlib import Path


def _markdown_to_plain_text(markdown: str) -> str:
    try:
        import markdown2
        from bs4 import BeautifulSoup

        html = markdown2.markdown(markdown)
        return BeautifulSoup(html, "html.parser").get_text("\n")
    except Exception:
        text = re.sub(r"^#+\s*", "", markdown, flags=re.MULTILINE)
        return re.sub(r"[*_`]", "", text)


def render_markdown_pdf(markdown: str, pdf_path: Path) -> None:
    try:
        from fpdf import FPDF
    except ImportError as exc:
        raise RuntimeError("fpdf2 is not installed. Install requirements.txt first.") from exc

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    for line in _markdown_to_plain_text(markdown).splitlines():
        clean = line.encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 6, clean)
    pdf.output(str(pdf_path))

