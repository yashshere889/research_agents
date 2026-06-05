from __future__ import annotations

import re
from pathlib import Path

from research_agents.config import settings
from research_agents.models.agent_outputs import PaperOutput
from research_agents.tools.pdf_renderer import render_markdown_pdf


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return slug[:80] or "research_paper"


def render_pdf(paper: PaperOutput) -> str:
    output_dir = settings.outputs_dir / "papers"
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"{_slug(paper.title)}.md"
    pdf_path = output_dir / f"{_slug(paper.title)}.pdf"
    md_path.write_text(paper.full_markdown, encoding="utf-8")
    render_markdown_pdf(paper.full_markdown, pdf_path)
    return str(pdf_path)

