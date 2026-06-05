from __future__ import annotations

import re

from research_agents.agents.base_agent import BaseAgent
from research_agents.agents.paper_writer.latex_builder import render_pdf
from research_agents.agents.paper_writer.prompts import SECTIONS, SYSTEM_PROMPT
from research_agents.models.agent_outputs import PaperOutput, PaperSection


class PaperWriterAgent(BaseAgent):
    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def run(self, state) -> PaperOutput:
        if not all([state.literature_review, state.research_plan, state.data_preparation, state.experiments]):
            raise ValueError("All prior pipeline outputs are required before paper writing.")

        sections: dict[str, str] = {}
        context = self._build_context(state)
        for section in SECTIONS:
            prior = "\n\n".join(f"## {title}\n{content}" for title, content in sections.items())
            sections[section] = self.call_llm(
                messages=[
                    {
                        "role": "user",
                        "content": f"""{context}

Previously written sections:
{prior if prior else "(none yet)"}

Now write the {section} section of the paper in academic Markdown.
Reference experiment IDs and metric values from the context. Be specific and concise.""",
                    }
                ]
            )

        full_markdown = "\n\n".join(f"## {title}\n{content}" for title, content in sections.items())
        references = self._generate_references(state.literature_review.papers_retrieved)
        if references:
            full_markdown += "\n\n## References\n" + "\n\n".join(references)

        paper = PaperOutput(
            title=self._extract_title(sections.get("Introduction", "")),
            authors=["Research Agent System"],
            abstract=sections.get("Abstract", ""),
            keywords=self._extract_keywords(sections),
            sections=[PaperSection(title=title, content=content) for title, content in sections.items()],
            references=references,
            full_markdown=full_markdown,
        )
        paper.pdf_path = render_pdf(paper)
        return paper

    def _build_context(self, state) -> str:
        return f"""RESEARCH CONTEXT FOR PAPER WRITING

Research Question: {state.research_question}
Hypothesis: {state.research_plan.hypothesis}

Key Literature Findings:
{state.literature_review.summary}

Research Plan:
Objectives: {state.research_plan.objectives}
Methodology: {state.research_plan.methodology}

Experimental Results:
{[run.model_dump() for run in state.experiments.runs]}
Best Run: {state.experiments.best_run_id}
Conclusions: {state.experiments.conclusions}

Data Used:
{[dataset.name for dataset in state.data_preparation.datasets]}"""

    def _generate_references(self, papers) -> list[str]:
        refs = []
        for paper in papers:
            year_match = re.match(r"(\d{4})", paper.arxiv_id)
            year = f"20{year_match.group(1)[:2]}" if year_match else "2024"
            refs.append(
                f"@article{{{paper.arxiv_id},\n"
                f"  title={{{paper.title}}},\n"
                f"  author={{{' and '.join(paper.authors)}}},\n"
                f"  journal={{arXiv preprint arXiv:{paper.arxiv_id}}},\n"
                f"  year={{{year}}}\n"
                "}"
            )
        return refs

    def _extract_title(self, introduction: str) -> str:
        for line in introduction.splitlines():
            candidate = line.strip("# ").strip()
            if candidate and len(candidate.split()) <= 18:
                return candidate
        return "Autonomous Scientific Research with Multi-Agent LLM Systems"

    def _extract_keywords(self, sections: dict[str, str]) -> list[str]:
        response = self.call_llm(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Extract 6 concise academic keywords from this paper draft. "
                        f"Return only a comma-separated list.\n\n{sections}"
                    ),
                }
            ]
        )
        return [keyword.strip() for keyword in response.split(",") if keyword.strip()][:8]

