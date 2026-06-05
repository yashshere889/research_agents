from __future__ import annotations

import json
import re

from research_agents.agents.base_agent import BaseAgent
from research_agents.agents.literature_review.prompts import SYSTEM_PROMPT
from research_agents.agents.literature_review.tools import search_arxiv
from research_agents.config import settings
from research_agents.models.agent_outputs import ArxivPaper, LiteratureReviewOutput
from research_agents.storage.vector_store import VectorStore


class LiteratureReviewAgent(BaseAgent):
    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def run(self, state) -> LiteratureReviewOutput:
        queries = self._generate_search_queries(state.research_question)[
            : settings.max_arxiv_queries
        ]
        all_papers: list[ArxivPaper] = []
        per_query = max(1, settings.max_arxiv_results // max(1, len(queries)))
        for query in queries:
            all_papers.extend(search_arxiv(query=query, max_results=per_query))

        unique_papers = list({paper.arxiv_id: paper for paper in all_papers}.values())
        if unique_papers:
            vs = VectorStore()
            vs.add_papers(unique_papers)
            relevant = vs.query(state.research_question, top_k=min(15, len(unique_papers)))
        else:
            relevant = []

        return self.call_llm(
            messages=[
                {
                    "role": "user",
                    "content": self._build_review_prompt(state.research_question, relevant),
                }
            ],
            response_model=LiteratureReviewOutput,
        )

    def _generate_search_queries(self, question: str) -> list[str]:
        response = self.call_llm(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Generate 5 diverse arXiv search queries for this research "
                        f"question: {question!r}. Return only a JSON list of strings."
                    ),
                }
            ]
        )
        try:
            queries = json.loads(self._extract_json_text(response))
        except (json.JSONDecodeError, ValueError):
            terms = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", question)
            compact = " ".join(terms[:8]) or question
            queries = [
                compact,
                f'all:"{compact}"',
                f'abs:"{compact}"',
                f'ti:"{compact}" OR abs:"{compact}"',
                f'cat:cs.AI AND all:"{compact}"',
            ]
        if not isinstance(queries, list) or not all(isinstance(q, str) for q in queries):
            raise ValueError("Search query generation must return a JSON list of strings.")
        return queries

    def _build_review_prompt(self, question: str, papers: list[ArxivPaper]) -> str:
        paper_texts = "\n\n".join(
            [
                f"[{p.arxiv_id}] {p.title}\nAuthors: {', '.join(p.authors)}\n"
                f"Abstract: {p.abstract}"
                for p in papers
            ]
        )
        return f"""Research question: {question}

Retrieved papers:
{paper_texts if paper_texts else "(No arXiv papers were retrieved. Note this limitation explicitly.)"}

Conduct a systematic literature review producing a LiteratureReviewOutput JSON."""
