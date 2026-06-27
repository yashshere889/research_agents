from __future__ import annotations

from loguru import logger

from research_agents.config import settings
from research_agents.models.agent_outputs import ArxivPaper


def search_arxiv(query: str, max_results: int = 20) -> list[ArxivPaper]:
    """Search arXiv and normalize results into ArxivPaper models."""
    try:
        import arxiv
    except ImportError as exc:
        raise RuntimeError("arxiv is not installed. Install requirements.txt first.") from exc

    client = arxiv.Client(
        page_size=min(max_results, 25),
        delay_seconds=settings.arxiv_delay_seconds,
        num_retries=0,
    )
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers: list[ArxivPaper] = []
    try:
        for result in client.results(search):
            arxiv_id = result.entry_id.rstrip("/").split("/")[-1]
            papers.append(
                ArxivPaper(
                    arxiv_id=arxiv_id,
                    title=result.title,
                    authors=[author.name for author in result.authors],
                    abstract=result.summary.replace("\n", " "),
                    url=result.pdf_url or result.entry_id,
                )
            )
    except arxiv.HTTPError as exc:
        if "429" in str(exc):
            logger.warning(
                "arXiv rate-limited this query; skipping it for this run. "
                f"Try again later or lower MAX_ARXIV_RESULTS. Query: {query}"
            )
            return []
        raise
    logger.info(f"Retrieved {len(papers)} arXiv papers for query: {query}")
    return papers
