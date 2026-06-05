from __future__ import annotations

from collections.abc import Iterable

from research_agents.config import settings
from research_agents.models.agent_outputs import ArxivPaper
from research_agents.tools.embedding import embed_texts


class VectorStore:
    """Small ChromaDB-backed vector store for retrieved papers."""

    def __init__(self, collection_name: str = "papers") -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("chromadb is not installed. Install requirements.txt first.") from exc

        settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
        self.collection = self.client.get_or_create_collection(collection_name)
        self._paper_cache: dict[str, ArxivPaper] = {}

    def add_papers(self, papers: Iterable[ArxivPaper]) -> None:
        unique = list({paper.arxiv_id: paper for paper in papers}.values())
        if not unique:
            return
        docs = [f"{p.title}\n\n{p.abstract}" for p in unique]
        embeddings = embed_texts(docs)
        self.collection.upsert(
            ids=[p.arxiv_id for p in unique],
            documents=docs,
            embeddings=embeddings,
            metadatas=[p.model_dump(mode="json") for p in unique],
        )
        self._paper_cache.update({p.arxiv_id: p for p in unique})

    def query(self, text: str, top_k: int = 15) -> list[ArxivPaper]:
        embedding = embed_texts([text])[0]
        results = self.collection.query(query_embeddings=[embedding], n_results=top_k)
        metadatas = results.get("metadatas", [[]])[0]
        return [ArxivPaper.model_validate(metadata) for metadata in metadatas if metadata]

