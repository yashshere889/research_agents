from __future__ import annotations

from functools import lru_cache

from research_agents.config import settings


@lru_cache(maxsize=1)
def _embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. Install requirements.txt first."
        ) from exc
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = _embedding_model()
    return model.encode(texts, normalize_embeddings=True).tolist()

