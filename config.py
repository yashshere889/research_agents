from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    llm_provider: str = "gemini"
    gemini_api_key: str = ""
    hf_token: str = ""
    e2b_api_key: str = ""
    chroma_persist_dir: Path = Path("./storage/chroma")
    outputs_dir: Path = Path("./outputs")
    max_arxiv_results: int = 30
    max_arxiv_queries: int = 2
    arxiv_delay_seconds: float = 10.0
    reviewer_approval_threshold: float = 7.0
    max_revision_cycles: int = 3
    log_level: str = "INFO"
    embedding_model: str = "all-MiniLM-L6-v2"
    agent_model: str = "gemini-2.5-pro"
    reviewer_model: str = "gemini-2.5-flash"
    max_output_tokens: int = Field(default=8192, ge=1)
    hf_load_in_4bit: bool = False
    hf_torch_dtype: str = "auto"
    hf_attn_implementation: str = "auto"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
