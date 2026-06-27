from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from .agent_outputs import (
    DataPreparationOutput,
    ExperimentOutput,
    LiteratureReviewOutput,
    PaperOutput,
    ResearchPlanOutput,
)
from .review_models import ReviewCycle


def _session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ResearchState(BaseModel):
    """Single shared truth object passed between all agents."""

    research_question: str
    session_id: str = Field(default_factory=_session_id)
    created_at: datetime = Field(default_factory=_utc_now)

    literature_review: LiteratureReviewOutput | None = None
    research_plan: ResearchPlanOutput | None = None
    data_preparation: DataPreparationOutput | None = None
    experiments: ExperimentOutput | None = None
    paper: PaperOutput | None = None

    review_cycles: list[ReviewCycle] = Field(default_factory=list)
    total_tokens_used: int = 0
    pipeline_complete: bool = False

