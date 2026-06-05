from __future__ import annotations

from pydantic import BaseModel, Field


class ReviewerFeedback(BaseModel):
    reviewer_role: str
    target_agent: str
    score: float = Field(ge=0.0, le=10.0)
    strengths: list[str]
    weaknesses: list[str]
    required_revisions: list[str]
    optional_suggestions: list[str]
    approved: bool


class ReviewCycle(BaseModel):
    cycle_number: int = Field(ge=1)
    target_agent: str
    feedbacks: list[ReviewerFeedback]
    aggregate_score: float = Field(ge=0.0, le=10.0)
    consensus_approved: bool
    revision_instructions: str

