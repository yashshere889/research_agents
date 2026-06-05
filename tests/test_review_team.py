from __future__ import annotations

from research_agents.agents.review_team.coordinator import ReviewTeamCoordinator
from research_agents.models.review_models import ReviewerFeedback


def test_synthesise_revision_instructions() -> None:
    coordinator = ReviewTeamCoordinator.__new__(ReviewTeamCoordinator)
    feedback = ReviewerFeedback(
        reviewer_role="Methodology Reviewer",
        target_agent="Plan",
        score=5.0,
        strengths=[],
        weaknesses=["missing baseline"],
        required_revisions=["Add a baseline"],
        optional_suggestions=[],
        approved=False,
    )
    text = coordinator._synthesise_instructions([feedback], approved=False)
    assert "Add a baseline" in text

