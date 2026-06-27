from __future__ import annotations

import asyncio

from research_agents.agents.review_team.reviewers.methodology import MethodologyReviewer
from research_agents.agents.review_team.reviewers.novelty import NoveltyReviewer
from research_agents.agents.review_team.reviewers.statistical import StatisticalReviewer
from research_agents.agents.review_team.reviewers.writing import AcademicWritingReviewer
from research_agents.models.review_models import ReviewCycle, ReviewerFeedback

REVIEWERS = [
    MethodologyReviewer,
    StatisticalReviewer,
    NoveltyReviewer,
    AcademicWritingReviewer,
]


class ReviewTeamCoordinator:
    """Orchestrate parallel reviewer agents and synthesize consensus feedback."""

    def __init__(self) -> None:
        self.reviewers = [reviewer() for reviewer in REVIEWERS]

    async def review_async(
        self,
        state,
        target_agent: str,
        target_output,
        cycle_number: int,
    ) -> ReviewCycle:
        tasks = [
            asyncio.to_thread(reviewer.review, state, target_agent, target_output)
            for reviewer in self.reviewers
        ]
        feedbacks: list[ReviewerFeedback] = await asyncio.gather(*tasks)
        aggregate = sum(feedback.score for feedback in feedbacks) / len(feedbacks)
        approved = all(feedback.approved for feedback in feedbacks)
        return ReviewCycle(
            cycle_number=cycle_number,
            target_agent=target_agent,
            feedbacks=feedbacks,
            aggregate_score=aggregate,
            consensus_approved=approved,
            revision_instructions=self._synthesise_instructions(feedbacks, approved),
        )

    def review(self, state, target_agent: str, target_output, cycle_number: int) -> ReviewCycle:
        return asyncio.run(self.review_async(state, target_agent, target_output, cycle_number))

    def _synthesise_instructions(
        self,
        feedbacks: list[ReviewerFeedback],
        approved: bool,
    ) -> str:
        if approved:
            return "All reviewers approved. Proceed to next stage."
        required = [
            f"[{feedback.reviewer_role}] {revision}"
            for feedback in feedbacks
            for revision in feedback.required_revisions
        ]
        if not required:
            required = [
                f"[{feedback.reviewer_role}] Improve weak areas: {', '.join(feedback.weaknesses)}"
                for feedback in feedbacks
                if feedback.weaknesses
            ]
        return "Required revisions before proceeding:\n" + "\n".join(required)

