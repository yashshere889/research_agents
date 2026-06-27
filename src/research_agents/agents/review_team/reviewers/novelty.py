from __future__ import annotations

from research_agents.agents.review_team.prompts import NOVELTY_PROMPT
from research_agents.agents.review_team.reviewers.base import BaseReviewer


class NoveltyReviewer(BaseReviewer):
    REVIEWER_ROLE = "Novelty Reviewer"

    @property
    def system_prompt(self) -> str:
        return NOVELTY_PROMPT

