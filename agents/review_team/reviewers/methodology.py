from __future__ import annotations

from research_agents.agents.review_team.prompts import METHODOLOGY_PROMPT
from research_agents.agents.review_team.reviewers.base import BaseReviewer


class MethodologyReviewer(BaseReviewer):
    REVIEWER_ROLE = "Methodology Reviewer"

    @property
    def system_prompt(self) -> str:
        return METHODOLOGY_PROMPT

