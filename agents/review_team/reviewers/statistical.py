from __future__ import annotations

from research_agents.agents.review_team.prompts import STATISTICAL_PROMPT
from research_agents.agents.review_team.reviewers.base import BaseReviewer


class StatisticalReviewer(BaseReviewer):
    REVIEWER_ROLE = "Statistical Reviewer"

    @property
    def system_prompt(self) -> str:
        return STATISTICAL_PROMPT

