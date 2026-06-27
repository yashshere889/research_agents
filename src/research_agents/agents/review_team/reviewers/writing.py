from __future__ import annotations

from research_agents.agents.review_team.prompts import WRITING_PROMPT
from research_agents.agents.review_team.reviewers.base import BaseReviewer


class AcademicWritingReviewer(BaseReviewer):
    REVIEWER_ROLE = "Academic Writing Reviewer"

    @property
    def system_prompt(self) -> str:
        return WRITING_PROMPT

