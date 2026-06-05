from __future__ import annotations

from research_agents.agents.base_agent import BaseAgent
from research_agents.config import settings
from research_agents.models.review_models import ReviewerFeedback


class BaseReviewer(BaseAgent):
    MODEL = settings.reviewer_model
    REVIEWER_ROLE = "Reviewer"

    def run(self, state):
        """Reviewers are invoked through review(), not as pipeline stages."""
        raise NotImplementedError("Reviewer agents must be called with review().")

    def review(self, state, target_agent: str, target_output) -> ReviewerFeedback:
        hypothesis = state.research_plan.hypothesis if state.research_plan else "N/A"
        return self.call_llm(
            messages=[
                {
                    "role": "user",
                    "content": f"""Review the {target_agent} output.

Research Question: {state.research_question}
Plan Hypothesis: {hypothesis}

Output to Review:
{target_output}

Produce a ReviewerFeedback JSON. Use reviewer_role="{self.REVIEWER_ROLE}" and target_agent="{target_agent}".
Set approved=True only if this output is strong enough for the pipeline to proceed.""",
                }
            ],
            response_model=ReviewerFeedback,
        )
