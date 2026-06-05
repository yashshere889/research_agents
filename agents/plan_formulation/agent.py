from __future__ import annotations

from research_agents.agents.base_agent import BaseAgent
from research_agents.agents.plan_formulation.prompts import SYSTEM_PROMPT
from research_agents.models.agent_outputs import ResearchPlanOutput


class PlanFormulationAgent(BaseAgent):
    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def run(self, state) -> ResearchPlanOutput:
        if not state.literature_review:
            raise ValueError("Literature review must exist before plan formulation.")
        lr = state.literature_review
        return self.call_llm(
            messages=[
                {
                    "role": "user",
                    "content": f"""Research question: {state.research_question}

Literature review summary:
{lr.summary}

Research gaps identified:
{chr(10).join(f"- {gap}" for gap in lr.research_gaps)}

Suggested directions:
{chr(10).join(f"- {direction}" for direction in lr.suggested_directions)}

Formulate a rigorous research plan as a ResearchPlanOutput JSON object.""",
                }
            ],
            response_model=ResearchPlanOutput,
        )

