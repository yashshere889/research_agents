from __future__ import annotations

from loguru import logger

from research_agents.agents.data_preparation.agent import DataPreparationAgent
from research_agents.agents.experiment_runner.agent import ExperimentRunnerAgent
from research_agents.agents.literature_review.agent import LiteratureReviewAgent
from research_agents.agents.paper_writer.agent import PaperWriterAgent
from research_agents.agents.plan_formulation.agent import PlanFormulationAgent
from research_agents.agents.review_team.coordinator import ReviewTeamCoordinator
from research_agents.config import settings
from research_agents.models.research_state import ResearchState
from research_agents.storage.state_manager import StateManager


class ResearchOrchestrator:
    """Central workflow coordinator for the full research pipeline."""

    def __init__(self, research_question: str) -> None:
        self.state = ResearchState(research_question=research_question)
        self.state_mgr = StateManager()
        self.reviewer = ReviewTeamCoordinator()
        self.pipeline = [
            ("Literature Review", LiteratureReviewAgent, "literature_review"),
            ("Plan Formulation", PlanFormulationAgent, "research_plan"),
            ("Data Preparation", DataPreparationAgent, "data_preparation"),
            ("Experiment Running", ExperimentRunnerAgent, "experiments"),
            ("Paper Writing", PaperWriterAgent, "paper"),
        ]

    def run(self) -> ResearchState:
        logger.info(f"Starting research pipeline for: {self.state.research_question}")

        for step_name, agent_class, state_attr in self.pipeline:
            logger.info(f"Starting step: {step_name}")
            agent = agent_class()
            for cycle in range(1, settings.max_revision_cycles + 1):
                output = agent.run(self.state)
                setattr(self.state, state_attr, output)
                self.state.total_tokens_used += getattr(agent, "total_tokens", 0)
                self.state_mgr.save(self.state)

                logger.info(f"Submitting {step_name} to review team, cycle {cycle}")
                review = self.reviewer.review(self.state, step_name, output, cycle)
                self.state.review_cycles.append(review)
                self.state_mgr.save(self.state)

                if (
                    review.consensus_approved
                    or review.aggregate_score >= settings.reviewer_approval_threshold
                ):
                    logger.success(
                        f"{step_name} approved with score {review.aggregate_score:.1f}"
                    )
                    break

                logger.warning(review.revision_instructions[:500])
                agent.revision_instructions = review.revision_instructions
            else:
                logger.warning(f"Max revisions reached for {step_name}; proceeding anyway.")

        self.state.pipeline_complete = True
        self.state_mgr.save(self.state)
        if self.state.paper:
            logger.success(f"Pipeline complete. Paper: {self.state.paper.pdf_path}")
        return self.state

