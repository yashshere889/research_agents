from __future__ import annotations

from research_agents.models.agent_outputs import LiteratureReviewOutput
from research_agents.models.research_state import ResearchState


def test_research_state_round_trip() -> None:
    state = ResearchState(research_question="Can agents automate research?")
    restored = ResearchState.model_validate_json(state.model_dump_json())
    assert restored.research_question == state.research_question
    assert restored.session_id == state.session_id


def test_literature_review_output_schema() -> None:
    output = LiteratureReviewOutput(
        papers_retrieved=[],
        research_gaps=["gap"],
        existing_methods=["method"],
        summary="summary",
        suggested_directions=["direction"],
    )
    assert output.research_gaps == ["gap"]

