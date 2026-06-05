from __future__ import annotations

from pathlib import Path

from research_agents.config import settings
from research_agents.models.research_state import ResearchState


class StateManager:
    """Persist and load ResearchState snapshots as JSON."""

    def __init__(self, state_dir: Path | None = None) -> None:
        self.state_dir = state_dir or settings.outputs_dir / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def save(self, state: ResearchState) -> Path:
        path = self.state_dir / f"{state.session_id}.json"
        path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        return path

    def load(self, session_id: str) -> ResearchState:
        path = self.state_dir / f"{session_id}.json"
        return ResearchState.model_validate_json(path.read_text(encoding="utf-8"))

