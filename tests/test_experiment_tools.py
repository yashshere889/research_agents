from __future__ import annotations

from research_agents.agents.experiment_runner.tools import parse_metrics


def test_parse_metrics_uses_last_json_object() -> None:
    stdout = 'warmup {"loss": 1.2}\nfinal {"accuracy": 0.91, "f1": 0.89}'
    assert parse_metrics(stdout) == {"accuracy": 0.91, "f1": 0.89}


def test_parse_metrics_returns_empty_for_missing_json() -> None:
    assert parse_metrics("no metrics here") == {}

