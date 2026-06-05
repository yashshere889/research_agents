from __future__ import annotations

import json
import re


def parse_metrics(stdout: str) -> dict[str, float]:
    """Extract the last flat JSON metrics object from stdout."""
    matches = re.findall(r"\{[^{}]+\}", stdout)
    for match in reversed(matches):
        try:
            parsed = json.loads(match)
        except json.JSONDecodeError:
            continue
        metrics: dict[str, float] = {}
        for key, value in parsed.items():
            if isinstance(value, int | float):
                metrics[key] = float(value)
        if metrics:
            return metrics
    return {}

