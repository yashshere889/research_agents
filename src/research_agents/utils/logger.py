from __future__ import annotations

import sys

from loguru import logger

from research_agents.config import settings


def setup_logger() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        colorize=True,
        backtrace=True,
        diagnose=False,
    )
