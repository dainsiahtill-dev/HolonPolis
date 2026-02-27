"""Central logging bootstrap inspired by Harborpilot conventions."""

from __future__ import annotations

import logging
from typing import Any

import structlog


_LOG_CONFIGURED = False


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """Configure stdlib logging and structlog once per process."""
    global _LOG_CONFIGURED
    if _LOG_CONFIGURED:
        return

    normalized = str(level or "INFO").upper()
    log_level = getattr(logging, normalized, logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(message)s",
    )

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    if json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _LOG_CONFIGURED = True

