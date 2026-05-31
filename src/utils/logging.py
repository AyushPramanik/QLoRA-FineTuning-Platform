"""Structured logging configuration using structlog.

All platform components should obtain loggers through ``get_logger`` rather
than using the standard ``logging`` module directly.  This ensures consistent
JSON-formatted output in production and human-readable output in development.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import structlog


def _configure_stdlib_logging(level: str) -> None:
    """Configure the stdlib logging root handler."""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )


def configure_logging(
    level: str | None = None,
    fmt: str | None = None,
    *,
    force: bool = False,
) -> None:
    """Configure the global structlog pipeline.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Defaults to the ``LOG_LEVEL`` environment variable or INFO.
        fmt:   Output format. One of ``"json"`` or ``"text"``.
               Defaults to the ``LOG_FORMAT`` environment variable or ``"text"``.
        force: Re-configure even if already configured.
    """
    # TODO: Add support for per-module log levels.
    # TODO: Integrate with OpenTelemetry trace/span context injection.

    if structlog.is_configured() and not force:
        return

    resolved_level = level or os.getenv("LOG_LEVEL", "INFO")
    resolved_fmt   = fmt   or os.getenv("LOG_FORMAT", "text")

    _configure_stdlib_logging(resolved_level)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if resolved_fmt == "json":
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, resolved_level.upper(), logging.INFO))


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger.

    Args:
        name: Logger name (typically ``__name__``).

    Returns:
        A configured ``BoundLogger`` instance.
    """
    configure_logging()
    return structlog.get_logger(name)
