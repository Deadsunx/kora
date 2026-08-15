"""Structured logging.

Plain `print` is fine until the first long training run, at which point you need
to answer "what were the retrieval settings when this warning fired?" from a log
file recorded three hours ago. Structured logs carry key/value context, so they
stay greppable and machine-readable while remaining pleasant to read in a
terminal.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import structlog


def configure_logging(
    level: str = "INFO",
    *,
    log_file: str | Path | None = None,
    json_output: bool = False,
) -> None:
    """Set up structlog once, at process start.

    Args:
        level: Standard logging level name.
        log_file: If given, JSON lines are also appended here. Long runs should
            always set this -- terminal scrollback is not a record.
        json_output: Emit JSON to stdout instead of the human-readable console
            renderer. Set this in CI and containers, where a log collector reads
            stdout and pretty colours are noise.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=handlers,
        force=True,
    )

    # Third-party libraries are chatty during model loading; quiet them down so
    # our own messages remain visible.
    for noisy in ("urllib3", "httpx", "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a logger bound to `name`.

    Usage:
        log = get_logger(__name__)
        log.info("indexed corpus", chunks=12_843, seconds=41.2)
    """
    return structlog.get_logger(name)
