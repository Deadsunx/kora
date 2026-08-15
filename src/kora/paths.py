"""Canonical filesystem layout.

Every path in the project resolves through here. The alternative -- relative
paths scattered across scripts -- breaks the moment you run something from a
different working directory, which you will, constantly (notebooks, tests, CI).
"""

from __future__ import annotations

import os
from pathlib import Path

# `paths.py` lives at <root>/src/kora/paths.py, so the root is three levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve(env_var: str, default: Path) -> Path:
    """Allow any location to be redirected by an environment variable.

    Useful when the corpus or model cache lives on a different drive -- a real
    concern on a laptop where the SSD fills up fast with model weights.
    """
    value = os.environ.get(env_var)
    return Path(value).expanduser().resolve() if value else default


DATA_DIR = _resolve("KORA_DATA_DIR", PROJECT_ROOT / "data")

RAW_DIR = DATA_DIR / "raw"  # Untouched source documents, as downloaded.
INTERIM_DIR = DATA_DIR / "interim"  # Parsed text, pre-chunking.
PROCESSED_DIR = DATA_DIR / "processed"  # Chunks ready for indexing.
INDEX_DIR = DATA_DIR / "indexes"  # Vector and lexical indexes, keyed by fingerprint.
EVAL_DIR = DATA_DIR / "eval"  # Gold QA sets.

MODELS_DIR = _resolve("KORA_MODELS_DIR", PROJECT_ROOT / "models")
RUNS_DIR = _resolve("KORA_RUNS_DIR", PROJECT_ROOT / "runs")
CONFIG_DIR = PROJECT_ROOT / "configs"

ALL_DIRS = (
    RAW_DIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    INDEX_DIR,
    EVAL_DIR,
    MODELS_DIR,
    RUNS_DIR,
)


def ensure_dirs() -> None:
    """Create every project directory. Idempotent; safe to call at startup."""
    for directory in ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def index_path(fingerprint: str) -> Path:
    """Directory holding the index built from a given chunk+embedding config.

    Keyed by fingerprint rather than by name so that two experiments sharing the
    same ingestion settings transparently share one index instead of rebuilding
    it -- and two that differ can never collide.
    """
    return INDEX_DIR / fingerprint


def run_path(run_id: str) -> Path:
    """Directory holding one experiment's outputs (metrics, predictions, config)."""
    return RUNS_DIR / run_id
