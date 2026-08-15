"""Tests for the config layer.

These are not ceremonial. The config system is the load-bearing wall of the
whole results story: if fingerprinting is wrong, two different systems can write
to the same run directory and the ablation table silently lies. That deserves
tests from day one.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from kora.config import ExperimentConfig, load_config, save_resolved


def write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


def test_defaults_are_complete() -> None:
    """A config needs only a name; every other field has a sane default."""
    cfg = ExperimentConfig(name="smoke")
    assert cfg.retrieval.dense is True
    assert cfg.reranker.enabled is False
    assert cfg.generator.adapter_path is None


def test_unknown_keys_are_rejected() -> None:
    """A typo must fail loudly rather than default silently."""
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"name": "x", "retrieval": {"top_kk": 10}})


def test_config_is_frozen() -> None:
    """Configs must not mutate mid-run, or the fingerprint stops describing the run."""
    cfg = ExperimentConfig(name="smoke")
    with pytest.raises(ValidationError):
        cfg.retrieval.top_k = 99  # type: ignore[misc]


def test_fingerprint_is_stable_and_label_independent() -> None:
    """Renaming an experiment must not change its identity."""
    a = ExperimentConfig(name="alpha", description="first")
    b = ExperimentConfig(name="beta", description="second")
    assert a.fingerprint() == b.fingerprint()
    assert a.run_id != b.run_id  # run_id still distinguishes them by label


def test_fingerprint_changes_with_substance() -> None:
    a = ExperimentConfig(name="x")
    b = ExperimentConfig.model_validate({"name": "x", "reranker": {"enabled": True}})
    assert a.fingerprint() != b.fingerprint()


def test_index_fingerprint_ignores_downstream_settings() -> None:
    """Swapping the generator must not force an expensive index rebuild."""
    a = ExperimentConfig(name="x")
    b = ExperimentConfig.model_validate(
        {"name": "x", "generator": {"model_name": "some/other-model"}}
    )
    assert a.index_fingerprint() == b.index_fingerprint()
    assert a.fingerprint() != b.fingerprint()


def test_index_fingerprint_changes_with_chunking() -> None:
    a = ExperimentConfig(name="x")
    b = ExperimentConfig.model_validate({"name": "x", "chunking": {"strategy": "fixed"}})
    assert a.index_fingerprint() != b.index_fingerprint()


def test_extends_merges_deeply(tmp_path: Path) -> None:
    """An experiment file overrides one leaf without discarding its siblings."""
    write(
        tmp_path,
        "base.yaml",
        """
        name: base
        retrieval:
          dense: true
          bm25: false
          top_k: 20
          final_k: 5
        """,
    )
    child = write(
        tmp_path,
        "child.yaml",
        """
        extends: base.yaml
        name: child
        retrieval:
          bm25: true
        """,
    )

    cfg = load_config(child)
    assert cfg.name == "child"
    assert cfg.retrieval.bm25 is True
    assert cfg.retrieval.top_k == 20  # inherited, not reset to the class default
    assert cfg.retrieval.final_k == 5


def test_save_resolved_roundtrips(tmp_path: Path) -> None:
    cfg = ExperimentConfig(name="roundtrip")
    destination = save_resolved(cfg, tmp_path / "nested" / "config.resolved.yaml")
    assert destination.exists()
    assert load_config(destination).fingerprint() == cfg.fingerprint()


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "x", "chunking": {"max_tokens": 0}},
        {"name": "x", "retrieval": {"top_k": -1}},
        {"name": "x", "generator": {"temperature": 5.0}},
        {"name": "x", "generator": {"quantization": "3bit"}},
    ],
)
def test_invalid_values_rejected(payload: dict) -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(payload)


def test_shipped_configs_all_load() -> None:
    """Every config in configs/ must be valid -- guards against drift."""
    from kora import paths

    config_files = sorted(paths.CONFIG_DIR.rglob("*.yaml"))
    assert config_files, "no configs found"
    for path in config_files:
        cfg = load_config(path)
        assert cfg.name
