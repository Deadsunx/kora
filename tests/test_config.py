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


def test_shipped_experiment_configs_all_load() -> None:
    """Every experiment config must be valid -- guards against drift.

    `configs/training/` is excluded: those are TrainingConfig, a different
    schema describing how an adapter is produced rather than how a system is
    evaluated. They are checked by test_shipped_training_configs_all_load.
    """
    from kora import paths

    config_files = [
        path for path in sorted(paths.CONFIG_DIR.rglob("*.yaml")) if "training" not in path.parts
    ]
    assert config_files, "no experiment configs found"
    for path in config_files:
        cfg = load_config(path)
        assert cfg.name


def test_shipped_training_configs_all_load() -> None:
    from kora import paths
    from kora.training.config import load_training_config

    config_files = sorted((paths.CONFIG_DIR / "training").glob("*.yaml"))
    assert config_files, "no training configs found"
    for path in config_files:
        cfg = load_training_config(path)
        assert cfg.name
        assert cfg.effective_batch_size > 0


# ---------------------------------------------------------------------------
# Fingerprint stability
#
# These pin the identity of runs whose numbers are published. They exist
# because that identity was found to have already drifted once: adding
# `max_length` and `precision` to RerankerConfig in Phase 3 re-hashed every
# config that existed before it, so nine of eleven recorded runs can no longer
# be regenerated -- see docs/reproducibility.md. The resolved config stored in
# each run directory still records exactly what ran, so nothing published is
# unverifiable; what broke is the config-file-to-run-id link.
#
# A test cannot undo that. It can make the next occurrence loud instead of
# silent, which is the whole reason it is here.
# ---------------------------------------------------------------------------

# run_id -> the config file that must continue to produce it.
PINNED_RUN_IDS = {
    "rerank-fast-54cefb1da669": "experiments/07_rerank_fast.yaml",
    "adapter-r16-6c121c457d35": "experiments/10_adapter.yaml",
}


def test_published_run_ids_are_stable() -> None:
    """The best retrieval system and the adapter must keep their identities.

    If this fails, a schema change has renamed a run that documents cite by id.
    Either revert the change, or exclude the new field from the fingerprint the
    way `agent` is excluded, or accept the rename and update every reference.
    """
    from kora import paths

    for expected_run_id, relative in PINNED_RUN_IDS.items():
        cfg = load_config(paths.CONFIG_DIR / relative)
        assert cfg.run_id == expected_run_id, (
            f"{relative} now hashes to {cfg.run_id!r}, not {expected_run_id!r}. "
            "A config field was added or a default changed."
        )


def test_disabled_agent_does_not_change_a_configs_identity() -> None:
    """Adding the agent must not rename systems that predate it.

    The narrow exception in `fingerprint()`. A subsystem that is switched off
    contributes nothing to behaviour, so it must not contribute to identity --
    otherwise every phase that adds a component renames every earlier run.
    """
    from kora.config import AgentConfig, ExperimentConfig

    plain = ExperimentConfig(name="x")
    explicitly_off = ExperimentConfig(name="x", agent=AgentConfig(enabled=False, decompose=False))

    # Off is off, whatever the other switches say.
    assert plain.fingerprint() == explicitly_off.fingerprint()

    enabled = ExperimentConfig(name="x", agent=AgentConfig(enabled=True))
    assert enabled.fingerprint() != plain.fingerprint()


def test_agent_settings_change_identity_once_enabled() -> None:
    """Two different agents must not share a run directory."""
    from kora.config import AgentConfig, ExperimentConfig

    decompose = ExperimentConfig(name="x", agent=AgentConfig(enabled=True, decompose=True))
    verify = ExperimentConfig(
        name="x", agent=AgentConfig(enabled=True, decompose=True, verify=True)
    )
    assert decompose.fingerprint() != verify.fingerprint()
