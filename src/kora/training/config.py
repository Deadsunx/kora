"""Configuration for QLoRA fine-tuning.

Separate from `ExperimentConfig` on purpose. A training run produces an
artefact -- an adapter on disk -- and an experiment *consumes* one via
`generator.adapter_path`. Folding training hyperparameters into the experiment
config would mean every evaluation carried settings that had no effect on it,
and changing a learning rate would invalidate the fingerprint of runs that never
trained anything.

Content-hashed on the same principle as the experiment config, so an adapter
directory is named for the settings that produced it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LoraConfig(_Base):
    """Low-rank adapter shape."""

    # r=16 with alpha=32 is the common starting point: alpha/r = 2 keeps the
    # effective update scale moderate. Both are exposed because rank is the
    # first thing worth ablating if the adapter underfits or overfits.
    r: int = Field(16, gt=0)
    alpha: int = Field(32, gt=0)
    dropout: float = Field(0.05, ge=0.0, lt=1.0)

    # Attention *and* MLP projections. Attention-only adapters are cheaper and
    # were the original LoRA recipe, but the QLoRA paper found targeting every
    # linear layer matters more than rank for matching full fine-tuning. The
    # MLP is also where a model's output style lives, and style -- citation
    # format, abstention, concision -- is precisely what is being taught here.
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


class TrainingConfig(_Base):
    """One fine-tuning run."""

    name: str
    description: str = ""

    base_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    dataset_path: str = "data/training/sft.jsonl"

    lora: LoraConfig = Field(default_factory=LoraConfig)

    # -- optimisation -------------------------------------------------------

    epochs: float = Field(2.0, gt=0)
    learning_rate: float = Field(2e-4, gt=0)
    lr_scheduler: Literal["cosine", "linear", "constant"] = "cosine"
    warmup_ratio: float = Field(0.03, ge=0.0, lt=1.0)
    weight_decay: float = Field(0.0, ge=0.0)
    max_grad_norm: float = Field(0.3, gt=0)

    # -- memory -------------------------------------------------------------
    #
    # These four exist because of the 8 GiB budget, not because they are
    # universally right. Batch size 1 with accumulation 8 gives an effective
    # batch of 8 while only ever holding one sequence of activations.

    batch_size: int = Field(1, gt=0)
    gradient_accumulation: int = Field(8, gt=0)
    max_length: int = Field(2048, gt=0)
    gradient_checkpointing: bool = True

    # Paged 8-bit AdamW: optimizer state at a quarter the size, and the paged
    # variant survives a transient spike by moving state to host memory rather
    # than dying with an out-of-memory error partway through an epoch.
    optimizer: str = "paged_adamw_8bit"

    seed: int = 42
    eval_fraction: float = Field(0.05, ge=0.0, lt=0.5)

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation

    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json", exclude={"name", "description"})
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]

    @property
    def run_id(self) -> str:
        return f"{self.name}-{self.fingerprint()}"


def load_training_config(path: str | Path) -> TrainingConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle) or {}
    return TrainingConfig.model_validate(data)
