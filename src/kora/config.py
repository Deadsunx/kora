"""Typed, hashable experiment configuration.

Why this file exists
--------------------
The whole point of this project is to make claims like "reranking improved
recall@5 by 14 points". A claim like that is only trustworthy if you can say
*exactly* which system produced each number, and reproduce it months later.

So the design rule is: **an experiment is fully described by its config.**
Nothing that affects a result may live in a notebook cell, a CLI flag we forget,
or a hardcoded constant. If it changes the output, it belongs in here.

Two consequences follow:

1. Configs are *typed* (Pydantic models), not dicts. A typo in a YAML key raises
   an error at load time instead of silently using a default and quietly
   invalidating an entire results table.
2. Configs are *content-hashed*. The `run_id` is derived from the config itself,
   so identical settings always map to the same run directory, and any change --
   however small -- produces a new one. You cannot accidentally overwrite the
   results of a different system.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    """Shared behaviour: reject unknown keys, freeze after construction.

    `extra="forbid"` is the important one. Without it, a YAML file containing
    `top_k: 10` under a section that expects `top_n` would load fine, use the
    default for `top_n`, and produce a result table that silently answers a
    different question than the one you thought you asked.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# Stage configs
# ---------------------------------------------------------------------------


class ChunkingConfig(_Base):
    """How source documents are split into retrievable units.

    Legal texts have real structure (Livre > Titre > Chapitre > Article). We
    exploit it: `strategy="structural"` splits on article boundaries rather than
    on a blind character count, so a retrieved chunk is a self-contained legal
    provision. `fixed` is kept as a deliberately weak baseline -- one of the
    first ablations is "does structure-aware chunking actually help?"
    """

    strategy: Literal["fixed", "structural"] = "structural"
    max_tokens: int = Field(512, gt=0, description="Upper bound on chunk size.")
    overlap_tokens: int = Field(64, ge=0, description="Only used by `fixed`.")
    min_tokens: int = Field(32, gt=0, description="Chunks smaller than this are merged.")


class EmbeddingConfig(_Base):
    """Dense encoder used to embed chunks and queries.

    The default is multilingual on purpose. This corpus is French, and most
    strong retrieval encoders are English-first -- picking a model that actually
    handles French is one of the substantive decisions in this project, and we
    will measure it rather than assume it.
    """

    model_name: str = "intfloat/multilingual-e5-base"
    device: Literal["cuda", "cpu", "auto"] = "auto"
    batch_size: int = Field(32, gt=0)
    normalize: bool = True
    # Some encoders (E5, BGE) are trained with asymmetric prefixes and degrade
    # noticeably without them. Kept explicit so the ablation is honest.
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "


class RetrievalConfig(_Base):
    """Which retrieval components are active, and how they are combined."""

    dense: bool = True
    bm25: bool = False
    # Reciprocal Rank Fusion constant, used when both dense and bm25 are on.
    rrf_k: int = Field(60, gt=0)
    top_k: int = Field(20, gt=0, description="Candidates fetched before reranking.")
    final_k: int = Field(5, gt=0, description="Passages actually shown to the LLM.")


class RerankerConfig(_Base):
    """Optional cross-encoder reranking stage."""

    enabled: bool = False
    model_name: str = "BAAI/bge-reranker-v2-m3"
    batch_size: int = Field(32, gt=0)

    # Cap on tokens per (query, passage) pair.
    #
    # This model advertises 8192, and leaving it there is what made reranking
    # 97.7% of end-to-end latency with a p90 four times its median. Attention is
    # quadratic, so a single long article dominates the batch it lands in --
    # measured at 1.7x the tokens costing 4.5x the time.
    #
    # 512 is a hypothesis, not a default to trust: it assumes a relevance
    # judgement is decided by the opening of an article rather than its tail.
    # The ablation tests it against recall.
    max_length: int = Field(512, gt=0)

    # Inference precision. The model loads in fp32, which wastes the Ada tensor
    # cores this project runs on. Ranking is an ordering problem and far less
    # sensitive to precision than generation, but "far less" is not "not at
    # all", so it is measured rather than assumed.
    precision: Literal["fp32", "fp16"] = "fp16"


class GeneratorConfig(_Base):
    """The model that writes the final answer.

    `adapter_path` points at a LoRA adapter when we get to Phase 4. Leaving it
    None gives the base model, which is exactly the comparison we need.
    """

    # Qwen3-4B-Instruct-2507. The bare "Qwen/Qwen3-4B-Instruct" that sat here
    # from Phase 0 does not resolve -- nothing caught it because nothing had
    # tried to load a generator yet. Llama-3.2-3B-Instruct was the other
    # candidate and is gated, which would have blocked mid-training.
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    adapter_path: str | None = None
    quantization: Literal["none", "4bit", "8bit"] = "4bit"
    max_new_tokens: int = Field(512, gt=0)
    temperature: float = Field(0.0, ge=0.0, le=2.0)

    # Passages given to the generator, truncated to this many characters each.
    # Reranking showed that an article's operative statement sits at its opening
    # and long enumerations dilute relevance; the same is likely true of the
    # generator's attention, and this makes it testable.
    max_passage_chars: int = Field(1200, gt=0)
    # Refusing to answer without evidence is a feature in a legal assistant,
    # not a bug. We measure abstention explicitly.
    allow_abstention: bool = True


class AgentConfig(_Base):
    """An LLM in the retrieval loop.

    Everything here is off by default, and every part is a separate switch, so
    "the agent helped" can be decomposed into which part helped. The alternative
    -- one `agentic: true` flag -- produces a result that cannot be attributed.

    The cost is unusually visible for this component. Decomposition is an extra
    generation call before retrieval even starts, and verification is another
    after. On this hardware a short generation is one to three seconds against a
    193 ms retrieval, so an agent that adds two calls is roughly a 20x latency
    increase on the retrieval stage. That is the number any accuracy gain has to
    be weighed against.
    """

    enabled: bool = False

    # Split a question into sub-questions, retrieve for each, fuse the rankings.
    # Aimed squarely at multi_hop: a question needing two articles gives the
    # retriever one query that is a blend of both, and a blend matches neither.
    decompose: bool = True
    max_subquestions: int = Field(3, ge=1, le=5)

    # Ask the model whether the retrieved passages actually answer the question,
    # and retrieve again with a rewritten query if not.
    verify: bool = False
    max_steps: int = Field(2, ge=1, le=4)

    # Tokens for the agent's own calls. Far below the generator's 512 because
    # sub-questions and a sufficiency verdict are short, and generation time is
    # linear in output length -- the single biggest lever on what the agent costs.
    max_new_tokens: int = Field(128, gt=0)

    # Whether the original question keeps a place among the fused rankings.
    # On by default as a floor: if decomposition produces nonsense, the original
    # query's ranking is still in the fusion and the result degrades gracefully
    # rather than collapsing.
    keep_original: bool = True


class EvalConfig(_Base):
    """What we measure and against what."""

    dataset_path: str = "data/eval/gold_qa.jsonl"
    recall_at_k: tuple[int, ...] = (1, 3, 5, 10, 20)
    judge_model: str | None = None
    seed: int = 42


class ExperimentConfig(_Base):
    """Root config. One instance == one fully specified system."""

    name: str = Field(..., description="Human-readable label, e.g. 'hybrid+rerank'.")
    description: str = ""

    # default_factory rather than a shared instance: these models are frozen, so
    # sharing would in fact be safe, but the factory keeps that safety a property
    # of the code instead of a fact you have to remember when unfreezing them.
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    # -- identity -----------------------------------------------------------

    def fingerprint(self) -> str:
        """Stable content hash of everything that affects results.

        `name` and `description` are excluded deliberately: renaming an
        experiment should not invalidate its cached index or results. Two
        configs that differ only in label are the same system.

        `agent` is excluded when it is disabled, and this is a deliberate
        exception rather than an oversight. Adding a field to this model changes
        the hash of *every* config that ever used it, which would rename every
        run directory recorded before the field existed and break the link
        between the published ablation table and the runs behind it. A subsystem
        that is switched off contributes nothing to the system's behaviour, so
        excluding it keeps the identity of a system that predates it intact.
        `tests/test_config.py` pins the run ids of the recorded runs against
        exactly this.

        The exception is narrow on purpose: it applies to `agent` alone, and a
        disabled reranker or BM25 still participates, because those were present
        when the recorded runs were made and changing them now would cause the
        same breakage in the other direction.
        """
        exclude: set[str] = {"name", "description"}
        if not self.agent.enabled:
            exclude.add("agent")
        payload = self.model_dump(mode="json", exclude=exclude)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]

    @property
    def run_id(self) -> str:
        return f"{self.name}-{self.fingerprint()}"

    def index_fingerprint(self) -> str:
        """Hash of only the parts that determine the *index* contents.

        Rebuilding embeddings takes minutes; reranking or generation settings do
        not change them. Separating the two hashes means swapping the generator
        reuses the existing index instead of recomputing it.
        """
        payload = {
            "chunking": self.chunking.model_dump(mode="json"),
            "embedding": self.embedding.model_dump(mode="json"),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into `base`, returning a new dict."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path, *, base: str | Path | None = None) -> ExperimentConfig:
    """Load a YAML config, optionally layered on top of a base config.

    Experiment files stay tiny -- they express only the delta from the baseline,
    which makes an ablation series readable at a glance:

        # configs/experiments/03_rerank.yaml
        name: hybrid+rerank
        reranker:
          enabled: true
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle) or {}

    # A config may name its own parent via `extends:`, or be given one directly.
    parent = data.pop("extends", None) or base
    if parent is not None:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        with parent_path.open("r", encoding="utf-8") as handle:
            parent_data: dict[str, Any] = yaml.safe_load(handle) or {}
        parent_data.pop("extends", None)
        data = _deep_merge(parent_data, data)

    return ExperimentConfig.model_validate(data)


def save_resolved(config: ExperimentConfig, destination: str | Path) -> Path:
    """Write the fully resolved config next to a run's results.

    This is what makes a results table auditable six months later: the numbers
    and the exact system that produced them sit in the same directory.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            config.model_dump(mode="json"),
            handle,
            sort_keys=True,
            allow_unicode=True,
        )
    return destination
