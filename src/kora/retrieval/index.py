"""Build and load the dense index.

Indexes are stored under a directory named for the config's
`index_fingerprint()`, which covers chunking and embedding settings and nothing
else. Two experiments differing only in reranking or generation therefore share
one index automatically, and two that differ in how text was chunked or encoded
can never collide. That is the whole reason the fingerprint was split in two
back in Phase 0.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kora import paths
from kora.config import ExperimentConfig
from kora.documents import Article
from kora.logging import get_logger

log = get_logger(__name__)

VECTORS_FILE = "vectors.npy"
CHUNKS_FILE = "chunks.jsonl"
META_FILE = "index_meta.json"


def load_articles(*, in_force_only: bool = False) -> list[Article]:
    """Load every parsed article from `data/interim`.

    `in_force_only` drops repealed acts *and* individually repealed articles.
    Left off by default: the repealed texts are the distractor set, and removing
    them would make the citation-safety metric unmeasurable by construction --
    a system cannot cite repealed law that was never indexed, which would look
    like a perfect score and mean nothing.
    """
    articles: list[Article] = []
    for path in sorted(paths.INTERIM_DIR.glob("*.articles.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                article = Article.model_validate_json(line)
                if in_force_only and (article.repealed or article.status == "superseded"):
                    continue
                articles.append(article)
    return articles


@dataclass(frozen=True, slots=True)
class DenseIndex:
    """Vectors plus the chunk ids they correspond to, in the same order."""

    vectors: np.ndarray  # shape (n, dim), L2-normalised
    chunk_ids: tuple[str, ...]
    model_name: str

    def __len__(self) -> int:
        return len(self.chunk_ids)

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])


def index_dir(config: ExperimentConfig) -> Path:
    return paths.index_path(config.index_fingerprint())


def index_exists(config: ExperimentConfig) -> bool:
    directory = index_dir(config)
    return all((directory / name).exists() for name in (VECTORS_FILE, CHUNKS_FILE, META_FILE))


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover - torch is a hard dependency in practice
        return "cpu"


def build_index(config: ExperimentConfig, articles: list[Article]) -> DenseIndex:
    """Encode every article and persist the result.

    The passage prefix matters more than it looks. E5 and BGE encoders are
    trained with asymmetric query/passage prefixes and lose measurable accuracy
    without them -- a silent degradation that looks like "embeddings are just
    mediocre on French" rather than like a bug. It is in the config so the
    ablation can test it rather than assume it.
    """
    from sentence_transformers import SentenceTransformer

    device = _resolve_device(config.embedding.device)
    log.info(
        "building index",
        model=config.embedding.model_name,
        device=device,
        articles=len(articles),
        fingerprint=config.index_fingerprint(),
    )

    model = SentenceTransformer(config.embedding.model_name, device=device)
    texts = [config.embedding.passage_prefix + a.to_indexed_text() for a in articles]

    vectors = model.encode(
        texts,
        batch_size=config.embedding.batch_size,
        normalize_embeddings=config.embedding.normalize,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    index = DenseIndex(
        vectors=vectors,
        chunk_ids=tuple(a.chunk_id for a in articles),
        model_name=config.embedding.model_name,
    )
    save_index(config, index, articles)
    return index


def save_index(config: ExperimentConfig, index: DenseIndex, articles: list[Article]) -> Path:
    directory = index_dir(config)
    directory.mkdir(parents=True, exist_ok=True)

    np.save(directory / VECTORS_FILE, index.vectors)

    # The articles are stored beside the vectors rather than re-read from
    # `interim`, so an index stays interpretable even if the corpus is re-parsed
    # underneath it. Otherwise a reused index and a changed parse disagree
    # silently about what chunk id means.
    with (directory / CHUNKS_FILE).open("w", encoding="utf-8") as handle:
        for article in articles:
            handle.write(article.model_dump_json())
            handle.write("\n")

    (directory / META_FILE).write_text(
        json.dumps(
            {
                "index_fingerprint": config.index_fingerprint(),
                "model_name": index.model_name,
                "count": len(index),
                "dim": index.dim,
                "normalize": config.embedding.normalize,
                "passage_prefix": config.embedding.passage_prefix,
                "chunking": config.chunking.model_dump(mode="json"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("index saved", path=str(directory), vectors=len(index), dim=index.dim)
    return directory


def load_index(config: ExperimentConfig) -> tuple[DenseIndex, list[Article]]:
    """Load a previously built index and its articles."""
    directory = index_dir(config)
    if not index_exists(config):
        raise FileNotFoundError(
            f"No index at {directory}. Run `kora index build --config <file>` first."
        )

    meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
    vectors = np.load(directory / VECTORS_FILE)

    articles: list[Article] = []
    with (directory / CHUNKS_FILE).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                articles.append(Article.model_validate_json(line))

    if len(articles) != vectors.shape[0]:
        raise ValueError(
            f"index at {directory} is inconsistent: "
            f"{vectors.shape[0]} vectors but {len(articles)} chunks"
        )

    index = DenseIndex(
        vectors=vectors,
        chunk_ids=tuple(a.chunk_id for a in articles),
        model_name=meta["model_name"],
    )
    return index, articles
