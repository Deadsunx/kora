"""Dense retrieval over the article index.

On not using FAISS
------------------
faiss is installed, and this module does not use it. At 3,056 vectors an exact
brute-force matmul takes under a millisecond, while FAISS's value is approximate
search on collections orders of magnitude larger -- where it trades recall for
speed. Here that trade is a pure loss: we would give up exactness and gain
nothing measurable, while adding a source of error that is indistinguishable
from a retrieval bug.

Worth saying plainly because reaching for a vector database is the reflex, and
the reflex is wrong at this scale. If the corpus grows past a few hundred
thousand chunks the calculation changes, and the swap is confined to this file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kora.config import ExperimentConfig
from kora.documents import Article
from kora.logging import get_logger
from kora.retrieval.index import DenseIndex

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Hit:
    """One retrieved article and its score."""

    chunk_id: str
    score: float
    article: Article

    @property
    def is_unsafe_to_cite(self) -> bool:
        """Whether citing this article would mean citing repealed law."""
        return self.article.repealed or self.article.status == "superseded"


class DenseRetriever:
    """Exact cosine-similarity retrieval over the article index."""

    def __init__(
        self,
        config: ExperimentConfig,
        index: DenseIndex,
        articles: list[Article],
    ) -> None:
        self.config = config
        self.index = index
        self._articles = {a.chunk_id: a for a in articles}
        self._model = None  # loaded lazily; encoding a query needs the encoder

        if not config.embedding.normalize:
            # Cosine similarity assumes unit vectors. Rather than silently
            # computing a dot product and calling it cosine, normalise here and
            # say so.
            norms = np.linalg.norm(index.vectors, axis=1, keepdims=True)
            self._vectors = index.vectors / np.clip(norms, 1e-12, None)
            log.warning("index vectors were not normalised; normalising for cosine similarity")
        else:
            self._vectors = index.vectors

    def _encoder(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            from kora.retrieval.index import _resolve_device

            device = _resolve_device(self.config.embedding.device)
            self._model = SentenceTransformer(self.config.embedding.model_name, device=device)
        return self._model

    def encode_queries(self, questions: list[str]) -> np.ndarray:
        """Embed questions, applying the encoder's query prefix.

        Batched across all questions rather than one at a time: on a laptop GPU
        the per-call overhead dominates for short texts, and a 60-question eval
        run should not pay it sixty times.
        """
        texts = [self.config.embedding.query_prefix + q for q in questions]
        return (
            self._encoder()
            .encode(
                texts,
                batch_size=self.config.embedding.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            .astype(np.float32)
        )

    def search_vectors(self, query_vectors: np.ndarray, k: int) -> list[list[Hit]]:
        """Top-k for each query vector."""
        if k <= 0:
            raise ValueError("k must be positive")
        k = min(k, len(self.index))

        # (queries, dim) @ (dim, chunks) -> (queries, chunks)
        scores = query_vectors @ self._vectors.T

        # argpartition finds the top k without sorting the whole row, then only
        # those k are sorted. Irrelevant at this size, but it costs nothing and
        # keeps the hot path honest if the corpus grows.
        top = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]

        results: list[list[Hit]] = []
        for row, indices in enumerate(top):
            ordered = indices[np.argsort(-scores[row, indices])]
            results.append(
                [
                    Hit(
                        chunk_id=self.index.chunk_ids[int(i)],
                        score=float(scores[row, int(i)]),
                        article=self._articles[self.index.chunk_ids[int(i)]],
                    )
                    for i in ordered
                ]
            )
        return results

    def search(self, questions: list[str], k: int | None = None) -> list[list[Hit]]:
        """Retrieve for a batch of questions."""
        k = k if k is not None else self.config.retrieval.top_k
        return self.search_vectors(self.encode_queries(questions), k)

    def retrieve(self, question: str, k: int) -> list[Hit]:
        """Retrieve for one question, encoding included.

        The single-question path exists so the experiment runner can time what
        a user actually waits for. Timing a batched encode and dividing by the
        batch size understates real latency by an order of magnitude, and the
        whole point of the ablation table is to price each component honestly.
        """
        return self.search_vectors(self.encode_queries([question]), k)[0]
