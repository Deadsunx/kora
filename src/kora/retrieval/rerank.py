"""Cross-encoder reranking.

What a reranker actually changes
--------------------------------
Bi-encoders (the dense index) embed the question and each article separately and
compare vectors. That is what makes them fast enough to search 3,000 documents
in under a millisecond -- and it is also their limit: the article is encoded
without ever having seen the question.

A cross-encoder reads the pair together and scores it directly. It cannot
pre-compute anything, so it is orders of magnitude slower per candidate, which
is exactly why it sits *after* retrieval rather than replacing it: a wide, cheap
pass proposes candidates, an expensive pass re-orders them.

Two consequences the ablation should show
-----------------------------------------
A reranker can only reorder what retrieval already found. If recall@50 is 0.87,
no reranker will produce recall@5 above 0.87 -- the ceiling is set upstream. So
the fair reading of a reranker's contribution is how much of the gap between
recall@final_k and recall@top_k it closes, not its absolute score.

And it costs real time. On an 8 GiB laptop GPU, scoring 50 candidates is tens of
milliseconds against the dense stage's fraction of one. The table reports both,
because "+8 points of recall for +200 ms" and "+8 points for +2 s" are different
engineering decisions.
"""

from __future__ import annotations

from typing import Protocol

from kora.config import ExperimentConfig
from kora.logging import get_logger
from kora.retrieval.dense import Hit
from kora.retrieval.index import _resolve_device

log = get_logger(__name__)


class Retriever(Protocol):
    """Anything that can return ranked hits for a question."""

    def retrieve(self, question: str, k: int) -> list[Hit]: ...


class RerankingRetriever:
    """Wraps a retriever and re-orders its candidates with a cross-encoder."""

    def __init__(self, config: ExperimentConfig, inner: Retriever) -> None:
        self.config = config
        self.inner = inner
        self._model = None

    def _cross_encoder(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            device = _resolve_device(self.config.embedding.device)
            log.info(
                "loading reranker",
                model=self.config.reranker.model_name,
                device=device,
            )
            self._model = CrossEncoder(self.config.reranker.model_name, device=device)
        return self._model

    def retrieve(self, question: str, k: int) -> list[Hit]:
        """Fetch a wide candidate pool, rescore it, return the best k.

        The pool is `retrieval.top_k` wide regardless of the `k` requested,
        because a reranker given five candidates has almost nothing to do. The
        whole point is to retrieve widely and cut precisely.
        """
        candidates = self.inner.retrieve(question, self.config.retrieval.top_k)
        if not candidates:
            return []

        pairs = [(question, hit.article.to_indexed_text()) for hit in candidates]
        scores = self._cross_encoder().predict(
            pairs,
            batch_size=self.config.reranker.batch_size,
            show_progress_bar=False,
        )

        rescored = [
            Hit(chunk_id=hit.chunk_id, score=float(score), article=hit.article)
            for hit, score in zip(candidates, scores, strict=True)
        ]
        rescored.sort(key=lambda hit: -hit.score)
        return rescored[:k]
