"""Assemble the retrieval pipeline described by a config.

One function, so that every experiment in the ablation series is built by the
same code from its config file. The alternative -- constructing retrievers by
hand per experiment -- is how an ablation table ends up comparing systems that
differ in ways nobody wrote down.
"""

from __future__ import annotations

from kora.config import ExperimentConfig
from kora.documents import Article
from kora.logging import get_logger
from kora.retrieval.dense import DenseRetriever
from kora.retrieval.hybrid import HybridRetriever
from kora.retrieval.index import DenseIndex
from kora.retrieval.lexical import BM25Retriever
from kora.retrieval.rerank import RerankingRetriever, Retriever

log = get_logger(__name__)


def describe(config: ExperimentConfig) -> str:
    """Human-readable summary of the active pipeline, e.g. 'dense + bm25 + rerank'."""
    parts = []
    if config.retrieval.dense:
        parts.append("dense")
    if config.retrieval.bm25:
        parts.append("bm25")
    if config.reranker.enabled:
        parts.append("rerank")
    return " + ".join(parts) or "none"


def build_retriever(
    config: ExperimentConfig,
    index: DenseIndex,
    articles: list[Article],
) -> Retriever:
    """Construct the retriever a config asks for."""
    if not config.retrieval.dense and not config.retrieval.bm25:
        raise ValueError("config enables neither dense nor bm25 retrieval")

    base: Retriever
    if config.retrieval.dense and config.retrieval.bm25:
        base = HybridRetriever(
            config,
            DenseRetriever(config, index, articles),
            BM25Retriever(articles),
            articles,
        )
    elif config.retrieval.dense:
        base = DenseRetriever(config, index, articles)
    else:
        # Lexical only. Useful as an ablation floor: it answers "how much of the
        # baseline's performance is semantic at all?", which the hybrid result
        # cannot tell you on its own.
        base = _LexicalOnly(BM25Retriever(articles), articles)

    if config.reranker.enabled:
        base = RerankingRetriever(config, base)

    log.info("retriever built", pipeline=describe(config))
    return base


class _LexicalOnly:
    """Adapts BM25Retriever to the Retriever protocol."""

    def __init__(self, lexical: BM25Retriever, articles: list[Article]) -> None:
        self.lexical = lexical
        self._articles = {a.chunk_id: a for a in articles}

    def retrieve(self, question: str, k: int):
        from kora.retrieval.dense import Hit

        return [
            Hit(chunk_id=chunk_id, score=score, article=self._articles[chunk_id])
            for chunk_id, score in self.lexical.search(question, k)
        ]
