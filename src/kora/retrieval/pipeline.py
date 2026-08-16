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
    """Human-readable summary of the active pipeline, e.g. 'dense + bm25 + rerank'.

    Written in pipeline order, so the string says where the agent stages sit
    rather than merely that they are on.
    """
    parts = []
    if config.retrieval.dense:
        parts.append("dense")
    if config.retrieval.bm25:
        parts.append("bm25")
    if config.agent.enabled and config.agent.decompose:
        parts.append("decompose")
    if config.reranker.enabled:
        parts.append("rerank")
    if config.agent.enabled and config.agent.verify:
        parts.append("verify")
    return " + ".join(parts) or "none"


def build_retriever(
    config: ExperimentConfig,
    index: DenseIndex,
    articles: list[Article],
    generator: object | None = None,
) -> Retriever:
    """Construct the retriever a config asks for.

    `generator` is required only when the config enables the agent, and is
    passed in rather than constructed here so that one model serves both the
    agent and the answer. Three models already occupy most of 8 GiB; a fourth
    is not a design choice, it is an out-of-memory error.
    """
    # Validate before building anything. Loading an encoder and then discovering
    # the config is incoherent wastes a minute and reports the wrong error.
    if not config.retrieval.dense and not config.retrieval.bm25:
        raise ValueError("config enables neither dense nor bm25 retrieval")
    if config.agent.enabled and generator is None:
        raise ValueError("agent.enabled requires a generator")

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

    # Order is the design; see kora/agent/retriever.py. Decomposition changes
    # what the candidate pool contains, so it goes under the reranker.
    # Verification judges the passages that would actually be shown, so it goes
    # over it.
    agent = config.agent
    if agent.enabled and agent.decompose:
        from kora.agent.retriever import DecomposingRetriever

        base = DecomposingRetriever(config, base, generator)

    if config.reranker.enabled:
        base = RerankingRetriever(config, base)

    if agent.enabled and agent.verify:
        from kora.agent.retriever import VerifyingRetriever

        base = VerifyingRetriever(config, base, generator)

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
