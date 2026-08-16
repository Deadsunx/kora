"""Hybrid retrieval: dense and lexical, fused by reciprocal rank."""

from __future__ import annotations

from kora.config import ExperimentConfig
from kora.documents import Article
from kora.logging import get_logger
from kora.retrieval.dense import DenseRetriever, Hit
from kora.retrieval.fusion import reciprocal_rank_fusion
from kora.retrieval.lexical import BM25Retriever

log = get_logger(__name__)


class HybridRetriever:
    """Dense + BM25, fused with RRF.

    Both retrievers are asked for the full candidate pool rather than half each.
    Splitting the budget would make the fusion's job easier and the comparison
    dishonest: hybrid would then be seen fewer dense candidates than the dense
    baseline, and any difference would confound fusion with pool size.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        dense: DenseRetriever,
        lexical: BM25Retriever,
        articles: list[Article],
    ) -> None:
        self.config = config
        self.dense = dense
        self.lexical = lexical
        self._articles = {a.chunk_id: a for a in articles}

    def retrieve(self, question: str, k: int) -> list[Hit]:
        dense_hits = self.dense.retrieve(question, k)
        lexical_hits = self.lexical.search(question, k)

        rankings = [
            [hit.chunk_id for hit in dense_hits],
            [chunk_id for chunk_id, _ in lexical_hits],
        ]
        fused = reciprocal_rank_fusion(rankings, k=self.config.retrieval.rrf_k)

        return [
            Hit(chunk_id=chunk_id, score=score, article=self._articles[chunk_id])
            for chunk_id, score in fused[:k]
        ]
