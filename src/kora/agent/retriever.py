"""Two agentic retrievers, deliberately kept separate.

Both satisfy the same `Retriever` protocol as every other stage, so the
evaluation harness runs them unchanged and their numbers land in the same
ablation table as everything else. That is the point: an agent that needed its
own evaluation path would be comparing itself against a different measurement.

They sit at different depths of the pipeline, and the placement is the design:

    dense -> [decompose] -> rerank -> [verify]

**Decomposition goes inside the reranker.** It changes what the candidate pool
*contains*, and the cross-encoder is the strongest component in the stack, so it
should be the thing that reads the improved pool. Fusing already-reranked lists
instead would spend one cross-encoder pass per sub-question to produce a ranking
the reranker never sees whole.

Phase 3 matters here. Widening the pool from 20 to 50 made reranking *worse* —
extra candidates gave distractors more chances to reach the top five than true
positives had to be rescued. So decomposition keeps the pool at `top_k` and
changes only its composition. If a wider pool were the mechanism, Phase 3 says
it would hurt.

**Verification goes outside the reranker**, because judging sufficiency is only
meaningful on the passages that would actually be shown to the generator, and
those are the reranked ones.
"""

from __future__ import annotations

import time

from kora.agent.prompts import (
    build_decompose_messages,
    build_verify_messages,
    parse_subquestions,
    parse_verdict,
)
from kora.config import ExperimentConfig
from kora.logging import get_logger
from kora.retrieval.dense import Hit
from kora.retrieval.fusion import reciprocal_rank_fusion
from kora.retrieval.rerank import Retriever

log = get_logger(__name__)


def _fuse(rankings: list[list[str]], articles: dict[str, Hit], *, rrf_k: int, k: int) -> list[Hit]:
    """Fuse ranked chunk-id lists back into hits."""
    fused = reciprocal_rank_fusion(rankings, k=rrf_k)
    return [
        Hit(chunk_id=chunk_id, score=score, article=articles[chunk_id].article)
        for chunk_id, score in fused[:k]
    ]


class DecomposingRetriever:
    """Splits a question into sub-questions, retrieves for each, fuses the ranks.

    Fusion is RRF, the same function the hybrid retriever uses, and for the same
    reason: two sub-questions produce similarity scores on incomparable scales,
    and combining ranks avoids inventing a normalisation that would quietly
    become a hyperparameter.
    """

    def __init__(self, config: ExperimentConfig, inner: Retriever, generator) -> None:
        self.config = config
        self.inner = inner
        self.generator = generator
        # Populated per call so the runner and any caller can see what the agent
        # actually did. An agent whose decisions are invisible cannot be debugged
        # from a results file.
        self.last_subquestions: list[str] = []
        self.last_decompose_ms: float = 0.0

    def _decompose(self, question: str) -> list[str]:
        started = time.perf_counter()
        try:
            reply = self.generator.chat(
                build_decompose_messages(question),
                max_new_tokens=self.config.agent.max_new_tokens,
            )
        except Exception:
            # A failed decomposition must degrade to single-shot retrieval, not
            # take down the run. Sixty questions into an evaluation is the wrong
            # place to discover that one prompt upset the tokenizer.
            log.exception("decomposition failed; falling back to the original question")
            return []
        finally:
            self.last_decompose_ms = (time.perf_counter() - started) * 1000

        return parse_subquestions(
            reply,
            original=question,
            max_subquestions=self.config.agent.max_subquestions,
        )

    def retrieve(self, question: str, k: int) -> list[Hit]:
        self.last_subquestions = self._decompose(question) if self.config.agent.decompose else []

        queries = [question] if self.config.agent.keep_original else []
        queries += self.last_subquestions
        if not queries:
            queries = [question]

        rankings: list[list[str]] = []
        seen: dict[str, Hit] = {}
        for query in queries:
            hits = self.inner.retrieve(query, k)
            rankings.append([hit.chunk_id for hit in hits])
            for hit in hits:
                seen.setdefault(hit.chunk_id, hit)

        if len(rankings) == 1:
            # Nothing to fuse. Returning the ranking untouched keeps an atomic
            # question bit-identical to the non-agentic system, so any change in
            # the lookup numbers is a real effect and not fusion noise.
            return [seen[chunk_id] for chunk_id in rankings[0]][:k]

        return _fuse(rankings, seen, rrf_k=self.config.retrieval.rrf_k, k=k)


class VerifyingRetriever:
    """Asks whether the retrieved passages suffice, and searches again if not.

    The follow-up query is whatever the model says is missing, which makes this
    a self-correction step rather than a fixed second query. Its results are
    fused with the first attempt rather than replacing them: the model is being
    asked what is *missing*, so its answer describes a gap, not a better version
    of the original question.
    """

    def __init__(self, config: ExperimentConfig, inner: Retriever, generator) -> None:
        self.config = config
        self.inner = inner
        self.generator = generator
        self.last_followups: list[str] = []
        self.last_verify_ms: float = 0.0

    def _verdict(self, question: str, hits: list[Hit]) -> str | None:
        from kora.generation.prompt import build_context

        context = build_context(
            [hit.article for hit in hits[: self.config.retrieval.final_k]],
            max_chars=self.config.generator.max_passage_chars,
        )
        try:
            reply = self.generator.chat(
                build_verify_messages(question, context),
                max_new_tokens=self.config.agent.max_new_tokens,
            )
        except Exception:
            log.exception("verification failed; keeping the current results")
            return None
        return parse_verdict(reply)

    def retrieve(self, question: str, k: int) -> list[Hit]:
        started = time.perf_counter()
        self.last_followups = []

        hits = self.inner.retrieve(question, k)
        rankings = [[hit.chunk_id for hit in hits]]
        seen = {hit.chunk_id: hit for hit in hits}

        # max_steps counts total retrieval attempts, so max_steps=2 means one
        # retry. Bounded because each step is a generation plus a full rerank,
        # and an unbounded loop against a model that always finds something
        # missing would never terminate.
        for _ in range(self.config.agent.max_steps - 1):
            followup = self._verdict(question, hits)
            if followup is None:
                break
            self.last_followups.append(followup)

            extra = self.inner.retrieve(followup, k)
            if not extra:
                break
            rankings.append([hit.chunk_id for hit in extra])
            for hit in extra:
                seen.setdefault(hit.chunk_id, hit)
            hits = _fuse(rankings, seen, rrf_k=self.config.retrieval.rrf_k, k=k)

        self.last_verify_ms = (time.perf_counter() - started) * 1000
        return hits
