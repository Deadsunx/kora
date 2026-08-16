"""Run one experiment end to end and write its results.

Output layout, per run:

    runs/<run_id>/
        config.resolved.yaml   the exact system that produced these numbers
        metrics.json           the numbers
        predictions.jsonl      what was retrieved for each question

The run directory is named for the config's content hash, so results and the
system that produced them cannot drift apart, and two different systems cannot
overwrite each other.

Every metric is reported twice: over validated questions (the headline) and over
all questions (diagnostic). The second is useful while building; only the first
may be quoted. Both carry their sample size, because a recall figure without an
n behind it is not a result.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from kora.config import ExperimentConfig, save_resolved
from kora.eval.dataset import GoldQuestion, GoldSet
from kora.eval.metrics import (
    hit_rate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from kora.logging import get_logger
from kora.paths import run_path
from kora.retrieval.dense import DenseRetriever, Hit

log = get_logger(__name__)


@dataclass(slots=True)
class QuestionResult:
    """What a system returned for one question."""

    question_id: str
    kind: str
    provenance: str
    retrieved: list[str]
    scores: list[float]
    gold: list[str]
    latency_ms: float
    # Retrieved articles that are repealed. Not an error by itself -- the
    # distractors are indexed on purpose -- but the rate at which they surface
    # is the leading indicator for the citation-safety metric downstream.
    unsafe_retrieved: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "kind": self.kind,
            "provenance": self.provenance,
            "gold": self.gold,
            "retrieved": self.retrieved,
            "scores": [round(s, 5) for s in self.scores],
            "unsafe_retrieved": self.unsafe_retrieved,
            "latency_ms": round(self.latency_ms, 2),
        }


def _retrieval_metrics(
    results: list[QuestionResult],
    ks: tuple[int, ...],
    final_k: int,
) -> dict[str, float]:
    """Aggregate retrieval metrics over questions that have gold articles.

    Unanswerable questions are excluded here rather than scored as zero: recall
    against an empty gold set is undefined, and counting them as failures would
    make a system look worse the more honest questions the set contains. They
    are measured by abstention instead, once generation exists.
    """
    scored = [r for r in results if r.gold]
    if not scored:
        return {}

    metrics: dict[str, float] = {"n": float(len(scored))}
    for k in ks:
        metrics[f"recall@{k}"] = sum(recall_at_k(r.retrieved, r.gold, k) for r in scored) / len(
            scored
        )
        metrics[f"hit@{k}"] = sum(hit_rate(r.retrieved, r.gold, k) for r in scored) / len(scored)

    metrics[f"precision@{final_k}"] = sum(
        precision_at_k(r.retrieved, r.gold, final_k) for r in scored
    ) / len(scored)
    metrics["mrr"] = sum(reciprocal_rank(r.retrieved, r.gold) for r in scored) / len(scored)
    metrics[f"ndcg@{final_k}"] = sum(ndcg_at_k(r.retrieved, r.gold, final_k) for r in scored) / len(
        scored
    )

    # Latency is reported with the accuracy it bought. A reranker that adds
    # eight points of recall for 200 ms is a different decision from one that
    # adds eight points for two seconds, and a table showing only accuracy
    # cannot express that.
    latencies = sorted(r.latency_ms for r in scored)
    metrics["latency_ms_median"] = latencies[len(latencies) // 2]
    metrics["latency_ms_p90"] = latencies[int(len(latencies) * 0.9)]

    unsafe = sum(1 for r in scored if r.unsafe_retrieved)
    metrics[f"questions_with_repealed_in_top{final_k}"] = unsafe / len(scored)
    return metrics


def run_retrieval_experiment(
    config: ExperimentConfig,
    gold: GoldSet,
    retriever: DenseRetriever,
) -> tuple[dict[str, Any], list[QuestionResult]]:
    """Retrieve for every gold question and compute metrics.

    Returns the aggregate report and the per-question results, because the
    second is what you actually read when a number looks wrong.
    """
    questions: list[GoldQuestion] = list(gold)
    if not questions:
        raise ValueError("gold set is empty")

    texts = [q.question for q in questions]

    # Encoding is batched, so per-question latency is measured over the search
    # only and the encode cost is reported separately. Attributing a batched
    # encode to individual questions would understate real single-query latency.
    encode_start = time.perf_counter()
    query_vectors = retriever.encode_queries(texts)
    encode_ms = (time.perf_counter() - encode_start) * 1000

    results: list[QuestionResult] = []
    for index, question in enumerate(questions):
        start = time.perf_counter()
        hits: list[Hit] = retriever.search_vectors(
            query_vectors[index : index + 1], config.retrieval.top_k
        )[0]
        latency_ms = (time.perf_counter() - start) * 1000

        window = hits[: config.retrieval.final_k]
        results.append(
            QuestionResult(
                question_id=question.id,
                kind=question.kind,
                provenance=question.provenance,
                retrieved=[h.chunk_id for h in hits],
                scores=[h.score for h in hits],
                gold=list(question.gold_chunk_ids),
                latency_ms=latency_ms,
                unsafe_retrieved=[h.chunk_id for h in window if h.is_unsafe_to_cite],
            )
        )

    ks = tuple(config.eval.recall_at_k)
    final_k = config.retrieval.final_k
    headline_ids = {q.id for q in gold.headline()}
    headline = [r for r in results if r.question_id in headline_ids]

    report: dict[str, Any] = {
        "run_id": config.run_id,
        "name": config.name,
        "index_fingerprint": config.index_fingerprint(),
        "corpus_size": len(retriever.index),
        "gold_composition": gold.composition(),
        "encode_ms_total": round(encode_ms, 2),
        "headline": _retrieval_metrics(headline, ks, final_k),
        "all_questions": _retrieval_metrics(results, ks, final_k),
        "by_kind": {
            kind: _retrieval_metrics([r for r in results if r.kind == kind], ks, final_k)
            for kind in sorted({r.kind for r in results})
        },
    }

    if not headline:
        report["warning"] = (
            "No validated questions: headline metrics are empty. Numbers under "
            "'all_questions' include model-drafted questions and must not be reported."
        )
        log.warning("no validated questions; headline metrics unavailable")

    return report, results


def write_run(
    config: ExperimentConfig,
    report: dict[str, Any],
    results: list[QuestionResult],
) -> str:
    """Persist a run's config, metrics and per-question predictions."""
    directory = run_path(config.run_id)
    directory.mkdir(parents=True, exist_ok=True)

    save_resolved(config, directory / "config.resolved.yaml")
    (directory / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (directory / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.to_json(), ensure_ascii=False))
            handle.write("\n")

    log.info("run written", run_id=config.run_id, path=str(directory))
    return str(directory)
