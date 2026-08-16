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
from kora.retrieval.dense import Hit
from kora.retrieval.rerank import Retriever

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

    # Populated only when generation is enabled.
    answer: str | None = None
    cited: list[str] = field(default_factory=list)
    cited_repealed: list[str] = field(default_factory=list)
    cited_not_retrieved: list[str] = field(default_factory=list)
    abstained: bool | None = None
    answer_latency_ms: float | None = None

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "question_id": self.question_id,
            "kind": self.kind,
            "provenance": self.provenance,
            "gold": self.gold,
            "retrieved": self.retrieved,
            "scores": [round(s, 5) for s in self.scores],
            "unsafe_retrieved": self.unsafe_retrieved,
            "latency_ms": round(self.latency_ms, 2),
        }
        if self.answer is not None:
            payload |= {
                "answer": self.answer,
                "cited": self.cited,
                "cited_repealed": self.cited_repealed,
                "cited_not_retrieved": self.cited_not_retrieved,
                "abstained": self.abstained,
                "answer_latency_ms": round(self.answer_latency_ms or 0.0, 2),
            }
        return payload


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


def _answer_metrics(results: list[QuestionResult]) -> dict[str, float]:
    """Metrics over generated answers.

    Reported separately from retrieval because they measure a different system:
    perfect retrieval followed by a fabricated answer scores well above and
    badly below, and averaging the two would hide it.
    """
    from kora.eval.metrics import abstention_scores, citation_precision, citation_recall

    answered = [r for r in results if r.answer is not None]
    if not answered:
        return {}

    metrics: dict[str, float] = {"n": float(len(answered))}

    scores = abstention_scores(
        abstained=[bool(r.abstained) for r in answered],
        answerable=[bool(r.gold) for r in answered],
    )
    metrics |= scores

    # Citation quality is scored only where the system chose to answer and the
    # question has a gold answer. Scoring an abstention's citations would
    # measure nothing, since a correct abstention cites nothing by design.
    citing = [r for r in answered if r.gold and not r.abstained]
    if citing:
        # Recall first: it is the metric that tracks legal correctness.
        # Precision is reported beside it, and read with the knowledge that a
        # correct supplementary citation lowers it.
        metrics["citation_recall"] = sum(citation_recall(r.cited, r.gold) for r in citing) / len(
            citing
        )
        metrics["citation_precision"] = sum(
            citation_precision(r.cited, r.gold) for r in citing
        ) / len(citing)
        metrics["answers_with_no_citation"] = sum(1 for r in citing if not r.cited) / len(citing)

        # Citing an article that was never retrieved means the model produced it
        # from parametric memory rather than the provided context. In a
        # grounded system that is fabrication, even when the article is real.
        metrics["answers_citing_unretrieved"] = sum(
            1 for r in citing if r.cited_not_retrieved
        ) / len(citing)

        # The metric the corpus exists to support.
        metrics["answers_citing_repealed"] = sum(1 for r in citing if r.cited_repealed) / len(
            citing
        )

    latencies = sorted(r.answer_latency_ms or 0.0 for r in answered)
    metrics["answer_latency_ms_median"] = latencies[len(latencies) // 2]
    metrics["answer_latency_ms_p90"] = latencies[int(len(latencies) * 0.9)]
    return metrics


def run_retrieval_experiment(
    config: ExperimentConfig,
    gold: GoldSet,
    retriever: Retriever,
    *,
    corpus_size: int,
    generator: Any | None = None,
) -> tuple[dict[str, Any], list[QuestionResult]]:
    """Retrieve for every gold question and compute metrics.

    Returns the aggregate report and the per-question results, because the
    second is what you actually read when a number looks wrong.
    """
    questions: list[GoldQuestion] = list(gold)
    if not questions:
        raise ValueError("gold set is empty")

    # One question at a time, timing the whole pipeline. Batching the encode and
    # dividing by batch size understates real latency by an order of magnitude,
    # and pricing each component honestly is the entire purpose of the table.
    #
    # The first question also pays for lazily loading the encoder and reranker,
    # so it is run once and discarded before timing begins.
    if questions:
        warmup_hits = retriever.retrieve(questions[0].question, config.retrieval.top_k)
        if generator is not None:
            # The generator loads lazily, so without this the first question
            # pays for loading -- or, on a cold cache, downloading -- several
            # gigabytes of weights inside a timed block. The first smoke test
            # reported an 782-second p90 for exactly this reason.
            generator.answer(
                questions[0].question,
                [h.article for h in warmup_hits[: config.retrieval.final_k]],
            )

    results: list[QuestionResult] = []
    for question in questions:
        start = time.perf_counter()
        hits: list[Hit] = retriever.retrieve(question.question, config.retrieval.top_k)
        latency_ms = (time.perf_counter() - start) * 1000

        window = hits[: config.retrieval.final_k]
        result = QuestionResult(
            question_id=question.id,
            kind=question.kind,
            provenance=question.provenance,
            retrieved=[h.chunk_id for h in hits],
            scores=[h.score for h in hits],
            gold=list(question.gold_chunk_ids),
            latency_ms=latency_ms,
            unsafe_retrieved=[h.chunk_id for h in window if h.is_unsafe_to_cite],
        )

        if generator is not None:
            start = time.perf_counter()
            generated = generator.answer(question.question, [h.article for h in window])
            result.answer_latency_ms = (time.perf_counter() - start) * 1000

            # Statuses come from the articles actually shown, so a citation of a
            # repealed article is detected even when the model reformatted it.
            shown = {h.chunk_id: h for h in window}
            cited = list(generated.cited_chunk_ids)

            result.answer = generated.text
            result.cited = cited
            result.abstained = generated.abstained
            result.cited_not_retrieved = [c for c in cited if c not in shown]
            result.cited_repealed = [c for c in cited if c in shown and shown[c].is_unsafe_to_cite]

        results.append(result)

    ks = tuple(config.eval.recall_at_k)
    final_k = config.retrieval.final_k
    headline_ids = {q.id for q in gold.headline()}
    headline = [r for r in results if r.question_id in headline_ids]

    from kora.retrieval.pipeline import describe

    report: dict[str, Any] = {
        "run_id": config.run_id,
        "name": config.name,
        "pipeline": describe(config),
        "index_fingerprint": config.index_fingerprint(),
        "corpus_size": corpus_size,
        "gold_composition": gold.composition(),
        "headline": _retrieval_metrics(headline, ks, final_k),
        "all_questions": _retrieval_metrics(results, ks, final_k),
        "by_kind": {
            kind: _retrieval_metrics([r for r in results if r.kind == kind], ks, final_k)
            for kind in sorted({r.kind for r in results})
        },
    }

    if generator is not None:
        report["answers"] = _answer_metrics(results)
        report["answers_headline"] = _answer_metrics(headline)

    # Recorded in the metrics file as well as the directory name, so a results
    # file read on its own still says how many questions stand behind it.
    report["questions_evaluated"] = len(questions)

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
    *,
    suffix: str = "",
) -> str:
    """Persist a run's config, metrics and per-question predictions.

    `suffix` distinguishes runs that share a config but not an evaluation --
    a smoke test over five questions produces the same `run_id` as the full
    sweep, and would otherwise overwrite it. The fingerprint describes the
    system; it cannot describe how much of the gold set was used.
    """
    directory = run_path(config.run_id + suffix)
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
