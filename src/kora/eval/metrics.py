"""Retrieval and answer metrics.

Pure functions over identifiers and labels: no models, no network, no corpus.
That is deliberate. Metrics are the instrument every later claim depends on, so
they must be trivially testable and impossible to get subtly wrong through a
dependency. If the ruler is wrong, every measurement taken with it is wrong, and
nothing downstream will tell you.

A note on what is *not* here
---------------------------
Lexical-overlap scoring, the usual shortcut for "relevance", is deliberately
absent. Off-the-shelf implementations filter English stopwords; on French legal
text that removes nothing and leaves scores dominated by `les`, `des`, `dans`,
`de la`. It produces numbers that look like measurements and are not. Retrieval
here is scored against gold article identifiers instead, which is unambiguous.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def _first_relevant_rank(retrieved: Sequence[str], gold: set[str]) -> int | None:
    """1-indexed rank of the first relevant item, or None."""
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in gold:
            return rank
    return None


def recall_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    """Fraction of gold articles present in the top k.

    Recall rather than precision is the primary retrieval metric here because
    the failure that matters is *missing* a governing provision. A legal answer
    built on four of the five relevant articles can be confidently wrong; one
    that also surfaced an irrelevant article costs the reader a few seconds.

    Multiple gold articles per question are the normal case, not an edge case:
    answering "how is a SA director appointed and for how long" requires two
    provisions, and a metric assuming a single gold answer would score a
    genuinely incomplete retrieval as perfect.
    """
    gold_set = set(gold)
    if not gold_set:
        raise ValueError("recall is undefined with no gold articles")
    if k <= 0:
        raise ValueError("k must be positive")
    hits = sum(1 for chunk_id in set(retrieved[:k]) if chunk_id in gold_set)
    return hits / len(gold_set)


def precision_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    """Fraction of the top k that is relevant.

    Reported alongside recall because it is what governs the generator's
    context window: at final_k=5, precision 0.2 means four of five passages are
    noise the model must ignore.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    gold_set = set(gold)
    window = retrieved[:k]
    if not window:
        return 0.0
    return sum(1 for chunk_id in window if chunk_id in gold_set) / len(window)


def reciprocal_rank(retrieved: Sequence[str], gold: Iterable[str]) -> float:
    """1 / rank of the first relevant article, or 0 if none was retrieved.

    Averaged over questions this is MRR. It answers a different question from
    recall -- not "did we find it" but "how far down" -- which matters because
    a reranker's whole job is to move the right passage upward without
    necessarily retrieving anything new.
    """
    rank = _first_relevant_rank(retrieved, set(gold))
    return 1.0 / rank if rank is not None else 0.0


def dcg(relevances: Sequence[float]) -> float:
    """Discounted cumulative gain with the standard log2(rank + 1) discount."""
    return sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevances, start=1))


def ndcg_at_k(
    retrieved: Sequence[str],
    gold: Iterable[str],
    k: int,
    *,
    graded: dict[str, float] | None = None,
) -> float:
    """Normalised discounted cumulative gain at k.

    The one metric that is sensitive to *ordering among the relevant results*.
    Recall treats a gold article at rank 1 and at rank 20 identically; nDCG does
    not. That matters here because only `final_k` passages reach the generator,
    so ranking is not cosmetic.

    `graded` allows non-binary relevance -- an article that directly answers the
    question can be weighted above one that merely bears on it. Absent it,
    relevance is binary.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    gold_set = set(gold)
    if not gold_set:
        raise ValueError("nDCG is undefined with no gold articles")

    grades = graded or dict.fromkeys(gold_set, 1.0)

    actual = [
        grades.get(chunk_id, 0.0) if chunk_id in gold_set else 0.0 for chunk_id in retrieved[:k]
    ]
    # The ideal ranking puts the highest-graded gold articles first.
    ideal = sorted((grades.get(chunk_id, 1.0) for chunk_id in gold_set), reverse=True)[:k]

    ideal_dcg = dcg(ideal)
    return dcg(actual) / ideal_dcg if ideal_dcg else 0.0


def hit_rate(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    """1.0 if any gold article appears in the top k, else 0.0.

    The most forgiving retrieval metric, and worth reporting precisely because
    of that: it separates "retrieval found nothing at all" from "retrieval found
    some of it", which recall alone blurs.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    gold_set = set(gold)
    return 1.0 if any(chunk_id in gold_set for chunk_id in retrieved[:k]) else 0.0


# ---------------------------------------------------------------------------
# Domain-specific answer metrics
# ---------------------------------------------------------------------------


def citation_recall(cited: Iterable[str], gold: Iterable[str]) -> float:
    """Fraction of the gold articles the answer actually cited.

    The primary citation metric, for the same reason recall leads on the
    retrieval side: failing to cite a governing provision is a legal error,
    while citing an additional relevant one is not.

    This distinction was not theoretical. Reading the first generated answers
    showed the model citing article 24 *and* article 25 on the seat of a
    company, or article 65 *and* article 387 on minimum capital -- in both
    cases a correct supplementary citation that precision alone scored as a
    mistake, because the gold set records the minimum articles needed rather
    than the only ones it is legitimate to cite.
    """
    gold_set = set(gold)
    if not gold_set:
        raise ValueError("citation recall is undefined with no gold articles")
    return sum(1 for chunk_id in gold_set if chunk_id in set(cited)) / len(gold_set)


def citation_precision(cited: Iterable[str], gold: Iterable[str]) -> float:
    """Fraction of the answer's citations that are correct.

    Distinct from retrieval precision: a system can retrieve well and still
    cite an article it did not use, or cite one that does not say what the
    answer claims. In a legal assistant a wrong citation is worse than no
    citation, because it is checkable and therefore trusted.
    """
    cited_list = list(cited)
    if not cited_list:
        return 0.0
    gold_set = set(gold)
    return sum(1 for c in cited_list if c in gold_set) / len(cited_list)


def repealed_citation_rate(cited_statuses: Iterable[str]) -> float:
    """Fraction of citations pointing at repealed law.

    The metric this corpus exists to support. An assistant that cites the 1997
    AUSCGIE for a question about company formation is not slightly wrong; it is
    giving advice under a text repealed in 2014. Because the manifest tracks
    legal status per document, this is measurable directly rather than by
    inspection -- and any number above zero is a defect, not a tuning target.
    """
    statuses = list(cited_statuses)
    if not statuses:
        return 0.0
    return sum(1 for s in statuses if s == "superseded") / len(statuses)


def abstention_scores(
    abstained: Sequence[bool],
    answerable: Sequence[bool],
) -> dict[str, float]:
    """How well the system declines to answer when the corpus cannot support it.

    Abstention is a two-sided failure and a single rate hides that:

      - answering an unanswerable question is a fabrication
      - abstaining on an answerable one is uselessness

    A system that always abstains scores perfectly on the first and worst on the
    second, so both are reported. `correct_abstention` is the fraction of
    unanswerable questions correctly declined; `false_abstention` the fraction
    of answerable questions wrongly declined.
    """
    if len(abstained) != len(answerable):
        raise ValueError("abstained and answerable must be the same length")

    unanswerable = [i for i, ok in enumerate(answerable) if not ok]
    answerable_idx = [i for i, ok in enumerate(answerable) if ok]

    return {
        "correct_abstention": (
            sum(1 for i in unanswerable if abstained[i]) / len(unanswerable)
            if unanswerable
            else float("nan")
        ),
        "false_abstention": (
            sum(1 for i in answerable_idx if abstained[i]) / len(answerable_idx)
            if answerable_idx
            else float("nan")
        ),
    }
