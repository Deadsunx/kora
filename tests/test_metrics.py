"""Tests for the evaluation metrics.

These matter more than most tests in the project. Every claim it eventually
makes is produced by these functions, so an error here does not cause a failure
-- it causes a wrong result that looks entirely plausible. Each metric is
therefore checked against hand-computed values, not against itself.
"""

from __future__ import annotations

import math

import pytest

from kora.eval.metrics import (
    abstention_scores,
    citation_precision,
    dcg,
    hit_rate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    repealed_citation_rate,
)

# A retrieval result with gold articles at ranks 2 and 5.
RETRIEVED = ["a", "GOLD1", "b", "c", "GOLD2", "d"]
GOLD = {"GOLD1", "GOLD2"}


# -- recall -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("k", "expected"),
    [(1, 0.0), (2, 0.5), (4, 0.5), (5, 1.0), (6, 1.0), (100, 1.0)],
)
def test_recall_at_k(k: int, expected: float) -> None:
    assert recall_at_k(RETRIEVED, GOLD, k) == expected


def test_recall_counts_distinct_gold_articles_not_duplicates() -> None:
    """A retriever returning the same article twice has not found two."""
    assert recall_at_k(["GOLD1", "GOLD1", "GOLD1"], GOLD, 3) == 0.5


def test_recall_requires_gold() -> None:
    with pytest.raises(ValueError, match="undefined"):
        recall_at_k(RETRIEVED, [], 5)


@pytest.mark.parametrize("k", [0, -1])
def test_recall_rejects_nonpositive_k(k: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        recall_at_k(RETRIEVED, GOLD, k)


# -- precision --------------------------------------------------------------


@pytest.mark.parametrize(("k", "expected"), [(1, 0.0), (2, 0.5), (5, 0.4), (6, 2 / 6)])
def test_precision_at_k(k: int, expected: float) -> None:
    assert precision_at_k(RETRIEVED, GOLD, k) == pytest.approx(expected)


def test_precision_of_empty_retrieval() -> None:
    assert precision_at_k([], GOLD, 5) == 0.0


# -- MRR --------------------------------------------------------------------


def test_reciprocal_rank_uses_first_hit() -> None:
    assert reciprocal_rank(RETRIEVED, GOLD) == pytest.approx(1 / 2)


def test_reciprocal_rank_rank_one() -> None:
    assert reciprocal_rank(["GOLD1", "x"], GOLD) == 1.0


def test_reciprocal_rank_no_hit() -> None:
    assert reciprocal_rank(["x", "y"], GOLD) == 0.0


# -- nDCG -------------------------------------------------------------------


def test_dcg_matches_hand_computation() -> None:
    # rank 1 discount log2(2)=1, rank 2 log2(3), rank 3 log2(4)=2
    assert dcg([1.0, 1.0, 1.0]) == pytest.approx(1 + 1 / math.log2(3) + 1 / 2)


def test_ndcg_is_one_when_gold_is_ranked_first() -> None:
    assert ndcg_at_k(["GOLD1", "GOLD2", "x"], GOLD, 3) == pytest.approx(1.0)


def test_ndcg_penalises_lower_ranking() -> None:
    """The property recall cannot see: same articles found, worse order."""
    good = ndcg_at_k(["GOLD1", "GOLD2", "x", "y"], GOLD, 4)
    bad = ndcg_at_k(["x", "y", "GOLD1", "GOLD2"], GOLD, 4)
    assert good > bad
    # Recall is blind to the difference, which is exactly why nDCG is reported.
    assert recall_at_k(["GOLD1", "GOLD2", "x", "y"], GOLD, 4) == recall_at_k(
        ["x", "y", "GOLD1", "GOLD2"], GOLD, 4
    )


def test_ndcg_zero_when_nothing_relevant_retrieved() -> None:
    assert ndcg_at_k(["x", "y"], GOLD, 2) == 0.0


def test_ndcg_with_graded_relevance() -> None:
    """A highly relevant article ranked first should beat the reverse order."""
    grades = {"GOLD1": 3.0, "GOLD2": 1.0}
    better = ndcg_at_k(["GOLD1", "GOLD2"], GOLD, 2, graded=grades)
    worse = ndcg_at_k(["GOLD2", "GOLD1"], GOLD, 2, graded=grades)
    assert better == pytest.approx(1.0)
    assert worse < better


# -- hit rate ---------------------------------------------------------------


def test_hit_rate_is_binary() -> None:
    assert hit_rate(RETRIEVED, GOLD, 2) == 1.0
    assert hit_rate(RETRIEVED, GOLD, 1) == 0.0


# -- citations --------------------------------------------------------------


def test_citation_precision() -> None:
    assert citation_precision(["GOLD1", "wrong"], GOLD) == 0.5
    assert citation_precision(["GOLD1", "GOLD2"], GOLD) == 1.0


def test_citation_precision_of_no_citations_is_zero() -> None:
    """An uncited legal answer earns no credit, however correct its prose."""
    assert citation_precision([], GOLD) == 0.0


def test_repealed_citation_rate() -> None:
    assert repealed_citation_rate(["in_force", "superseded"]) == 0.5
    assert repealed_citation_rate(["in_force", "in_force"]) == 0.0
    assert repealed_citation_rate([]) == 0.0


# -- abstention -------------------------------------------------------------


def test_abstention_two_sided() -> None:
    # q0, q1 answerable; q2, q3 unanswerable.
    scores = abstention_scores(
        abstained=[False, True, True, False],
        answerable=[True, True, False, False],
    )
    assert scores["correct_abstention"] == 0.5  # one of two unanswerable declined
    assert scores["false_abstention"] == 0.5  # one of two answerable wrongly declined


def test_always_abstaining_is_caught_by_the_second_number() -> None:
    """The reason abstention is reported as two numbers rather than one."""
    scores = abstention_scores(abstained=[True, True], answerable=[True, False])
    assert scores["correct_abstention"] == 1.0  # looks perfect
    assert scores["false_abstention"] == 1.0  # and is useless


def test_abstention_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        abstention_scores(abstained=[True], answerable=[True, False])


def test_abstention_with_no_unanswerable_questions_is_nan() -> None:
    """Undefined rather than 0.0 or 1.0: the set could not test it."""
    scores = abstention_scores(abstained=[False], answerable=[True])
    assert math.isnan(scores["correct_abstention"])
    assert scores["false_abstention"] == 0.0
