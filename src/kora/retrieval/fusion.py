"""Reciprocal rank fusion.

Why fuse ranks rather than scores
---------------------------------
Cosine similarity lives in [-1, 1] and clusters tightly around 0.8 for this
corpus. BM25 is unbounded and depends on document length and corpus statistics;
scores of 4 and 22 are both routine. Combining them numerically requires
normalising two distributions that have no shared meaning, and every choice of
normalisation is a hidden hyperparameter that will quietly shape the ablation.

RRF sidesteps this entirely by using only the *ordering* each retriever
produces::

    score(d) = sum over retrievers of  1 / (k + rank(d))

A document ranked 1st by either retriever gets a large contribution; one ranked
50th gets almost nothing. Nothing about the two score scales enters the
calculation, so the fusion cannot be accidentally dominated by whichever
retriever happens to produce larger numbers.

The constant `k` (60 by convention, from Cormack et al. 2009) flattens the
difference between the top few ranks. Smaller k makes rank 1 dominate; larger k
makes the fusion closer to a plain vote. It is in the config so it can be
ablated rather than trusted.
"""

from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse several ranked id lists into one, best first.

    Args:
        rankings: One ranked list of chunk ids per retriever, best first.
        k: RRF constant. Larger values flatten the advantage of the top ranks.
        weights: Optional per-retriever weight. Defaults to equal weighting.
            Present so an ablation can ask whether dense and lexical deserve
            equal say, rather than assuming they do.

    Returns:
        (chunk_id, fused_score) pairs sorted by descending score.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if weights is not None and len(weights) != len(rankings):
        raise ValueError("weights must have one entry per ranking")

    fused: dict[str, float] = {}
    for index, ranking in enumerate(rankings):
        weight = 1.0 if weights is None else weights[index]
        for rank, chunk_id in enumerate(ranking, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + weight / (k + rank)

    # Ties broken by chunk id so the output is deterministic. Two documents with
    # identical fused scores are genuinely indistinguishable to RRF, and letting
    # dict ordering decide would make runs irreproducible for no reason.
    return sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))
