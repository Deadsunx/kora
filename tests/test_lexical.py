"""Tests for French tokenisation and rank fusion.

The tokeniser decides what BM25 can ever match. A bug here does not raise; it
quietly removes a term from the index and shows up much later as "lexical
retrieval didn't help much", which is indistinguishable from the hypothesis
being wrong.
"""

from __future__ import annotations

import pytest

from kora.retrieval.fusion import reciprocal_rank_fusion
from kora.retrieval.lexical import fold_accents, tokenize_fr

# -- accent folding ---------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("société", "societe"),
        ("sûreté", "surete"),
        ("exécution", "execution"),
        ("créancier", "creancier"),
        ("À", "A"),
        ("hypothèque", "hypotheque"),
    ],
)
def test_fold_accents(text: str, expected: str) -> None:
    assert fold_accents(text) == expected


def test_folding_makes_unaccented_queries_match() -> None:
    """A user typing `surete` must reach an article containing `sûreté`."""
    assert tokenize_fr("sûreté réelle") == tokenize_fr("surete reelle")


# -- elision ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("l'article", ["article"]),
        ("d'un contrat", ["contrat"]),
        ("qu'il", []),  # `il` is a stopword, `qu` is dropped
        ("s'applique", ["applique"]),
        ("l'acte uniforme", ["acte", "uniforme"]),
        ("jusqu'à expiration", ["expiration"]),
        ("lorsqu'une saisie", ["saisie"]),
    ],
)
def test_elision_is_split(text: str, expected: list[str]) -> None:
    assert tokenize_fr(text) == expected


def test_typographic_apostrophe_behaves_like_ascii() -> None:
    """These PDFs use U+2019 as often as the ASCII apostrophe."""
    assert tokenize_fr("l\u2019article 133") == tokenize_fr("l'article 133")


def test_elided_query_matches_unelided_document() -> None:
    """The failure this exists to prevent: `l'article` never matching `article`."""
    document = tokenize_fr("Le présent article s'applique aux sociétés")
    query = tokenize_fr("l'article")
    assert set(query) & set(document) == {"article"}


# -- article numbers --------------------------------------------------------


def test_compound_article_numbers_survive() -> None:
    """`133-1` must stay one token: splitting it yields a different provision."""
    assert "133-1" in tokenize_fr("Article 133-1")
    assert "133" not in tokenize_fr("Article 133-1")


def test_plain_and_compound_numbers_are_distinct() -> None:
    assert tokenize_fr("article 133") != tokenize_fr("article 133-1")


def test_multi_level_numbering() -> None:
    assert "245-6" in tokenize_fr("l'article 245-6 du présent acte")


# -- stopwords --------------------------------------------------------------


def test_common_words_are_dropped() -> None:
    assert tokenize_fr("le la les de des du") == []


@pytest.mark.parametrize("word", ["contre", "sans", "sous", "avant", "apres"])
def test_legally_meaningful_words_are_kept(word: str) -> None:
    """An aggressive stoplist would remove these, and they carry real meaning.

    "saisie sans commandement prealable" and "recours contre la caution" both
    turn on a word a generic French stoplist discards.
    """
    assert word in tokenize_fr(f"recours {word} la caution")


def test_realistic_legal_sentence() -> None:
    tokens = tokenize_fr(
        "Tout créancier muni d'un titre exécutoire constatant une créance "
        "liquide et exigible peut, sans commandement préalable, saisir "
        "l'article 153."
    )
    for expected in ("creancier", "titre", "executoire", "liquide", "exigible"):
        assert expected in tokens
    assert "sans" in tokens
    assert "153" in tokens


# -- reciprocal rank fusion -------------------------------------------------


def test_rrf_rewards_agreement() -> None:
    """A document both retrievers rank highly must beat one only either likes."""
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["a", "c", "b"]], k=60))
    assert fused["a"] > fused["b"]
    assert fused["a"] > fused["c"]


def test_rrf_ignores_score_scale() -> None:
    """The property that makes fusion possible at all.

    Only orderings enter the calculation, so a retriever producing large raw
    scores cannot dominate one producing small ones.
    """
    fused = reciprocal_rank_fusion([["x", "y"], ["y", "x"]], k=60)
    scores = dict(fused)
    assert scores["x"] == pytest.approx(scores["y"])


def test_rrf_single_ranking_preserves_order() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"]], k=60)
    assert [chunk_id for chunk_id, _ in fused] == ["a", "b", "c"]


def test_rrf_k_controls_top_rank_advantage() -> None:
    """Smaller k sharpens the advantage of rank 1; larger k flattens it."""
    sharp = dict(reciprocal_rank_fusion([["a", "b"]], k=1))
    flat = dict(reciprocal_rank_fusion([["a", "b"]], k=1000))
    assert sharp["a"] / sharp["b"] > flat["a"] / flat["b"]


def test_rrf_weights_apply() -> None:
    equal = dict(reciprocal_rank_fusion([["a"], ["b"]], k=60))
    weighted = dict(reciprocal_rank_fusion([["a"], ["b"]], k=60, weights=[2.0, 1.0]))
    assert equal["a"] == pytest.approx(equal["b"])
    assert weighted["a"] > weighted["b"]


def test_rrf_is_deterministic_on_ties() -> None:
    """Tied documents must not depend on dict ordering, or runs stop reproducing."""
    first = reciprocal_rank_fusion([["b", "a"], ["a", "b"]], k=60)
    second = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=60)
    assert [c for c, _ in first] == [c for c, _ in second]


def test_rrf_rejects_mismatched_weights() -> None:
    with pytest.raises(ValueError, match="one entry per ranking"):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])
