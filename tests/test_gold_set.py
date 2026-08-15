"""Tests for the gold question set schema.

The schema's job is to make the two classic evaluation frauds impossible to
commit accidentally: questions that leak their own answer, and a set made
entirely of easy lookups. Both produce excellent numbers and mean nothing, so
the constraints are enforced rather than documented.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from kora.eval.dataset import GoldQuestion, GoldSet, load_gold_set, save_gold_set


def question(**overrides: object) -> dict:
    base = {
        "id": "q1",
        "question": "Comment est nomme le president du conseil d'administration ?",
        "kind": "lookup",
        "provenance": "human",
        "gold_chunk_ids": ["AUSCGIE-2014#art477"],
    }
    return {**base, **overrides}


# -- answerability ----------------------------------------------------------


def test_answerable_question_requires_gold_articles() -> None:
    with pytest.raises(ValidationError, match="no gold articles"):
        GoldQuestion.model_validate(question(gold_chunk_ids=[]))


def test_unanswerable_question_must_have_no_gold_articles() -> None:
    with pytest.raises(ValidationError, match="unanswerable question has gold"):
        GoldQuestion.model_validate(question(kind="unanswerable"))


def test_unanswerable_question_is_valid_without_gold() -> None:
    q = GoldQuestion.model_validate(question(kind="unanswerable", gold_chunk_ids=[], id="q-unans"))
    assert q.is_answerable is False


# -- kinds must match their evidence ----------------------------------------


def test_lookup_cannot_have_several_articles() -> None:
    with pytest.raises(ValidationError, match="implies one article"):
        GoldQuestion.model_validate(
            question(gold_chunk_ids=["AUSCGIE-2014#art477", "AUSCGIE-2014#art478"])
        )


def test_multi_hop_requires_at_least_two() -> None:
    with pytest.raises(ValidationError, match="requires at least two"):
        GoldQuestion.model_validate(question(kind="multi_hop"))


def test_cross_act_requires_two_different_acts() -> None:
    """Two articles of the same act are multi-hop, not cross-act."""
    with pytest.raises(ValidationError, match="articles from two acts"):
        GoldQuestion.model_validate(
            question(
                kind="cross_act",
                gold_chunk_ids=["AUSCGIE-2014#art477", "AUSCGIE-2014#art478"],
            )
        )


def test_cross_act_accepts_articles_from_two_acts() -> None:
    q = GoldQuestion.model_validate(
        question(
            kind="cross_act",
            gold_chunk_ids=["AUSCGIE-2014#art477", "AUS-2010#art1"],
        )
    )
    assert len(q.gold_chunk_ids) == 2


# -- grades -----------------------------------------------------------------


def test_grades_must_reference_gold_articles() -> None:
    with pytest.raises(ValidationError, match="non-gold"):
        GoldQuestion.model_validate(question(grades={"AUSCGIE-2014#art999": 2.0}))


# -- provenance and headline eligibility ------------------------------------


@pytest.mark.parametrize(
    ("provenance", "eligible"),
    [
        ("human", True),
        ("llm_drafted_human_validated", True),
        ("llm_drafted", False),
    ],
)
def test_headline_eligibility_follows_provenance(provenance: str, eligible: bool) -> None:
    """Unvalidated model-drafted questions may not appear in a reported result.

    They leak: a question generated from a passage is retrieved by that passage
    almost by construction. Keeping them in the file but out of the headline is
    the distinction the schema encodes.
    """
    q = GoldQuestion.model_validate(question(provenance=provenance))
    assert q.counts_towards_headline is eligible


def test_headline_filters_the_set() -> None:
    gold = GoldSet(
        version=1,
        questions=(
            GoldQuestion.model_validate(question(id="a", provenance="human")),
            GoldQuestion.model_validate(question(id="b", provenance="llm_drafted")),
        ),
    )
    assert len(gold) == 2
    assert [q.id for q in gold.headline()] == ["a"]


# -- set-level invariants ---------------------------------------------------


def test_duplicate_ids_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        GoldSet(
            version=1,
            questions=(
                GoldQuestion.model_validate(question(id="dup")),
                GoldQuestion.model_validate(question(id="dup")),
            ),
        )


def test_composition_reports_the_shape_of_the_set() -> None:
    gold = GoldSet(
        version=1,
        questions=(
            GoldQuestion.model_validate(question(id="a")),
            GoldQuestion.model_validate(
                question(
                    id="b",
                    kind="multi_hop",
                    gold_chunk_ids=["AUSCGIE-2014#art477", "AUSCGIE-2014#art478"],
                )
            ),
            GoldQuestion.model_validate(question(id="c", kind="unanswerable", gold_chunk_ids=[])),
        ),
    )
    composition = gold.composition()
    assert composition["total"] == 3
    assert composition["kind:lookup"] == 1
    assert composition["kind:multi_hop"] == 1
    assert composition["kind:unanswerable"] == 1
    assert composition["headline_eligible"] == 3


def test_referenced_documents() -> None:
    gold = GoldSet(
        version=1,
        questions=(
            GoldQuestion.model_validate(question(id="a")),
            GoldQuestion.model_validate(
                question(
                    id="b",
                    kind="cross_act",
                    gold_chunk_ids=["AUS-2010#art1", "AUDCG-2010#art2"],
                )
            ),
        ),
    )
    assert gold.referenced_documents() == {"AUSCGIE-2014", "AUS-2010", "AUDCG-2010"}


# -- round trip -------------------------------------------------------------


def test_jsonl_round_trip(tmp_path) -> None:
    gold = GoldSet(
        version=1,
        questions=(
            GoldQuestion.model_validate(question(id="a")),
            GoldQuestion.model_validate(question(id="b", provenance="llm_drafted")),
        ),
    )
    path = save_gold_set(gold, tmp_path / "gold.jsonl")
    reloaded = load_gold_set(path)
    assert [q.id for q in reloaded] == ["a", "b"]
    assert reloaded.questions[0].gold_chunk_ids == ("AUSCGIE-2014#art477",)


def test_jsonl_is_one_question_per_line(tmp_path) -> None:
    """Keeps adding a question to a one-line diff, so review stays readable."""
    gold = GoldSet(
        version=1,
        questions=tuple(GoldQuestion.model_validate(question(id=f"q{n}")) for n in range(3)),
    )
    path = save_gold_set(gold, tmp_path / "gold.jsonl")
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 3
