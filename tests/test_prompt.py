"""Tests for prompt construction and answer parsing.

No model required. Everything here decides what the generator is asked and how
its output is scored, so a bug produces a plausible wrong number rather than an
error -- the same reason the retrieval metrics are tested this closely.
"""

from __future__ import annotations

import pytest

from kora.config import ExperimentConfig
from kora.documents import Article
from kora.generation.prompt import (
    ABSTENTION_TOKEN,
    build_context,
    build_messages,
    extract_citations,
    is_abstention,
)


def article(**overrides) -> Article:
    base = {
        "document_id": "AUSCGIE-2014",
        "abbrev": "AUSCGIE",
        "status": "in_force",
        "number": "477",
        "text": "Le conseil d'administration designe parmi ses membres un president.",
        "page_start": 0,
        "page_end": 0,
    }
    return Article.model_validate({**base, **overrides})


# -- citation extraction ----------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Selon l'article 477 AUSCGIE (2014), le conseil designe un president.",
        "Voir article 477 AUSCGIE(2014).",
        "art. 477 AUSCGIE ( 2014 )",
        "Article 477 AUSCGIE (2014)",
    ],
)
def test_citation_formats(text: str) -> None:
    citations = extract_citations(text)
    assert len(citations) == 1
    assert citations[0].chunk_id == "AUSCGIE-2014#art477"


def test_inserted_article_numbers_are_cited_correctly() -> None:
    """`133-1` must not be truncated to `133` -- a different provision."""
    citations = extract_citations("Voir article 133-1 AUSCGIE (2014).")
    assert citations[0].chunk_id == "AUSCGIE-2014#art133-1"


def test_year_distinguishes_versions() -> None:
    """The whole point: the 1997 and 2014 acts must not collapse together."""
    citations = extract_citations(
        "L'article 1 AUSCGIE (1997) a ete remplace par l'article 1 AUSCGIE (2014)."
    )
    assert {c.chunk_id for c in citations} == {
        "AUSCGIE-1997#art1",
        "AUSCGIE-2014#art1",
    }


def test_citations_are_deduplicated_but_ordered() -> None:
    citations = extract_citations(
        "article 5 AUA (2017) puis article 2 AUA (2017) puis encore article 5 AUA (2017)"
    )
    assert [c.number for c in citations] == ["5", "2"]


def test_uncited_answer_yields_nothing() -> None:
    assert extract_citations("Le conseil designe un president parmi ses membres.") == []


def test_citation_without_year_is_not_extracted() -> None:
    """A citation that cannot be resolved to a version is not a usable citation."""
    assert extract_citations("Voir l'article 477 AUSCGIE.") == []


# -- abstention -------------------------------------------------------------


def test_abstention_detected_at_start() -> None:
    assert is_abstention(f"{ABSTENTION_TOKEN} : les extraits ne traitent pas des sanctions.")


def test_abstention_is_case_insensitive_and_tolerates_whitespace() -> None:
    assert is_abstention("  insuffisant : rien dans les extraits.")


def test_answer_mentioning_insufficiency_later_is_not_an_abstention() -> None:
    """An answer that states a rule and then hedges is an answer."""
    text = (
        "Le capital minimum est de dix millions de francs CFA, article 387 AUSCGIE (2014). "
        "Les extraits sont insuffisants sur le cas camerounais."
    )
    assert is_abstention(text) is False
    assert len(extract_citations(text)) == 1


# -- context construction ---------------------------------------------------


def test_context_carries_the_exact_citation_string() -> None:
    """The model should copy the citation, not reconstruct it."""
    context = build_context([article()], max_chars=1200)
    assert "article 477 AUSCGIE (2014)" in context


def test_repealed_articles_are_marked() -> None:
    context = build_context([article(number="12", repealed=True)], max_chars=1200)
    assert "ABROGÉ" in context


def test_superseded_documents_are_marked() -> None:
    context = build_context(
        [article(document_id="AUSCGIE-1997", status="superseded")], max_chars=1200
    )
    assert "ABROGÉ" in context


def test_in_force_articles_are_not_marked() -> None:
    assert "ABROGÉ" not in build_context([article()], max_chars=1200)


def test_long_passages_are_truncated_on_a_word_boundary() -> None:
    long_article = article(text="mot " * 500)
    context = build_context([long_article], max_chars=100)
    assert "[…]" in context
    assert len(context) < 300


def test_rubric_is_included_when_present() -> None:
    context = build_context([article(rubric="Nomination du president")], max_chars=1200)
    assert "Nomination du president" in context


# -- messages ---------------------------------------------------------------


def test_messages_have_system_and_user_roles() -> None:
    config = ExperimentConfig(name="t")
    messages = build_messages("Qui preside le conseil ?", [article()], config)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "Qui preside le conseil ?" in messages[1]["content"]
    assert "article 477 AUSCGIE (2014)" in messages[1]["content"]


def test_abstention_instruction_present_by_default() -> None:
    config = ExperimentConfig(name="t")
    system = build_messages("q ?", [article()], config)[0]["content"]
    assert ABSTENTION_TOKEN in system


def test_abstention_instruction_removed_when_disabled() -> None:
    """The ablation: removing the option should raise coverage and fabrication together."""
    config = ExperimentConfig.model_validate(
        {"name": "t", "generator": {"allow_abstention": False}}
    )
    system = build_messages("q ?", [article()], config)[0]["content"]
    assert ABSTENTION_TOKEN not in system
    # The grounding and citation rules must survive.
    assert "UNIQUEMENT" in system
    assert "ABROGÉ" in system
