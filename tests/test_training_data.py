"""Tests for training-data construction.

The first test is the one that matters. Training on articles that appear in the
gold set would make every number this project reports meaningless, and it would
do so invisibly -- the metrics would simply improve. It is therefore enforced in
code and asserted here rather than left to care.
"""

from __future__ import annotations

from kora.documents import Article
from kora.eval.dataset import GoldQuestion, GoldSet
from kora.generation.prompt import ABSTENTION_TOKEN, extract_citations, is_abstention
from kora.training.dataset import build_dataset, save_dataset


def make_articles(count: int = 60) -> list[Article]:
    articles = []
    for index in range(1, count + 1):
        articles.append(
            Article(
                document_id="AUSCGIE-2014",
                abbrev="AUSCGIE",
                status="in_force",
                number=str(index),
                text=(
                    f"La règle numéro {index} dispose que toute société commerciale "
                    f"est soumise aux dispositions du présent Acte uniforme. "
                    f"Elle précise également les modalités applicables."
                ),
                page_start=0,
                page_end=0,
            )
        )
    # A repealed article, so the third example type has something to work with.
    articles.append(
        Article(
            document_id="AUDCIF-2017",
            abbrev="AUDCIF",
            status="in_force",
            number="12",
            text="Abrogé",
            repealed=True,
            page_start=0,
            page_end=0,
        )
    )
    return articles


def make_gold(chunk_ids: list[str]) -> GoldSet:
    return GoldSet(
        version=1,
        questions=tuple(
            GoldQuestion(
                id=f"q{index}",
                question="Une question d'évaluation suffisamment longue ?",
                kind="lookup",
                provenance="human",
                gold_chunk_ids=(chunk_id,),
            )
            for index, chunk_id in enumerate(chunk_ids)
        ),
    )


# -- leakage ----------------------------------------------------------------


def test_gold_articles_never_appear_in_training_data() -> None:
    """The one mistake that would invalidate every result in the project."""
    articles = make_articles()
    held_out = ["AUSCGIE-2014#art1", "AUSCGIE-2014#art2", "AUSCGIE-2014#art3"]
    gold = make_gold(held_out)

    examples = build_dataset(articles, gold, n_cite=40, n_abstain=10, n_repealed=10)
    assert examples

    forbidden = set(held_out)
    for example in examples:
        rendered = " ".join(m["content"] for m in example.messages)
        for chunk_id in forbidden:
            number = chunk_id.split("#art")[1]
            assert f"article {number} AUSCGIE (2014)" not in rendered


def test_excluding_everything_yields_nothing_rather_than_leaking() -> None:
    articles = make_articles(5)
    gold = make_gold([a.chunk_id for a in articles])
    examples = build_dataset(articles, gold, n_cite=10, n_abstain=0, n_repealed=0)
    assert examples == []


# -- shape ------------------------------------------------------------------


def test_every_example_is_a_three_turn_chat() -> None:
    examples = build_dataset(make_articles(), make_gold([]), n_cite=20, n_abstain=5, n_repealed=5)
    for example in examples:
        assert [m["role"] for m in example.messages] == ["system", "user", "assistant"]
        assert example.messages[2]["content"].strip()


def test_all_three_kinds_are_produced() -> None:
    examples = build_dataset(make_articles(), make_gold([]), n_cite=20, n_abstain=5, n_repealed=5)
    assert {e.kind for e in examples} == {"cite", "abstain", "repealed"}


# -- the contract the adapter is meant to learn -----------------------------


def test_cite_examples_produce_parseable_citations() -> None:
    """A target the citation parser cannot read teaches an unscoreable format."""
    examples = build_dataset(make_articles(), make_gold([]), n_cite=30, n_abstain=0, n_repealed=0)
    for example in examples:
        answer = example.messages[2]["content"]
        assert extract_citations(answer), f"no parseable citation in: {answer!r}"


def test_abstain_examples_start_with_the_sentinel() -> None:
    examples = build_dataset(make_articles(), make_gold([]), n_cite=0, n_abstain=20, n_repealed=0)
    assert examples
    for example in examples:
        assert is_abstention(example.messages[2]["content"])


def test_abstain_examples_cite_nothing() -> None:
    """An abstention that cites an article is not an abstention."""
    examples = build_dataset(make_articles(), make_gold([]), n_cite=0, n_abstain=20, n_repealed=0)
    for example in examples:
        assert extract_citations(example.messages[2]["content"]) == []


def test_repealed_examples_cite_the_in_force_article_first() -> None:
    examples = build_dataset(make_articles(), make_gold([]), n_cite=0, n_abstain=0, n_repealed=15)
    assert examples
    for example in examples:
        answer = example.messages[2]["content"]
        citations = extract_citations(answer)
        assert citations, answer
        # The in-force article leads; the repealed one is named as abrogated.
        assert citations[0].abbrev == "AUSCGIE"
        assert "abrogé" in answer.lower()


def test_context_marks_the_repealed_article() -> None:
    examples = build_dataset(make_articles(), make_gold([]), n_cite=0, n_abstain=0, n_repealed=10)
    for example in examples:
        assert "ABROGÉ" in example.messages[1]["content"]


# -- determinism and persistence --------------------------------------------


def test_same_seed_gives_the_same_dataset() -> None:
    articles, gold = make_articles(), make_gold([])
    first = build_dataset(articles, gold, n_cite=20, n_abstain=5, n_repealed=5, seed=7)
    second = build_dataset(articles, gold, n_cite=20, n_abstain=5, n_repealed=5, seed=7)
    assert [e.to_json() for e in first] == [e.to_json() for e in second]


def test_different_seeds_differ() -> None:
    articles, gold = make_articles(), make_gold([])
    first = build_dataset(articles, gold, n_cite=20, n_abstain=5, n_repealed=5, seed=1)
    second = build_dataset(articles, gold, n_cite=20, n_abstain=5, n_repealed=5, seed=2)
    assert [e.to_json() for e in first] != [e.to_json() for e in second]


def test_saved_as_one_example_per_line(tmp_path) -> None:
    examples = build_dataset(make_articles(), make_gold([]), n_cite=10, n_abstain=3, n_repealed=3)
    path = save_dataset(examples, tmp_path / "train.jsonl")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(examples)


def test_abstention_sentinel_is_the_one_the_parser_looks_for() -> None:
    """Guards against the training target and the parser drifting apart."""
    examples = build_dataset(make_articles(), make_gold([]), n_cite=0, n_abstain=5, n_repealed=0)
    assert all(ABSTENTION_TOKEN in e.messages[2]["content"] for e in examples)
