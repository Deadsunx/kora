"""Agent tests.

The parsing rules get the attention here, because they are where an agent
quietly stops being an agent. A decomposition that returns the question back
unchanged, or a verification that reads a polite refusal as a search query,
produces a system that looks agentic in the logs and behaves like noise in the
metrics. Every one of these is a case that a real model produced or plausibly
will.

The retrievers are tested against a stub model, so what is verified is the loop:
that fallbacks fall back, that a failed model call degrades to plain retrieval,
and that an atomic question comes out bit-identical to the non-agentic system.
"""

from __future__ import annotations

import pytest

from kora.agent.prompts import parse_subquestions, parse_verdict
from kora.agent.retriever import DecomposingRetriever, VerifyingRetriever
from kora.config import AgentConfig, ExperimentConfig, RerankerConfig
from kora.documents import Article
from kora.retrieval.dense import Hit

ORIGINAL = "Quelles sont les conditions de validité d'une hypothèque conventionnelle ?"


# --- decomposition parsing -------------------------------------------------


def test_atomic_reply_yields_no_subquestions():
    assert parse_subquestions("ATOMIQUE", original=ORIGINAL, max_subquestions=3) == []


def test_atomic_reply_tolerates_trailing_prose():
    """Models add a sentence however firmly they are told not to."""
    reply = "ATOMIQUE\nUn seul article suffit ici."
    assert parse_subquestions(reply, original=ORIGINAL, max_subquestions=3) == []


def test_two_subquestions_are_parsed():
    reply = "Qu'est-ce qu'une hypothèque conventionnelle ?\nQuelles formes doit-elle respecter ?"
    assert len(parse_subquestions(reply, original=ORIGINAL, max_subquestions=3)) == 2


def test_list_furniture_is_stripped():
    reply = "1. Qu'est-ce qu'une hypothèque conventionnelle ?\n2) Quelles formes respecter ?"
    parsed = parse_subquestions(reply, original=ORIGINAL, max_subquestions=3)
    assert parsed[0].startswith("Qu'est-ce")
    assert parsed[1].startswith("Quelles")


def test_a_single_subquestion_is_rejected():
    """One part is not a decomposition -- it is query rewriting.

    A different technique with a different hypothesis. Letting it through here
    would mean the decomposition experiment silently measured two things.
    """
    reply = "Quelles sont les conditions de validité d'une hypothèque ?"
    assert parse_subquestions(reply, original=ORIGINAL, max_subquestions=3) == []


def test_restating_the_question_is_not_a_decomposition():
    reply = f"{ORIGINAL}\n{ORIGINAL}"
    assert parse_subquestions(reply, original=ORIGINAL, max_subquestions=3) == []


def test_subquestions_are_capped():
    reply = "\n".join(f"Question numéro {n} sur le droit ?" for n in range(6))
    assert len(parse_subquestions(reply, original=ORIGINAL, max_subquestions=3)) == 3


def test_empty_reply_yields_no_subquestions():
    assert parse_subquestions("   \n  ", original=ORIGINAL, max_subquestions=3) == []


def test_duplicate_subquestions_are_dropped():
    """A repeated sub-question must not get two votes in the fusion."""
    reply = "Qu'est-ce qu'une hypothèque ?\nQu'est-ce qu'une hypothèque ?\nQuelles formes exiger ?"
    parsed = parse_subquestions(reply, original=ORIGINAL, max_subquestions=3)
    assert parsed == ["Qu'est-ce qu'une hypothèque ?", "Quelles formes exiger ?"]


def test_fragments_too_short_to_be_questions_are_dropped():
    reply = "Qu'est-ce qu'une hypothèque conventionnelle ?\n?\n-\nQuelles formes exiger ?"
    assert len(parse_subquestions(reply, original=ORIGINAL, max_subquestions=3)) == 2


# --- verification parsing --------------------------------------------------


def test_sufficient_verdict_returns_none():
    assert parse_verdict("SUFFISANT") is None


def test_unparseable_verdict_is_read_as_sufficient():
    """Silence must not trigger a search on noise."""
    assert parse_verdict("") is None
    assert parse_verdict("...") is None


def test_missing_information_becomes_a_followup_query():
    assert parse_verdict("inscription au registre du commerce, publicité foncière") is not None


def test_followup_strips_list_furniture():
    assert parse_verdict("- publicité foncière et inscription") == "publicité foncière et inscription"


# --- the loop --------------------------------------------------------------


def make_article(number: str) -> Article:
    return Article(
        document_id="AUS-2010",
        abbrev="AUS",
        status="in_force",
        number=number,
        rubric="",
        text=f"Texte de l'article {number}.",
        page_start=1,
        page_end=1,
    )


class StubInner:
    """Returns a different ranking per query, so fusion is observable."""

    def __init__(self, by_query: dict[str, list[str]], default: list[str]) -> None:
        self.by_query = by_query
        self.default = default
        self.queries: list[str] = []

    def retrieve(self, question: str, k: int) -> list[Hit]:
        self.queries.append(question)
        numbers = self.by_query.get(question, self.default)
        return [
            Hit(chunk_id=f"AUS-2010#art{n}", score=1.0 - i / 100, article=make_article(n))
            for i, n in enumerate(numbers)
        ][:k]


class StubModel:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls = 0

    def chat(self, messages, *, max_new_tokens=None) -> str:
        self.calls += 1
        return self.replies.pop(0) if self.replies else "ATOMIQUE"


class ExplodingModel:
    def chat(self, messages, *, max_new_tokens=None) -> str:
        raise RuntimeError("tokenizer blew up")


def agent_config(**kwargs) -> ExperimentConfig:
    return ExperimentConfig(
        name="agent-test",
        reranker=RerankerConfig(enabled=True),
        agent=AgentConfig(enabled=True, **kwargs),
    )


def test_atomic_question_is_identical_to_plain_retrieval():
    """The control. If this drifts, every lookup number is fusion noise."""
    inner = StubInner({}, default=["1", "2", "3"])
    plain = inner.retrieve(ORIGINAL, 3)

    inner.queries.clear()
    agent = DecomposingRetriever(agent_config(), inner, StubModel(["ATOMIQUE"]))
    assert [h.chunk_id for h in agent.retrieve(ORIGINAL, 3)] == [h.chunk_id for h in plain]
    assert inner.queries == [ORIGINAL]


def test_subquestions_are_each_retrieved_and_fused():
    inner = StubInner(
        {"Sous-question numéro un ?": ["9", "1"], "Sous-question numéro deux ?": ["8", "1"]},
        default=["1", "2", "3"],
    )
    model = StubModel(["Sous-question numéro un ?\nSous-question numéro deux ?"])
    agent = DecomposingRetriever(agent_config(), inner, model)

    results = agent.retrieve(ORIGINAL, 5)

    assert inner.queries == [ORIGINAL, "Sous-question numéro un ?", "Sous-question numéro deux ?"]
    assert agent.last_subquestions == ["Sous-question numéro un ?", "Sous-question numéro deux ?"]
    # Article 1 is ranked by all three queries, so RRF puts it first.
    assert results[0].chunk_id == "AUS-2010#art1"


def test_keep_original_false_drops_the_original_query():
    inner = StubInner({"Sous-question un ?": ["9"], "Sous-question deux ?": ["8"]}, default=["1"])
    model = StubModel(["Sous-question un ?\nSous-question deux ?"])
    agent = DecomposingRetriever(agent_config(keep_original=False), inner, model)

    agent.retrieve(ORIGINAL, 5)
    assert ORIGINAL not in inner.queries


def test_a_failed_decomposition_degrades_to_plain_retrieval():
    """A model failure must not take down a sixty-question evaluation."""
    inner = StubInner({}, default=["1", "2"])
    agent = DecomposingRetriever(agent_config(), inner, ExplodingModel())

    assert [h.chunk_id for h in agent.retrieve(ORIGINAL, 2)] == ["AUS-2010#art1", "AUS-2010#art2"]
    assert inner.queries == [ORIGINAL]


def test_decompose_disabled_does_not_call_the_model():
    inner = StubInner({}, default=["1"])
    model = StubModel([])
    agent = DecomposingRetriever(agent_config(decompose=False), inner, model)

    agent.retrieve(ORIGINAL, 1)
    assert model.calls == 0


def test_verification_stops_when_satisfied():
    inner = StubInner({}, default=["1", "2"])
    model = StubModel(["SUFFISANT"])
    agent = VerifyingRetriever(agent_config(verify=True), inner, model)

    agent.retrieve(ORIGINAL, 2)
    assert agent.last_followups == []
    assert inner.queries == [ORIGINAL]


def test_verification_retrieves_again_on_a_gap():
    inner = StubInner({"publicité foncière": ["7"]}, default=["1", "2"])
    model = StubModel(["publicité foncière"])
    agent = VerifyingRetriever(agent_config(verify=True, max_steps=2), inner, model)

    results = agent.retrieve(ORIGINAL, 5)
    assert agent.last_followups == ["publicité foncière"]
    assert "AUS-2010#art7" in [h.chunk_id for h in results]


def test_verification_respects_max_steps():
    """An always-dissatisfied model must still terminate."""
    inner = StubInner({}, default=["1", "2"])
    model = StubModel(["il manque ceci"] * 10)
    agent = VerifyingRetriever(agent_config(verify=True, max_steps=3), inner, model)

    agent.retrieve(ORIGINAL, 2)
    assert len(agent.last_followups) == 2  # max_steps counts attempts, not retries


# --- wiring ----------------------------------------------------------------


def test_pipeline_rejects_an_enabled_agent_without_a_generator():
    from kora.retrieval.pipeline import build_retriever

    with pytest.raises(ValueError, match="requires a generator"):
        build_retriever(agent_config(), index=None, articles=[], generator=None)


def test_describe_reports_agent_stages_in_pipeline_order():
    from kora.retrieval.pipeline import describe

    assert describe(agent_config(decompose=True, verify=True)) == (
        "dense + decompose + rerank + verify"
    )
