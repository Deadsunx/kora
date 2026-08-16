"""Serving tests.

Everything here runs without a GPU, a model or an index. The engine is replaced
by a stub, because what needs testing at this layer is the contract -- what a
client receives, and whether the safety signals survive the trip to JSON -- not
whether the transformer produces good French. Answer quality is measured by the
evaluation harness, against a gold set, and duplicating that here would produce
a test that is slow, flaky and weaker than the thing it duplicates.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from kora.documents import Article
from kora.generation.generator import GeneratedAnswer
from kora.generation.prompt import Citation
from kora.retrieval.dense import Hit
from kora.serving import app as app_module
from kora.serving.engine import StreamChunk
from kora.serving.schemas import AnswerResponse, AskRequest, Passage


def make_article(number: str = "28", *, repealed: bool = False, status: str = "in_force") -> Article:
    return Article(
        document_id="AUSCGIE-2014",
        abbrev="AUSCGIE",
        status=status,  # type: ignore[arg-type]
        number=number,
        rubric="Durée de la société",
        text="La durée de la société ne peut excéder quatre-vingt-dix-neuf ans.",
        page_start=1,
        page_end=1,
        repealed=repealed,
    )


def make_hit(article: Article, score: float = 0.9) -> Hit:
    return Hit(chunk_id=article.chunk_id, score=score, article=article)


# --- schemas ---------------------------------------------------------------


def test_passage_marks_repealed_articles_unsafe():
    assert Passage.from_article(make_article(repealed=True), score=0.5).unsafe is True
    assert Passage.from_article(make_article(), score=0.5).unsafe is False


def test_passage_marks_superseded_acts_unsafe():
    """Status is a property of both the act and the article; either one is a warning."""
    article = make_article(status="superseded")
    assert Passage.from_article(article, score=0.5).unsafe is True


def test_answer_response_separates_unretrieved_citations():
    """A cited article that was never shown is fabrication, even if it is real."""
    shown = make_article("28")
    generated = GeneratedAnswer(
        text="La durée est de 99 ans (article 28 AUSCGIE (2014)) et article 999 AUSCGIE (2014).",
        citations=(Citation("28", "AUSCGIE", "2014"), Citation("999", "AUSCGIE", "2014")),
        abstained=False,
        prompt_tokens=100,
        generated_tokens=20,
    )
    response = AnswerResponse.build(
        question="Durée ?",
        generated=generated,
        hits=[make_hit(shown)],
        retrieval_ms=190.0,
        generation_ms=15000.0,
    )
    assert response.citations_unretrieved == ["AUSCGIE-2014#art999"]
    assert response.citations_repealed == []
    assert response.citations == ["AUSCGIE-2014#art28", "AUSCGIE-2014#art999"]


def test_answer_response_flags_repealed_citation():
    repealed = make_article("12", repealed=True)
    generated = GeneratedAnswer(
        text="Voir article 12 AUSCGIE (2014).",
        citations=(Citation("12", "AUSCGIE", "2014"),),
        abstained=False,
        prompt_tokens=100,
        generated_tokens=10,
    )
    response = AnswerResponse.build(
        question="?",
        generated=generated,
        hits=[make_hit(repealed)],
        retrieval_ms=1.0,
        generation_ms=1.0,
    )
    assert response.citations_repealed == ["AUSCGIE-2014#art12"]
    assert response.citations_unretrieved == []


def test_ask_request_rejects_unknown_fields():
    """Same discipline as the experiment configs: a typo must not be ignored."""
    with pytest.raises(ValueError):
        AskRequest(question="Durée ?", top_k=10)  # type: ignore[call-arg]


def test_ask_request_rejects_empty_question():
    with pytest.raises(ValueError):
        AskRequest(question="")


# --- routes ----------------------------------------------------------------


class StubEngine:
    """Stands in for a loaded pipeline."""

    def __init__(self) -> None:
        self.article = make_article()
        self.calls: list[tuple[str, int | None]] = []

    def health(self):
        from kora.serving.schemas import HealthResponse

        return HealthResponse(
            status="ok",
            run_id="rerank-fast-54cefb1da669",
            config_name="rerank-fast",
            pipeline="dense + rerank",
            index_fingerprint="32e88149561b",
            corpus_size=3056,
            generator_model="Qwen/Qwen3-4B-Instruct-2507",
            adapter=None,
            generator_loaded=True,
        )

    def search(self, question, final_k=None):
        from kora.serving.schemas import SearchResponse

        self.calls.append((question, final_k))
        return SearchResponse(
            question=question,
            passages=[Passage.from_article(self.article, score=0.91)],
            retrieval_ms=193.0,
        )

    def answer(self, question, final_k=None):
        self.calls.append((question, final_k))
        generated = GeneratedAnswer(
            text="99 ans (article 28 AUSCGIE (2014)).",
            citations=(Citation("28", "AUSCGIE", "2014"),),
            abstained=False,
            prompt_tokens=100,
            generated_tokens=12,
        )
        return AnswerResponse.build(
            question=question,
            generated=generated,
            hits=[make_hit(self.article)],
            retrieval_ms=193.0,
            generation_ms=15500.0,
        )

    def stream(self, question, final_k=None):
        self.calls.append((question, final_k))
        yield StreamChunk(
            "passages",
            {"passages": [Passage.from_article(self.article, score=0.91).model_dump()],
             "retrieval_ms": 193.0},
        )
        yield StreamChunk("token", {"text": "99 ans "})
        yield StreamChunk("token", {"text": "(article 28 AUSCGIE (2014))."})
        yield StreamChunk(
            "done",
            {
                "answer": "99 ans (article 28 AUSCGIE (2014)).",
                "abstained": False,
                "citations": ["AUSCGIE-2014#art28"],
                "citations_unretrieved": [],
                "citations_repealed": [],
                "retrieval_ms": 193.0,
                "generation_ms": 15500.0,
                "first_token_ms": 310.0,
            },
        )


@pytest.fixture
def client(monkeypatch):
    """A client over a stubbed engine.

    Constructed without the context manager on purpose: entering it would run
    the lifespan, which loads several gigabytes of weights.
    """
    engine = StubEngine()
    monkeypatch.setattr(app_module, "_engine", engine)
    test_client = TestClient(app_module.app)
    test_client.engine = engine  # type: ignore[attr-defined]
    return test_client


def test_health_reports_which_system_is_serving(client):
    body = client.get("/health").json()
    assert body["run_id"] == "rerank-fast-54cefb1da669"
    assert body["pipeline"] == "dense + rerank"
    assert body["corpus_size"] == 3056


def test_search_returns_passages_without_an_answer(client):
    body = client.post("/search", json={"question": "Durée ?"}).json()
    assert "answer" not in body
    assert body["passages"][0]["citation"] == "article 28 AUSCGIE (2014)"


def test_ask_returns_the_full_contract(client):
    body = client.post("/ask", json={"question": "Durée ?"}).json()
    assert body["citations"] == ["AUSCGIE-2014#art28"]
    assert body["citations_unretrieved"] == []
    assert body["abstained"] is False
    assert body["passages"][0]["unsafe"] is False


def test_ask_rejects_unknown_fields(client):
    assert client.post("/ask", json={"question": "?", "model": "gpt-4"}).status_code == 422


def test_ask_rejects_out_of_range_final_k(client):
    assert client.post("/ask", json={"question": "?", "final_k": 99}).status_code == 422


def test_final_k_reaches_the_engine(client):
    client.post("/ask", json={"question": "Durée ?", "final_k": 3})
    assert client.engine.calls[-1] == ("Durée ?", 3)


def test_stream_emits_passages_then_tokens_then_done(client):
    with client.stream("GET", "/ask/stream", params={"question": "Durée ?"}) as response:
        assert response.status_code == 200
        events, data = [], []
        for line in response.iter_lines():
            if line.startswith("event:"):
                events.append(line.split(":", 1)[1].strip())
            elif line.startswith("data:"):
                data.append(json.loads(line.split(":", 1)[1].strip()))

    assert events[0] == "passages"
    assert events[-1] == "done"
    assert events.count("token") == 2
    # Passages must arrive before any token: the UI renders sources while the
    # answer is still decoding, which is the reason for streaming at all.
    assert events.index("passages") < events.index("token")
    assert data[-1]["citations"] == ["AUSCGIE-2014#art28"]
    assert data[-1]["first_token_ms"] < data[-1]["generation_ms"]


def test_requests_before_startup_return_503(monkeypatch):
    monkeypatch.setattr(app_module, "_engine", None)
    assert TestClient(app_module.app).get("/health").status_code == 503


def test_ui_is_served_at_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "/ask/stream" in response.text
