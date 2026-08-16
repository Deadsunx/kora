"""The API contract.

Deliberately shaped like the evaluation record rather than like a chat API. Every
field a client receives is one the harness already scores: the passages that were
shown, whether the model abstained, which articles it cited, and whether any of
them are repealed. A response a user can read is therefore a response the
ablation table can explain.

The one addition is `unsafe` on a passage. The evaluation stores repealed hits as
a list of chunk ids; a UI needs the flag attached to the passage it is about to
render, so it can mark it before a reader treats it as current law.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from kora.documents import Article
from kora.generation.generator import GeneratedAnswer
from kora.retrieval.dense import Hit


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AskRequest(_Base):
    """One question."""

    question: str = Field(..., min_length=1, max_length=1000)
    # Overridable per request so the effect of the context window is visible
    # from the UI rather than only from a config file. Bounded because the
    # generator's context is finite and a large k silently truncates passages.
    final_k: int | None = Field(None, ge=1, le=20)


class Passage(_Base):
    """One retrieved article, as shown to the model."""

    chunk_id: str
    citation: str
    document_id: str
    number: str
    rubric: str
    text: str
    score: float
    # Repealed, or belonging to a superseded act. Rendered as a warning rather
    # than filtered out: a system that never sees repealed law cannot
    # demonstrate that it declines to rely on it.
    unsafe: bool

    @classmethod
    def from_hit(cls, hit: Hit) -> Passage:
        return cls.from_article(hit.article, score=hit.score)

    @classmethod
    def from_article(cls, article: Article, *, score: float) -> Passage:
        return cls(
            chunk_id=article.chunk_id,
            citation=article.citation,
            document_id=article.document_id,
            number=article.number,
            rubric=article.rubric,
            text=article.text,
            score=score,
            unsafe=article.repealed or article.status == "superseded",
        )


class AnswerResponse(_Base):
    """A complete answer, with everything needed to check it."""

    question: str
    answer: str
    abstained: bool
    citations: list[str]
    # Cited articles that were not among the passages shown. In a grounded
    # system this is fabrication even when the article is real, so it is
    # reported to the client rather than left for a log.
    citations_unretrieved: list[str]
    citations_repealed: list[str]
    passages: list[Passage]

    retrieval_ms: float
    generation_ms: float
    prompt_tokens: int
    generated_tokens: int

    @classmethod
    def build(
        cls,
        *,
        question: str,
        generated: GeneratedAnswer,
        hits: list[Hit],
        retrieval_ms: float,
        generation_ms: float,
    ) -> AnswerResponse:
        shown = {h.chunk_id: h for h in hits}
        cited = list(generated.cited_chunk_ids)
        return cls(
            question=question,
            answer=generated.text,
            abstained=generated.abstained,
            citations=cited,
            citations_unretrieved=[c for c in cited if c not in shown],
            citations_repealed=[
                c for c in cited if c in shown and shown[c].is_unsafe_to_cite
            ],
            passages=[Passage.from_hit(h) for h in hits],
            retrieval_ms=round(retrieval_ms, 2),
            generation_ms=round(generation_ms, 2),
            prompt_tokens=generated.prompt_tokens,
            generated_tokens=generated.generated_tokens,
        )


class SearchResponse(_Base):
    """Retrieval without generation.

    Worth exposing on its own: retrieval is ~200 ms and generation is ~26 s, so
    a client that only needs the articles should not pay for an answer. It also
    makes the retrieval half of the system inspectable from the UI.
    """

    question: str
    passages: list[Passage]
    retrieval_ms: float


class HealthResponse(_Base):
    """What is actually loaded and serving.

    Reports the config fingerprint, not just "ok". The whole project rests on
    knowing which system produced a given output, and a service that cannot say
    which config it is running breaks that chain at the last step.
    """

    status: str
    run_id: str
    config_name: str
    pipeline: str
    index_fingerprint: str
    corpus_size: int
    generator_model: str
    adapter: str | None
    generator_loaded: bool
