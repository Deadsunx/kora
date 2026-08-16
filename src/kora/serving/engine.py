"""The loaded system, shared by every request.

Two constraints shape this file, and both come from measurements taken earlier
in the project rather than from taste.

**Everything loads once.** The embedder, cross-encoder and generator together
are most of 8 GiB. Loading them per request is not slow, it is impossible. They
are constructed at startup and held for the process's lifetime.

**Generation is serialised.** One GPU holds one generator, and Phase 4 measured
a median answer at 15-26 s. Two concurrent `generate` calls on the same 8 GiB
card contend for memory that is not there, and the failure mode is an OOM that
takes down the process rather than a slow response. A lock turns that into a
queue: honest, bounded, and visible in the benchmark as p90 latency rising with
concurrency while throughput stays flat. That is a real property of serving a
4B model on a laptop GPU, and the benchmark reports it rather than hiding it
behind a batch size nobody can afford.

Retrieval is *not* serialised. It is ~200 ms, read-only against a FAISS index,
and the encoders are thread-safe for inference, so search requests proceed
while an answer is being generated.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from threading import Lock

from kora.config import ExperimentConfig
from kora.documents import Article
from kora.generation.generator import GeneratedAnswer, Generator
from kora.logging import get_logger
from kora.retrieval.dense import Hit
from kora.retrieval.index import index_exists, load_index
from kora.retrieval.pipeline import build_retriever, describe
from kora.serving.schemas import AnswerResponse, HealthResponse, Passage, SearchResponse

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class StreamChunk:
    """One event in a streamed answer.

    `kind` is the SSE event name. Passages arrive first, as a single event, so
    the UI can render the sources while the answer is still decoding -- which is
    the whole point of streaming a system whose retrieval is 200 ms and whose
    generation is twenty seconds.
    """

    kind: str
    data: dict


class Engine:
    """Holds the loaded pipeline and answers questions with it."""

    def __init__(self, config: ExperimentConfig, *, preload_generator: bool = True) -> None:
        if not index_exists(config):
            raise FileNotFoundError(
                f"No index for fingerprint {config.index_fingerprint()}. "
                f"Build one first: kora index build --config <config>"
            )

        self.config = config
        self._index, self._articles = load_index(config)
        self._retriever = build_retriever(config, self._index, self._articles)
        self._generator = Generator(config)
        self._generator_loaded = False
        # Serialises generation only. See the module docstring.
        self._generation_lock = Lock()

        log.info(
            "engine ready",
            run_id=config.run_id,
            pipeline=describe(config),
            corpus_size=len(self._articles),
        )

        if preload_generator:
            self.warm()

    # -- lifecycle ---------------------------------------------------------

    def warm(self) -> None:
        """Load the generator and run one short generation.

        Called at startup so the first real user does not pay for several
        gigabytes of weights. The evaluation runner does the same thing for the
        same reason -- a smoke test once reported a 782-second p90 that was
        entirely the model download happening inside a timed block.
        """
        started = time.perf_counter()
        self._generator.answer("Test.", self._articles[:1])
        self._generator_loaded = True
        log.info("generator warm", seconds=round(time.perf_counter() - started, 1))

    # -- retrieval ---------------------------------------------------------

    def _retrieve(self, question: str, final_k: int | None) -> tuple[list[Hit], float]:
        k = final_k or self.config.retrieval.final_k
        started = time.perf_counter()
        hits = self._retriever.retrieve(question, self.config.retrieval.top_k)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return hits[:k], elapsed_ms

    def search(self, question: str, final_k: int | None = None) -> SearchResponse:
        """Retrieve passages without generating an answer."""
        hits, elapsed_ms = self._retrieve(question, final_k)
        return SearchResponse(
            question=question,
            passages=[Passage.from_hit(h) for h in hits],
            retrieval_ms=round(elapsed_ms, 2),
        )

    # -- generation --------------------------------------------------------

    def answer(self, question: str, final_k: int | None = None) -> AnswerResponse:
        """Retrieve and generate one complete answer."""
        hits, retrieval_ms = self._retrieve(question, final_k)
        articles: list[Article] = [h.article for h in hits]

        started = time.perf_counter()
        with self._generation_lock:
            generated: GeneratedAnswer = self._generator.answer(question, articles)
        generation_ms = (time.perf_counter() - started) * 1000

        return AnswerResponse.build(
            question=question,
            generated=generated,
            hits=hits,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
        )

    def stream(self, question: str, final_k: int | None = None) -> Iterator[StreamChunk]:
        """Retrieve, emit the passages, then stream the answer.

        The final `done` event carries the same citation analysis the batch
        endpoint returns. It cannot be computed incrementally: a citation is only
        parseable once its closing year has been decoded, and whether it is
        fabricated depends on the complete set. Clients render the text as it
        arrives and the verdict at the end.
        """
        hits, retrieval_ms = self._retrieve(question, final_k)
        articles: list[Article] = [h.article for h in hits]

        yield StreamChunk(
            "passages",
            {
                "passages": [Passage.from_hit(h).model_dump() for h in hits],
                "retrieval_ms": round(retrieval_ms, 2),
            },
        )

        from kora.generation.prompt import extract_citations, is_abstention

        started = time.perf_counter()
        first_token_ms: float | None = None
        parts: list[str] = []

        with self._generation_lock:
            for fragment in self._generator.stream(question, articles):
                if not fragment:
                    continue
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - started) * 1000
                parts.append(fragment)
                yield StreamChunk("token", {"text": fragment})

        text = "".join(parts).strip()
        generation_ms = (time.perf_counter() - started) * 1000

        shown = {h.chunk_id: h for h in hits}
        cited = [c.chunk_id for c in extract_citations(text)]
        yield StreamChunk(
            "done",
            {
                "answer": text,
                "abstained": is_abstention(text),
                "citations": cited,
                "citations_unretrieved": [c for c in cited if c not in shown],
                "citations_repealed": [
                    c for c in cited if c in shown and shown[c].is_unsafe_to_cite
                ],
                "retrieval_ms": round(retrieval_ms, 2),
                "generation_ms": round(generation_ms, 2),
                # Time to first token, the number streaming actually improves.
                # Total generation time is unchanged by streaming, so reporting
                # only the total would make streaming look like it did nothing.
                "first_token_ms": round(first_token_ms or generation_ms, 2),
            },
        )

    # -- introspection -----------------------------------------------------

    def health(self) -> HealthResponse:
        return HealthResponse(
            status="ok",
            run_id=self.config.run_id,
            config_name=self.config.name,
            pipeline=describe(self.config),
            index_fingerprint=self.config.index_fingerprint(),
            corpus_size=len(self._articles),
            generator_model=self.config.generator.model_name,
            adapter=self.config.generator.adapter_path,
            generator_loaded=self._generator_loaded,
        )
