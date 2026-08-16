"""Benchmark the running service over HTTP.

Against a live server rather than the engine in-process, because the question is
what a client experiences: HTTP framing, SSE parsing and the request queue are
part of the latency whether or not they are interesting.

Three things are measured, and each corresponds to a claim that would otherwise
be an assumption.

**Time to first token against total time.** Streaming does not make generation
faster -- the same tokens are decoded either way. It changes when the first one
is visible. Reporting only total latency would make streaming look like it did
nothing; reporting only TTFT would suggest the answer arrives in 300 ms. Both,
side by side, are the honest form.

**Throughput and latency under concurrency.** Generation is serialised behind a
lock (see `engine.py`), so concurrent askers queue. The prediction is that
throughput stays flat while p90 latency rises roughly linearly with concurrency.
If throughput rose, the lock would be unnecessary; if latency rose faster than
linearly, something worse than queuing is happening. The benchmark exists to
tell those apart.

**Retrieval under the same concurrency.** `/search` is not locked, so it should
scale where `/ask` does not. Measuring both against the same server is what
makes the lock's cost attributable to the lock.
"""

from __future__ import annotations

import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

import httpx

from kora.logging import get_logger

log = get_logger(__name__)

# Four questions spanning the kinds the gold set distinguishes, including one
# unanswerable. A benchmark over lookups alone would understate latency, since
# abstentions are short and lookups are the shortest real answers.
DEFAULT_QUESTIONS = (
    "Quelle est la durée maximale de vie d'une société commerciale ?",
    "Qu'est-ce qu'une hypothèque ?",
    "Quel est le capital social minimum d'une société anonyme ?",
    "Quelle est la peine encourue pour abus de biens sociaux ?",
)


@dataclass(slots=True)
class Sample:
    """One request's timings."""

    question: str
    ok: bool
    total_ms: float
    first_token_ms: float | None = None
    retrieval_ms: float | None = None
    passages_ms: float | None = None
    generated_chars: int = 0
    error: str = ""


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * fraction), len(ordered) - 1)]


def _summarise(samples: list[Sample]) -> dict[str, Any]:
    ok = [s for s in samples if s.ok]
    if not ok:
        return {"n": 0, "failures": len(samples)}

    totals = [s.total_ms for s in ok]
    summary: dict[str, Any] = {
        "n": len(ok),
        "failures": len(samples) - len(ok),
        "total_ms_median": round(statistics.median(totals), 1),
        "total_ms_p90": round(_percentile(totals, 0.9), 1),
    }

    firsts = [s.first_token_ms for s in ok if s.first_token_ms is not None]
    if firsts:
        summary |= {
            "first_token_ms_median": round(statistics.median(firsts), 1),
            "first_token_ms_p90": round(_percentile(firsts, 0.9), 1),
            # The fraction of the wait that streaming removes from the user's
            # perception. This is the only number that justifies the UI.
            "perceived_speedup": round(statistics.median(totals) / statistics.median(firsts), 1),
        }

    passages = [s.passages_ms for s in ok if s.passages_ms is not None]
    if passages:
        summary["passages_ms_median"] = round(statistics.median(passages), 1)

    chars = [s.generated_chars for s in ok if s.generated_chars]
    if chars:
        summary["answer_chars_median"] = int(statistics.median(chars))
    return summary


def _stream_once(client: httpx.Client, base_url: str, question: str) -> Sample:
    """One streamed answer, timing the first token separately."""
    started = time.perf_counter()
    first_token_ms: float | None = None
    passages_ms: float | None = None
    retrieval_ms: float | None = None
    chars = 0
    event = ""

    try:
        with client.stream(
            "GET", f"{base_url}/ask/stream", params={"question": question}
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("event:"):
                    event = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    payload = json.loads(line.split(":", 1)[1].strip())
                    if event == "passages":
                        passages_ms = (time.perf_counter() - started) * 1000
                        retrieval_ms = payload.get("retrieval_ms")
                    elif event == "token":
                        if first_token_ms is None:
                            first_token_ms = (time.perf_counter() - started) * 1000
                        chars += len(payload.get("text", ""))
                    elif event == "done":
                        chars = len(payload.get("answer", "")) or chars
    except Exception as exc:
        return Sample(question, False, (time.perf_counter() - started) * 1000, error=str(exc))

    return Sample(
        question=question,
        ok=True,
        total_ms=(time.perf_counter() - started) * 1000,
        first_token_ms=first_token_ms,
        retrieval_ms=retrieval_ms,
        passages_ms=passages_ms,
        generated_chars=chars,
    )


def _search_once(client: httpx.Client, base_url: str, question: str) -> Sample:
    """One retrieval-only request."""
    started = time.perf_counter()
    try:
        response = client.post(f"{base_url}/search", json={"question": question})
        response.raise_for_status()
        retrieval_ms = response.json().get("retrieval_ms")
    except Exception as exc:
        return Sample(question, False, (time.perf_counter() - started) * 1000, error=str(exc))

    return Sample(
        question=question,
        ok=True,
        total_ms=(time.perf_counter() - started) * 1000,
        retrieval_ms=retrieval_ms,
    )


def _run_wave(
    base_url: str,
    questions: list[str],
    concurrency: int,
    timeout: float,
    streaming: bool,
) -> tuple[list[Sample], float]:
    """Issue every question with a fixed number of workers in flight."""
    call = _stream_once if streaming else _search_once

    def one(question: str) -> Sample:
        with httpx.Client(timeout=timeout) as client:
            return call(client, base_url, question)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        samples = list(pool.map(one, questions))
    return samples, time.perf_counter() - started


def run_benchmark(
    base_url: str,
    *,
    questions: tuple[str, ...] = DEFAULT_QUESTIONS,
    repeats: int = 2,
    concurrencies: tuple[int, ...] = (1, 2, 4),
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Benchmark `/ask/stream` and `/search` at several concurrency levels."""
    base_url = base_url.rstrip("/")

    with httpx.Client(timeout=30.0) as client:
        health = client.get(f"{base_url}/health").json()
    log.info("benchmarking", run_id=health.get("run_id"), pipeline=health.get("pipeline"))

    workload = list(questions) * repeats
    report: dict[str, Any] = {
        "base_url": base_url,
        "health": health,
        "requests_per_wave": len(workload),
        "generation": {},
        "retrieval": {},
    }

    # One untimed streamed answer first. Even with the generator warmed at
    # startup, the first request through a fresh connection pays for TLS-less
    # socket setup and the first CUDA graph, and folding that into c=1 would
    # inflate the baseline every other level is compared against.
    with httpx.Client(timeout=timeout) as client:
        _stream_once(client, base_url, questions[0])

    for concurrency in concurrencies:
        samples, wall = _run_wave(base_url, workload, concurrency, timeout, streaming=True)
        summary = _summarise(samples)
        summary["wall_seconds"] = round(wall, 1)
        summary["throughput_per_min"] = round(60 * summary.get("n", 0) / wall, 2) if wall else 0.0
        report["generation"][f"c{concurrency}"] = summary
        log.info(
            "generation wave",
            concurrency=concurrency,
            median_ms=summary.get("total_ms_median"),
            throughput=summary.get("throughput_per_min"),
        )

    for concurrency in concurrencies:
        samples, wall = _run_wave(base_url, workload, concurrency, timeout, streaming=False)
        summary = _summarise(samples)
        summary["wall_seconds"] = round(wall, 2)
        summary["throughput_per_sec"] = round(summary.get("n", 0) / wall, 1) if wall else 0.0
        report["retrieval"][f"c{concurrency}"] = summary

    report["samples"] = [asdict(s) for s in samples]
    return report
