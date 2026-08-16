"""FastAPI application.

The service is deliberately thin: it loads one `ExperimentConfig`, builds the
same pipeline the evaluation harness builds, and exposes it. There is no
serving-only code path, no separate prompt, no "production" retrieval setting.
If the served system could drift from the measured one, the ablation table would
stop describing anything a user actually touches.

Which config is served is an environment variable rather than a request
parameter. Letting a client choose the pipeline per request would mean the
service has no single identity, and `/health` could no longer answer the one
question that matters when an output looks wrong: *which system produced this?*
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool

from kora.config import load_config
from kora.logging import get_logger
from kora.paths import CONFIG_DIR
from kora.serving.engine import Engine
from kora.serving.schemas import AnswerResponse, AskRequest, HealthResponse, SearchResponse

log = get_logger(__name__)

# The frozen best system from Phase 3: dense + reranking at fp16, 512 tokens.
# Not the adapter -- Phase 4 measured it as a regression and it is not shipped.
DEFAULT_CONFIG = CONFIG_DIR / "experiments" / "07_rerank_fast.yaml"

STATIC_DIR = Path(__file__).parent / "static"

_engine: Engine | None = None


def get_engine() -> Engine:
    """The loaded engine, or a 503 if startup has not finished."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine is still loading.")
    return _engine


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the pipeline once, before the first request."""
    global _engine

    config_path = Path(os.environ.get("KORA_CONFIG", DEFAULT_CONFIG))
    # Off by default only in tests: warming downloads and loads several GiB, and
    # a test that merely checks routing should not need a GPU.
    preload = os.environ.get("KORA_PRELOAD", "1") != "0"

    log.info("starting service", config=str(config_path), preload=preload)
    _engine = Engine(load_config(config_path), preload_generator=preload)
    try:
        yield
    finally:
        _engine = None


app = FastAPI(
    title="Kora",
    description="Retrieval-augmented question answering over the Actes uniformes OHADA.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Which system is serving, not merely whether it is up."""
    return get_engine().health()


@app.post("/search", response_model=SearchResponse)
def search(request: AskRequest) -> SearchResponse:
    """Retrieve passages without generating an answer (~200 ms)."""
    return get_engine().search(request.question, request.final_k)


@app.post("/ask", response_model=AnswerResponse)
def ask(request: AskRequest) -> AnswerResponse:
    """Retrieve and generate one complete answer.

    Blocking, and slow: Phase 4 measured a 26.5 s median. Prefer `/ask/stream`
    for anything a person is waiting on.
    """
    return get_engine().answer(request.question, request.final_k)


@app.get("/ask/stream")
async def ask_stream(
    question: str = Query(..., min_length=1, max_length=1000),
    final_k: int | None = Query(None, ge=1, le=20),
) -> EventSourceResponse:
    """Stream one answer as server-sent events.

    A GET rather than a POST because `EventSource` in the browser cannot issue a
    POST, and the alternative -- hand-rolling a fetch-based reader -- buys
    nothing here.

    Generation is synchronous and blocking, so it is iterated in a worker thread
    and the event loop stays free to serve `/search` and `/health` meanwhile.
    """
    engine = get_engine()

    async def events() -> AsyncIterator[dict]:
        async for chunk in iterate_in_threadpool(engine.stream(question, final_k)):
            yield {"event": chunk.kind, "data": json.dumps(chunk.data, ensure_ascii=False)}

    return EventSourceResponse(events())


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """The demo UI."""
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))
