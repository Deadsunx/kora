# Serving

`kora serve start` · FastAPI + SSE · one RTX 4070 Laptop, 8 GiB · config
`07_rerank_fast` (`rerank-fast-54cefb1da669`)

The service exists to make the measured system usable, not to become a second
system. There is no serving-only prompt, no serving-only retrieval setting and
no serving-only model path: the app loads one `ExperimentConfig` and builds the
pipeline with the same `build_retriever` the evaluation harness calls. If those
could drift, the ablation table would stop describing anything a user touches.

## The endpoints

| route | what it does | cost |
|---|---|---|
| `GET /health` | which system is serving — run id, pipeline, index fingerprint | instant |
| `POST /search` | retrieval only, no generation | ~200 ms |
| `POST /ask` | complete answer, blocking | tens of seconds |
| `GET /ask/stream` | passages, then the answer token by token (SSE) | same total, far lower TTFT |
| `GET /` | the demo UI | |

Three decisions in that table are worth defending.

**`/health` reports identity, not liveness.** It returns the run id, the
pipeline description and the index fingerprint. The whole project rests on being
able to say which system produced a given output; a service that answers only
"ok" breaks that chain at the last step, where it matters most.

**Which config is served is an environment variable, not a request parameter.**
Letting a client pick the pipeline per request would mean the service has no
single identity and `/health` could no longer answer the question that matters
when an output looks wrong.

**`/search` is exposed separately** because retrieval is ~200 ms and generation
is tens of seconds. A client that only needs the articles should not pay for an
answer, and it makes the retrieval half inspectable from the UI — the "sources
seules" button.

## The API returns the contract, not just prose

Every field a client receives is one the evaluation harness already scores:

```json
{
  "answer": "…",
  "abstained": false,
  "citations": ["AUSCGIE-2014#art28"],
  "citations_unretrieved": [],
  "citations_repealed": [],
  "passages": [{ "citation": "article 28 AUSCGIE (2014)", "unsafe": false, "…": "…" }]
}
```

`citations_unretrieved` is the one that matters. An article cited but never
shown to the model came from parametric memory, which in a grounded system is
fabrication even when the article is real. Phase 4 measured that at 0.000 across
59 questions; the API reports it per response rather than leaving it to a log,
so the property is visible to a client instead of merely having been true during
an evaluation.

`unsafe` on a passage is the same argument one level down. Repealed articles are
indexed on purpose — a system that never sees repealed law cannot demonstrate
that it declines to rely on it — so the UI marks them rather than hiding them.

## Streaming, and what it actually buys

Phase 4 established that answer latency is essentially linear in output length:
21 words → 6.0 s, 101 words → 26.5 s, 286 words → 66.5 s. Streaming does not
change that. The same tokens are decoded in the same order and the total is
unchanged. What changes is when the first one is visible.

So the benchmark reports **time to first token and total time side by side**.
Reporting only the total would make streaming look like it did nothing;
reporting only TTFT would suggest an answer arrives in a fraction of a second.

The stream emits passages as a single event *before* generation starts, so the
UI renders the sources — the checkable part — while the answer is still
decoding. The `done` event carries the citation analysis, which cannot be
computed incrementally: a citation is only parseable once its closing year has
been decoded, and whether it is fabricated depends on the complete set.

## Generation is serialised, on purpose

One GPU holds one generator. The embedder, cross-encoder and a 4-bit 4B model
together are most of 8 GiB, and two concurrent `generate` calls contend for
memory that is not there. The failure mode is not a slow response; it is an OOM
that takes the process down.

A lock turns that into a queue: bounded, honest, and visible in the benchmark as
p90 latency rising with concurrency while throughput stays flat. That is a real
property of serving a 4B model on a laptop GPU, and the benchmark reports it
rather than hiding it behind a batch size this hardware cannot afford.

Retrieval is deliberately **not** serialised.

## Results

8 requests per wave, four questions twice · `runs/serving-rerank-fast-54cefb1da669/bench.json`

### Generation — `/ask/stream`

| concurrency | first token | total median | total p90 | answers/min | wall |
|---:|---:|---:|---:|---:|---:|
| 1 | **1.1 s** | 20.1 s | 37.9 s | 2.68 | 179 s |
| 2 | 13.2 s | 39.5 s | 68.1 s | 2.83 | 170 s |
| 4 | 40.8 s | 69.5 s | 86.8 s | 2.83 | 169 s |

### Retrieval — `/search`

| concurrency | median | p90 | req/s |
|---:|---:|---:|---:|
| 1 | 200 ms | 251 ms | 1.8 |
| 2 | 208 ms | 330 ms | 3.0 |
| 4 | 447 ms | 715 ms | 3.5 |

## Four findings

### 1. Streaming's headline number is a single-user number

At one client, the first token arrives in **1.1 s** against a 20.1 s complete
answer — **18.9× sooner**. That is the number a demo shows.

It does not survive concurrency:

| concurrency | first token | total | perceived speedup |
|---:|---:|---:|---:|
| 1 | 1.1 s | 20.1 s | **18.9×** |
| 2 | 13.2 s | 39.5 s | 3.0× |
| 4 | 40.8 s | 69.5 s | **1.7×** |

Under load, time-to-first-token is not prefill — it is **waiting for the lock**.
A request that queues behind three others cannot emit a token until they finish,
so TTFT converges on total latency and streaming stops buying anything.

This is the finding worth keeping from Phase 6. Streaming is a genuine and large
improvement to a single user's experience and close to worthless under
contention, and only measuring both revealed where the boundary sits. Quoting
18.9× without the other two rows would have been the serving equivalent of the
Phase 4 latency win.

### 2. The lock behaves exactly as a queue should

The prediction, written in `bench.py` before the run: *throughput stays flat
while p90 latency rises roughly linearly with concurrency.*

| concurrency | throughput | median latency | ratio to c=1 |
|---:|---:|---:|---:|
| 1 | 2.68 / min | 20.1 s | 1.0× |
| 2 | 2.83 / min | 39.5 s | 2.0× |
| 4 | 2.83 / min | 69.5 s | 3.5× |

Throughput is flat to within 6%. Latency rises 2.0× and 3.5× for 2× and 4×
concurrency. And the decisive number is the wall clock: **179 s, 170 s, 169 s**
for the same eight answers. Identical work, identical time, regardless of how
many clients asked at once — which is what a serialised resource means.

Nothing worse than queuing is happening. No OOM, no thrash, no superlinear
collapse. The lock converts a hardware limit into a predictable wait, which was
the entire justification for it.

### 3. Retrieval scales, which is what makes the lock attributable

`/search` throughput nearly doubles from c=1 to c=2 (1.8 → 3.0 req/s) before
saturating at 3.5. Generation's does not move at all.

That contrast is the control. Flat generation throughput on its own would be
consistent with "the GPU is simply saturated"; measuring an unlocked path on the
same server at the same time shows it is the lock, not the hardware.

Retrieval does slow down under load — 200 ms → 447 ms at c=4 — because the
cross-encoder is a GPU model too. It is contention, not queuing: requests
proceed, they just proceed more slowly.

### 4. The service adds ~7 ms

Retrieval at c=1 measures **200 ms** end-to-end over HTTP against the **193 ms**
Phase 3 recorded in-process. HTTP framing, JSON serialisation and SSE parsing
cost about seven milliseconds combined.

Worth stating because it closes a gap in the ablation table: those numbers were
measured in a Python loop, and this confirms they survive contact with a
network client rather than being a laboratory artefact.

A related number: inside a streamed request, passages reach the client at
252 ms median at c=1, rising to 636 ms at c=4 — the same cross-encoder
contention as above, seen from inside the generation path.

### A reproducibility check that passed

Median answer length is **416 characters at every concurrency level**. Same
questions, greedy decoding, so identical answers regardless of load. If that
number had moved, something in the serving path would have been altering
generation — the exact drift the shared `_prepare` exists to prevent.

## Docker

The image is `python:3.13-slim` rather than an `nvidia/cuda` base, because the
PyTorch cu128 wheels bundle the CUDA runtime libraries they need; a CUDA base
layer would ship a second copy. The host still needs an NVIDIA driver and the
container toolkit, and the container needs `--gpus all`.

Model weights are **not** baked in. They are ~4 GB of Hugging Face cache that
changes independently of this code, so they live in a named volume and a cold
container populates it once. `data/` and `models/` mount read-only: the service
never writes to them, and read-only means a serving bug cannot corrupt a
measured index.

Nothing in the compose file builds an index. `data/indexes/` must already
contain the fingerprint the config asks for. Building an index inside a
short-lived container and losing it on exit is a worse default than failing
loudly with the command to run.

```bash
docker compose up --build
```

## Limits

- **Single process, single GPU.** Horizontal scaling would need one process per
  GPU behind a load balancer; nothing here does that, and pretending otherwise
  with a worker count would just multiply the OOM.
- **2.8 answers per minute is the ceiling on this hardware**, at any
  concurrency. The service is a demonstrator, not a system that serves a
  classroom. Raising it means a smaller generator, a batching server such as
  vLLM, or more than one GPU — none of which this project measured.
- **No queue admission control.** A fourth concurrent client waits 40 s for its
  first token with no indication it is queued. A production service would return
  a position or shed load; this one just waits.
- **No authentication, no rate limiting.** It binds to localhost by default and
  is a demo, not a deployment.
- **The benchmark is n=8 requests per wave.** Enough to separate queuing from
  contention; not enough for a percentile claim finer than p90.
- **Answer quality is not measured here.** That is what the gold set and the
  evaluation harness are for; the serving tests check the contract and the
  transport, deliberately not the French.
