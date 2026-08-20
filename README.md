# Kora

**Retrieval-augmented question answering over French-language African business law (OHADA).**

> Status: **all eight phases complete.** Every number below links to the
> write-up that produced it.

**→ [Read the technical report](https://deadsunx.github.io/kora/)** — the whole
project in one document, organised around how each finding was found.
([markdown version](docs/report.md))

---

## Why this project

Most retrieval systems are built and evaluated in English. The techniques that
work there are assumed to transfer, and usually nobody checks. This project
checks.

The corpus is the **Actes uniformes OHADA** — the body of business law shared by
17 mostly francophone African states. It is a good testbed for three reasons:

- **It is French, and legal French.** Dense retrievers trained mainly on English
  web text tend to blur exactly the tokens that matter here: article numbers,
  fixed terms of art like *société anonyme* or *sûreté mobilière*, cross
  references between texts.
- **It is highly structured.** Livre → Titre → Chapitre → Article. That
  structure is a gift for chunking and a fair test of whether structure-aware
  chunking beats naive splitting.
- **Answers must be attributable.** "Article 640 AUSCGIE" is checkable. A legal
  assistant that cannot cite, or that will not say *I don't know*, is worse than
  useless — so citation accuracy and abstention are first-class metrics here,
  not afterthoughts.

## The actual thesis

The deliverable is not a chatbot. The deliverable is an **ablation table**: a
defensible account of which components earn their cost, in accuracy and in
latency, on this corpus. A demo that answers questions is the by-product.

This ordering is deliberate. The evaluation harness (Phase 2) was built *before*
any retrieval improvement (Phase 3), because an improvement you cannot measure
is a preference, not a result.

## What exists

**3,056 articles** parsed from source PDFs across 10 acts · **64 human-validated
questions**, 56 answerable · **23 recorded runs** · 252 tests · single RTX 4070
Laptop, 8 GiB.

| | |
|---|---|
| Corpus | AUSCGIE 1089 · AUPSRVE 448 · AUSCOOP 397 · AUPCAP 377 · AUDCG 307 · AUS 228 · AUDCIF 123 · AUA 38 · AUCTMR 31 · AUM 18 |
| Retrieval | `multilingual-e5-base` dense + `bge-reranker-v2-m3` cross-encoder |
| Generation | Qwen3-4B-Instruct-2507, 4-bit NF4 |
| Fine-tuning | QLoRA r=16, paged 8-bit AdamW, gradient checkpointing |
| Serving | FastAPI + SSE, streaming UI, Docker |

## Results

### Retrieval — [`docs/ablations.md`](docs/ablations.md)

| system | recall@5 | MRR | nDCG@5 | latency (median) |
|---|---:|---:|---:|---:|
| BM25 only | 0.637 | 0.576 | 0.551 | **5 ms** |
| dense + BM25 (RRF) | 0.708 | 0.625 | 0.611 | 26 ms |
| dense (frozen baseline) | 0.735 | 0.652 | 0.641 | 17 ms |
| dense + BM25 + rerank, pool 50 | 0.815 | **0.800** | 0.763 | 461 ms |
| dense + BM25 + rerank, pool 20 | **0.824** | 0.786 | 0.760 | 235 ms |
| **dense + rerank, fp16, 512 tokens** | 0.815 | 0.796 | **0.765** | **205 ms** |

**+8.0 points of recall@5 over the baseline (+11% relative), at 205 ms.**
Reranking is the only component that earned its cost.

These were re-measured after `cross_act` was rebuilt (below), which grew the
gold set from 59 questions to 64 and **changed one conclusion**.

### Generation — [`docs/generation-baseline.md`](docs/generation-baseline.md)

| metric | value | |
|---|---:|---|
| answers_citing_unretrieved | **0.000** | nothing invented from parametric memory |
| answers_citing_repealed | **0.000** | no repealed article presented as law |
| answers_with_no_citation | **0.000** | every answer cited something |
| correct_abstention | 0.875 | 7 of 8 unanswerable questions declined |
| citation_recall | 0.765 | bounded by retrieval, not by the model |

Only **3 of 51** answers left a gold article uncited when it was actually in
context. The remaining gap is retrieval never surfacing the article.

### Fine-tuning — [`docs/fine-tuning-results.md`](docs/fine-tuning-results.md)

**The adapter is a regression and is not shipped.** Latency fell 41%, answers
got 67% shorter, abstention hit 8/8 — and citation recall fell 11.9 points while
the answers stopped answering the question.

| | base | adapter |
|---|---:|---:|
| citation_recall | **0.765** | 0.646 |
| answer length (median) | 101 w | 33 w |
| latency (median) | 26.5 s | 15.5 s |
| similarity to cited article's opening | 0.222 | **0.896** |

That last row is the finding. The training targets were built as
`_first_sentence(article.text) + citation`, so the adapter learned to echo an
article's opening regardless of the question — **38 of 47** answers, at 98.4%
token accuracy and 0.027 loss. Every structural metric improved or held while
quality collapsed. A dashboard would have called it a successful fine-tune.

### Agentic layer — [`docs/agentic.md`](docs/agentic.md)

**Ship neither.** Decomposition is an exact no-op at 2.7× the latency;
verification is actively harmful at 35× p90.

| | rerank-fast | + decompose | + verify |
|---|---:|---:|---:|
| recall@5 | **0.824** | 0.824 | 0.814 |
| MRR | **0.776** | 0.776 | 0.717 |
| latency p90 | **312 ms** | 804 ms | 11049 ms |

The metrics don't explain themselves, so the raw replies were read. The
decomposer called **47 of 59** questions atomic — including **8 of the 9
multi_hop** questions it exists for, several with an explicit *et* joining two
distinct points of law. The one kind it did split was `unanswerable`, where
splitting is useless. The verifier is the mirror image: it asked for more on
**55 of 59**, improved **0**, and damaged 1.

### Serving — [`docs/serving.md`](docs/serving.md)

FastAPI + SSE, streaming UI, one GPU. Generation is serialised behind a lock,
because two concurrent 4-bit generations on 8 GiB is an OOM, not a slow request.

| concurrency | first token | total median | answers/min |
|---:|---:|---:|---:|
| 1 | **1.1 s** | 20.1 s | 2.68 |
| 2 | 13.2 s | 39.5 s | 2.83 |
| 4 | 40.8 s | 69.5 s | 2.83 |

Streaming shows the first token **18.9× sooner** than the full answer — at one
client. At four it is **1.7×**, because time-to-first-token becomes queue wait
rather than prefill. Throughput is flat by design and the wall clock proves it:
179 s, 170 s, 169 s for the same eight answers.

## Six findings, five of them negative

1. **BM25 does not help on legal French — except in one place.** The hypothesis
   was recorded in `configs/experiments/02_hybrid.yaml` *before* measurement:
   exact tokens like article numbers should favour lexical search. Refuted alone
   and fused. Under reranking it was written up as contributing *nothing
   measurable* — until `cross_act` was rebuilt, at which point its whole effect
   appeared there and only there: **+0.056 recall@5 on cross-act questions, and
   0.000 on every other kind.** Still half a question, still not shipped, but the
   original claim was stronger than the evidence.
2. **A wider candidate pool made reranking worse at the k that matters.** Pool
   50 against pool 20: lower recall@5 and 2× slower. More candidates gave
   distractors more chances to score into the top five than true positives had
   to be rescued. It does win on recall@1 and MRR, so "dominated" would
   overstate it.
3. **Truncating passages to 512 tokens improved accuracy** (+4.5 recall@1), and
   was nearly written up as a speed optimisation. Separating it from the fp16
   change into a 2×2 showed fp16 = pure speed, truncation = the entire accuracy
   gain. An article's operative rule sits at its opening; long enumerations
   dilute the relevance signal.
4. **Fine-tuning optimised the proxy, not the task** — see above.
5. **Streaming's 18.9x speedup holds for one user**, and falls to 1.7x at four,
   because time-to-first-token becomes queue wait rather than prefill.
6. **Neither agent is calibrated.** The decomposer answers "one article" to 47
   of 59 questions; the verifier answers "not enough" to 55 of 59. One almost
   never acts, the other almost always does, and only the acting one moves the
   metrics — downward.

Each was found by reading outputs, not by reading metrics. The same method
caught table-of-contents contamination in the parser that both existing
self-checks passed, and a `citation_precision` definition that penalised
correct supplementary citations.

## Architecture

```
corpus (PDF/HTML)
  → parse → structure-aware chunking
  → hybrid retrieval (dense + BM25, RRF-fused)
  → cross-encoder reranking
  → generation with citations + abstention   [base model | QLoRA fine-tuned]
  → evaluation harness (retrieval + answer metrics)
  → FastAPI service + minimal UI
```

Every arrow in that diagram is a switch in a config file, so every arrow can be
turned off and measured.

## Roadmap

| Phase | Deliverable | Status |
|------:|-------------|--------|
| 0 | Project foundation: typed config, logging, CLI, tests | done |
| 1 | Ingestion, chunking, dense retrieval baseline | done — 3,056 articles, zero numbering gaps |
| 2 | Evaluation harness + gold QA set | done — 59 validated questions |
| 3 | Hybrid retrieval, reranking + ablations | done — +8.9 recall@5, BM25 refuted |
| 4 | QLoRA fine-tuning on a single 8 GiB GPU | done — negative result, not shipped |
| 5 | Agentic layer: decomposition, self-correction | done — [`docs/agentic.md`](docs/agentic.md); neither shipped |
| 6 | FastAPI service, streaming UI, Docker, benchmarks | done — [`docs/serving.md`](docs/serving.md) |
| 7 | Technical report and demo | report done — [`docs/report.md`](docs/report.md); demo waits on phase 6 |

## Setup

Requires **Python 3.13**. Not 3.14 — at time of writing PyTorch ships no CUDA
wheels for cp314, so a 3.14 environment silently runs on CPU.

```bash
py -3.13 -m venv .venv
```

```bash
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

```bash
.venv\Scripts\python.exe -m pip install -e ".[ingest,retrieval,eval,train,serve,dev]"
```

Then verify the environment actually sees the GPU:

```bash
.venv\Scripts\kora.exe doctor
```

Reproduce any row of the ablation table:

```bash
.venv\Scripts\kora.exe eval run -c configs/experiments/07_rerank_fast.yaml
```

Serve it, with a streaming UI at `http://127.0.0.1:8000`:

```bash
.venv\Scripts\kora.exe serve start -c configs/experiments/07_rerank_fast.yaml
```

## Repository layout

```
configs/            experiment definitions; base.yaml + one file per ablation
  experiments/
  training/
src/kora/
  config.py         typed, content-hashed experiment configuration
  ingest/           PDF parsing with extraction self-checks
  retrieval/        dense, lexical, RRF fusion, cross-encoder reranking
  generation/       prompt, citation parsing, abstention
  training/         QLoRA loop with loss-masking verification
  eval/             metrics, gold-set schema, experiment runner
docs/               one write-up per phase, with the negative results
data/               corpus, indexes, gold QA sets  (git-ignored, reproducible)
runs/               per-experiment metrics + resolved config
tests/
```

## Design notes

**Configs are content-hashed.** A run's identity is a hash of the settings that
produced it, so two experiments can never overwrite each other's results, and
each run directory stores the resolved config that produced its numbers.
Chunking and embedding settings get their own separate hash, so changing the
generator reuses the existing index instead of rebuilding it.

**That scheme broke, and the break is documented rather than patched over.**
Adding a field to a config model re-hashes every config that never mentions it,
so 9 of 11 runs recorded before a Phase 3 schema change can no longer be
regenerated from a config file. Every published number stays verifiable — the
stored resolved config is what diagnosed it — but the config-to-run link is
gone for those runs. See [`docs/reproducibility.md`](docs/reproducibility.md)
for the measurement, the cause, and why the fix is a pinning test rather than a
renumbering.

**Configs reject unknown keys.** A mistyped YAML key raises at load time rather
than falling back to a default and quietly invalidating a whole results table.

**The baseline is frozen.** `configs/base.yaml` is the reference point for every
comparison. Once experiments began it did not get "improved" — a moving baseline
makes an ablation table meaningless.

**Hypotheses are written into the config file before the run.** The BM25
hypothesis in `02_hybrid.yaml` and the fine-tuning hypothesis in
`docs/generation-baseline.md` were both recorded before measurement. Both
predicted the result, and the BM25 one predicted it wrong.

**Self-checks are assumed insufficient.** The parser reports numbering gaps, text
coverage and longest-article length, and a defect got past the first two of those.
Each phase ends by reading raw outputs.

## Known limits

- **n = 51 answerable questions.** Differences of one or two points are noise.
  The BM25 result holds because it is consistent across three independent
  comparisons, not because any single one is decisive.
- **`cross_act` was 4 questions and never moved.** Rebuilt from real inter-act
  textual bridges to 9 questions, at which point the row responds and BM25's one
  measurable contribution appeared. It is still 9 questions: a direction, not a
  magnitude.
- **`answers_citing_repealed` is near-vacuous.** The corpus holds three repealed
  articles and none of the seven superseded acts, so 0.000 means "the situation
  rarely arose", not "the system is safe".
- **No judgement of answer correctness.** Every metric is structural — citations,
  abstention, length. Whether the French prose states the law accurately is
  unmeasured, and would need an LLM judge or human review.
- **One embedding model.** Whether `multilingual-e5-base` suits French legal text
  is untested; only its combination with a reranker has been measured.

## Corpus and licensing

Source texts are the Actes uniformes published by OHADA. **They are not
redistributed in this repository.** The corpus, indexes and instruction data are
git-ignored and rebuilt on demand with `kora corpus download`,
`kora corpus parse`, `kora index build` and `kora train build-data`. Provenance
for each document is recorded at ingestion time.

Two derived files *are* committed, because they are the project's own work
rather than the corpus:

- `data/eval/gold_qa.jsonl` — the 64 hand-validated questions, ~21 KB, quoting
  only what a reference answer needs.
- `data/corpus_manifest.yaml` — title, year, legal status and source URL per
  act. No article text.

`data/training/sft.jsonl` is **not** committed: it quotes article bodies
verbatim inside its prompts, around 3.2 M characters of the Actes uniformes,
which would be redistribution. `kora train build-data` regenerates it.

## License

[MIT](LICENSE) for the code. The corpus and the models carry their own terms —
see [NOTICE.md](NOTICE.md).
