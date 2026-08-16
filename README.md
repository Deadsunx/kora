# Kora

**Retrieval-augmented question answering over French-language African business law (OHADA).**

> Status: phases 0–4 complete, report written. Every number below is
> reproducible from a config file in `configs/`, and links to the write-up that
> produced it.

**→ [Technical report](docs/report.md)** — the whole project in one document,
organised around how each finding was found.

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

**3,056 articles** parsed from source PDFs across 10 acts · **59 human-validated
questions**, 51 answerable · **12 recorded runs** · 202 tests · single RTX 4070
Laptop, 8 GiB.

| | |
|---|---|
| Corpus | AUSCGIE 1089 · AUPSRVE 448 · AUSCOOP 397 · AUPCAP 377 · AUDCG 307 · AUS 228 · AUDCIF 123 · AUA 38 · AUCTMR 31 · AUM 18 |
| Retrieval | `multilingual-e5-base` dense + `bge-reranker-v2-m3` cross-encoder |
| Generation | Qwen3-4B-Instruct-2507, 4-bit NF4 |
| Fine-tuning | QLoRA r=16, paged 8-bit AdamW, gradient checkpointing |

## Results

### Retrieval — [`docs/ablations.md`](docs/ablations.md)

| system | recall@5 | MRR | nDCG@5 | latency (median) |
|---|---:|---:|---:|---:|
| BM25 only | 0.637 | 0.550 | 0.544 | **5 ms** |
| dense + BM25 (RRF) | 0.716 | 0.611 | 0.612 | 26 ms |
| dense (frozen baseline) | 0.735 | 0.647 | 0.647 | 19 ms |
| dense + BM25 + rerank, pool 50 | 0.804 | 0.755 | 0.742 | 1592 ms |
| dense + rerank, pool 20 | **0.824** | 0.740 | 0.744 | 530 ms |
| **dense + rerank, fp16, 512 tokens** | **0.824** | **0.776** | **0.766** | **193 ms** |

**+8.9 points of recall@5 over the baseline (+12% relative), at 193 ms.**
Reranking is the only component that earned its cost.

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

## Four findings, three of them negative

1. **BM25 does not help on legal French.** The hypothesis was recorded in
   `configs/experiments/02_hybrid.yaml` *before* measurement: exact tokens like
   article numbers should favour lexical search. Refuted three independent ways —
   alone, fused, and under reranking, where it contributes nothing measurable for
   ~100 ms per query.
2. **A wider candidate pool made reranking worse.** Pool 50 against pool 20:
   lower recall@5 and 2.5× slower. More candidates gave distractors more chances
   to score into the top five than true positives had to be rescued.
3. **Truncating passages to 512 tokens improved accuracy** (+4.9 recall@1), and
   was nearly written up as a speed optimisation. Separating it from the fp16
   change into a 2×2 showed fp16 = pure speed, truncation = the entire accuracy
   gain. An article's operative rule sits at its opening; long enumerations
   dilute the relevance signal.
4. **Fine-tuning optimised the proxy, not the task** — see above.

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
| 5 | Agentic layer: decomposition, self-correction | — |
| 6 | FastAPI service, streaming UI, Docker, benchmarks | — |
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
produced it, so two experiments can never overwrite each other's results, and a
results table months old can be traced back to the exact system that produced it.
Chunking and embedding settings get their own separate hash, so changing the
generator reuses the existing index instead of rebuilding it.

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
- **`cross_act` is 4 questions and never moved** across any system. That is a
  count, not a measurement. Rebuilding it from real inter-act cross-references is
  outstanding.
- **`answers_citing_repealed` is near-vacuous.** The corpus holds three repealed
  articles and none of the seven superseded acts, so 0.000 means "the situation
  rarely arose", not "the system is safe".
- **No judgement of answer correctness.** Every metric is structural — citations,
  abstention, length. Whether the French prose states the law accurately is
  unmeasured, and would need an LLM judge or human review.
- **One embedding model.** Whether `multilingual-e5-base` suits French legal text
  is untested; only its combination with a reranker has been measured.

## Corpus and licensing

Source texts are the Actes uniformes published by OHADA. They are not
redistributed in this repository; `data/` is git-ignored and rebuilt on demand
with `kora corpus download` and `kora corpus parse`. Provenance for each document
is recorded at ingestion time.

## License

MIT for the code. The corpus is subject to its own terms.
