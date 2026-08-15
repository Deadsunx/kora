# Kora

**Retrieval-augmented question answering over French-language African business law (OHADA).**

> Status: Phase 0 of 8 — foundation. Nothing here is measured yet, and this
> README will not claim otherwise. Every number that eventually appears below
> will be reproducible from a config file in `configs/`.

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

This ordering is deliberate. The evaluation harness (Phase 2) is built *before*
any retrieval improvements (Phase 3), because an improvement you cannot measure
is a preference, not a result.

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
| 0 | Project foundation: typed config, logging, CLI, tests | in progress |
| 1 | Ingestion, chunking, dense retrieval baseline | — |
| 2 | Evaluation harness + gold QA set | — |
| 3 | Hybrid retrieval, reranking, query rewriting + ablations | — |
| 4 | QLoRA fine-tuning on a single 8 GiB GPU | — |
| 5 | Agentic layer: decomposition, self-correction, abstention | — |
| 6 | FastAPI service, streaming UI, Docker, benchmarks | — |
| 7 | Technical report and demo | — |

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

## Repository layout

```
configs/            experiment definitions; base.yaml + one file per ablation
  experiments/
src/kora/
  config.py         typed, content-hashed experiment configuration
  paths.py          canonical filesystem layout
  logging.py        structured logging
  cli.py            single entry point for every pipeline stage
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
comparison. Once experiments begin it does not get "improved" — a moving
baseline makes an ablation table meaningless.

## Corpus and licensing

Source texts are the Actes uniformes published by OHADA. They are not
redistributed in this repository; `data/` is git-ignored and rebuilt from
`scripts/` on demand. Provenance for each document is recorded at ingestion time.

## License

MIT for the code. The corpus is subject to its own terms.
