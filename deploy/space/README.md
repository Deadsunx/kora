---
title: Kora
emoji: ⚖️
colorFrom: gray
colorTo: red
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Cited question answering over OHADA business law
---

# Kora

Retrieval-augmented question answering over the **Actes uniformes OHADA**, the
business law shared by 17 mostly francophone African states.

Ask in French. Every answer cites the articles it came from, or declines.

- **Technical report:** https://deadsunx.github.io/kora/
- **Code:** https://github.com/Deadsunx/kora

## What is running

The configuration in `configs/experiments/07_rerank_fast.yaml`: dense retrieval
with `multilingual-e5-base`, cross-encoder reranking with `bge-reranker-v2-m3`
at fp16 and 512 tokens, and `Qwen3-4B-Instruct-2507` for generation.

Measured on 64 hand-validated questions: recall@5 0.815, and across the answer
set zero fabricated citations, zero repealed articles cited as current law, and
7 of 8 unanswerable questions correctly declined.

## Corpus

The Actes uniformes are published by OHADA and are **not redistributed**. The
index is loaded at start-up from a private dataset repository, which is why this
Space needs an `HF_TOKEN` secret with read access to it.

## Environment

| variable | effect |
|---|---|
| `KORA_DATASET_REPO` | private dataset holding the index and parsed articles |
| `KORA_QUANT` | `4bit` (measured) or `none` for fp16 where bitsandbytes cannot load |
| `KORA_RETRIEVAL_ONLY` | `1` serves search without generation, for CPU hardware |
