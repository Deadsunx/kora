# Third-party content

The [MIT licence](LICENSE) covers the code in this repository. It does not cover
the corpus.

## The Actes uniformes OHADA

The source texts are published by OHADA and are subject to their own terms.
**They are not redistributed here.** These paths are git-ignored and rebuilt
from source:

| path | rebuilt by |
|---|---|
| `data/raw/` | `kora corpus download` |
| `data/interim/` | `kora corpus parse` |
| `data/processed/`, `data/indexes/` | `kora index build` |
| `data/training/` | `kora train build-data` |

`data/training/sft.jsonl` is called out specifically: it quotes article bodies
verbatim inside its prompts — roughly 3.2 million characters of the Actes
uniformes — which would be redistribution. It is generated locally and never
committed.

## What is committed

Two derived files, both the project's own work rather than the corpus:

- **`data/eval/gold_qa.jsonl`** — 64 hand-written, hand-validated questions with
  short reference answers, about 21 KB. Quotes only what is needed to state the
  expected answer for each question.
- **`data/corpus_manifest.yaml`** — bibliographic metadata per act: title, year,
  legal status, supersession links, source URL. No article text.

## Models

The models are downloaded from Hugging Face at run time and carry their own
licences:

| model | role |
|---|---|
| `intfloat/multilingual-e5-base` | dense retrieval |
| `BAAI/bge-reranker-v2-m3` | cross-encoder reranking |
| `Qwen/Qwen3-4B-Instruct-2507` | answer generation |
