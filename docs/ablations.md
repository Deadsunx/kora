# Retrieval ablations

Corpus 3,056 articles · 59 human-validated questions, 51 answerable · RTX 4070
Laptop (8 GiB) · all runs share index `32e88149561b`

Every row is one config file in `configs/`, run through the same code path.
Latency is end-to-end per question — encoding, retrieval and reranking — measured
one question at a time, because that is what a user waits for.

## Results

| # | system | recall@1 | recall@5 | recall@20 | MRR | nDCG@5 | latency (median) |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | BM25 only | 0.382 | 0.637 | 0.824 | 0.550 | 0.544 | **5 ms** |
| 2 | dense + BM25 (RRF) | 0.431 | 0.716 | 0.902 | 0.611 | 0.612 | 26 ms |
| 3 | **dense** (baseline) | 0.490 | 0.735 | 0.873 | 0.647 | 0.647 | 19 ms |
| 4 | dense + BM25 + rerank, pool 50 | **0.608** | 0.804 | **0.931** | **0.755** | 0.742 | 1592 ms |
| 5 | dense + BM25 + rerank, pool 20 | 0.598 | **0.824** | 0.902 | 0.739 | 0.742 | 636 ms |
| 6 | **dense + rerank, pool 20** | 0.598 | **0.824** | 0.873 | 0.740 | **0.744** | **530 ms** |

**Best system: row 6.** recall@5 0.824 against the baseline's 0.735 — **+8.9
points, +12% relative** — for 28× the latency.

## Three findings, two of them negative

### 1. The BM25 hypothesis is refuted

The hypothesis recorded in `configs/experiments/02_hybrid.yaml` before any
measurement:

> Legal French is full of exact tokens that dense encoders blur — article
> numbers, "société anonyme", "acte uniforme". BM25 should help precisely where
> embeddings are weakest.

It does not, at any point in the pipeline:

- **Alone** (row 1) it is worse than dense everywhere: 0.637 against 0.735.
- **Fused with dense** (row 2) it is *worse than dense alone* at recall@5 —
  0.716 against 0.735 — and clearly worse at recall@1 and MRR. RRF surfaces
  articles BM25 found that dense missed, visible as better recall@10 and @20,
  but it displaces dense's precise top hits in the process. At `final_k=5`,
  which is what reaches the generator, that trade is a loss.
- **Under reranking** (rows 5 vs 6) it contributes *nothing measurable*:
  identical recall@5 (0.824), MRR within 0.001, nDCG within 0.002 — while
  costing about 100 ms per query.

The reasonable-sounding mechanism was real: BM25 does retrieve articles dense
retrieval misses, which is why recall@20 improves. It just does not survive the
cut to five, and a cross-encoder recovers the same articles without it.

**Decision: drop BM25.** It is retained in the config surface so the result stays
reproducible, not because it earns a place in the system.

### 2. A wider candidate pool made reranking worse

Rows 4 and 5 differ only in pool width — 50 candidates against 20.

| | recall@5 | recall@20 (ceiling) | latency |
|---|---:|---:|---:|
| pool 50 | 0.804 | 0.931 | 1592 ms |
| pool 20 | **0.824** | 0.902 | **636 ms** |

The wider pool raises the ceiling and the reranker cannot exploit it. Thirty
extra candidates offer more opportunities for a distractor to score into the top
five than for a true positive to be rescued. Pool 50 is strictly dominated: less
accurate at the k that matters and 2.5× slower.

This was very nearly missed. `03_hybrid_rerank.yaml` was written in Phase 0 with
`top_k: 50` while every other config used 20, so the first rerank result
confounded "add a cross-encoder" with "widen the pool". Configs 04 and 05 exist
because that confound was spotted while reading the table, not because it was
planned.

### 3. Reranking is the only component that earns its cost — and the cost is severe

+8.9 points of recall@5 and +9.3 of MRR is a large gain, and reranking is the
only change here that produced one. But 530 ms median and 1.7 s at p90 is not an
interactive latency, and this is a 3,000-document corpus on a dedicated GPU.

The honest reading is that the accuracy is real and the current implementation
is not yet shippable. Options not yet tested: a smaller cross-encoder, ONNX or
quantised inference, or reranking only when the dense scores are close.

## By question kind, best system (row 6)

| kind | n | recall@5 | MRR |
|---|---:|---:|---:|
| temporal | 5 | **1.000** | 0.850 |
| lookup | 33 | 0.879 | 0.767 |
| multi_hop | 9 | 0.722 | 0.794 |
| cross_act | 4 | 0.375 | 0.252 |

Reranking lifts every kind. Temporal questions reach 1.000, though on n=5.

The cross_act row moved not at all across all six systems — 0.375, 0.250, 0.375,
0.375, 0.375, 0.375. With four questions each worth 0.25, this is a count, not a
measurement, and no conclusion should be drawn from it. The category was cut
from eight to four during gold-set validation because most of its questions were
not genuinely cross-act; rebuilding it from inter-act textual cross-references
is the way to make it measurable.

## What these numbers do not cover

- **No generation.** These are retrieval metrics. Nothing here measures answer
  quality, citation accuracy or abstention.
- **`questions_with_repealed_in_top5` is 0.000 across every run and remains
  near-vacuous.** The corpus holds three individually repealed articles and none
  of the seven superseded acts, so the distractor set barely exists.
- **One embedding model.** Whether `multilingual-e5-base` is a good choice for
  French legal text is untested; only its combination with a reranker has been
  measured.
- **n = 51.** Differences of one or two points are within noise. The BM25 result
  is safe because it is consistent across three independent comparisons, not
  because any single one is decisive.
