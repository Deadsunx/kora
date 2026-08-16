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
| 6 | dense + rerank, pool 20 | 0.598 | **0.824** | 0.873 | 0.740 | 0.744 | 530 ms |
| 7 | **dense + rerank, fp16, 512 tokens** | **0.647** | **0.824** | 0.873 | **0.776** | **0.766** | **193 ms** |

**Best system: row 7.** recall@5 0.824 against the baseline's 0.735 — **+8.9
points, +12% relative** — for 10× the latency, at 193 ms median and 247 ms p90.

Rows 6 and 7 differ only in reranker precision and passage truncation; see
"Making reranking fast" below, where those two changes are separated.

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

### 3. Reranking is the only component that earns its cost

+8.9 points of recall@5 is a large gain, and reranking is the only change here
that produced one. As first configured it cost 538 ms median and 1.7 s at p90,
which is not an interactive latency on a 3,000-document corpus with a dedicated
GPU. That has since been fixed — see below.

## Making reranking fast

Profiling first, rather than guessing: **reranking was 97.7% of end-to-end
latency** (encode 15 ms, search 0.8 ms, rerank 486 ms median / 1771 ms p90). Two
causes, both configuration rather than model choice:

- the cross-encoder loaded in **fp32**, on a GPU built for half precision
- **`max_length` defaulted to 8192**. Attention is quadratic, so one long
  article dominated whatever batch it landed in — measured at 2,622 tokens →
  389 ms against 4,514 tokens → 1,771 ms. **1.7× the tokens, 4.5× the time.**

Both were changed at once and the result was faster *and* more accurate, which
is exactly the shape of a confounded result, so it was split into a 2×2:

| max_length | precision | recall@1 | recall@5 | MRR | nDCG@5 | median | p90 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 8192 | fp32 | 0.598 | 0.824 | 0.740 | 0.744 | 538 ms | 1673 ms |
| 8192 | fp16 | 0.598 | 0.824 | 0.740 | 0.744 | 200 ms | 599 ms |
| 512 | fp32 | **0.647** | 0.824 | **0.776** | **0.766** | 624 ms | 794 ms |
| **512** | **fp16** | **0.647** | 0.824 | **0.776** | **0.766** | **193 ms** | **247 ms** |

The separation is complete:

**fp16 costs no accuracy at all** — identical to three decimals in both length
settings. A 2.7× speedup for nothing. Ranking is an ordering problem, and score
gaps here are far larger than fp16 error.

**Truncation is responsible for the entire accuracy gain**: +4.9 points of
recall@1 and +3.6 of MRR, in both precisions. This is a finding rather than an
optimisation. An article's operative statement sits at its opening, and long
enumerations — article 13's list of statutory mentions, article 51's six
categories of insaisissable goods — dilute the relevance signal across text that
does not bear on the question. Capping at 512 tokens acts as a prior about where
legal meaning lives.

Combined: **2.8× faster median, 6.8× faster p90**, and better ranking. The
gap over the dense baseline falls from 28× latency to 10×.

Had the two changes not been separated, this would have been written up as "we
made reranking fast and it happened to help", and the more interesting half
would have been invisible.

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
not genuinely cross-act.

**This has since been rebuilt from inter-act textual cross-references** — see
"Making cross_act measurable" below. The numbers in the table above are
unchanged, because they are the headline over validated questions and the new
questions are not yet validated.

## Making cross_act measurable

The category was rebuilt the way it should have been built: from the text rather
than from topic. The corpus was mined for articles that **name another acte
uniforme**, which found **52 bridges across 21 act pairs**. Seven of those became
questions, three of them resting on article-level pointers where the source
article names the exact target article:

| bridge | question |
|---|---|
| AUDCG art 40 → *article 51 AUS* | who may request a sûreté inscription, on what form |
| AUPSRVE art 245-2 → *article 136 AUDCG* | what a saisie of a fonds de commerce covers |
| AUDCIF art 102 → *article 849 AUSCGIE* | who publishes a half-year activity statement, and what it contains |
| AUSCGIE art 149 → AUA | which text governs arbitration between shareholders |
| AUSCOOP art 118 → AUA | may a cooperative create its own arbitration organ |
| AUS art 137 → AUPSRVE saisie-attribution | how a pledged bank balance is determined |
| AUS art 142 → AUPSRVE saisie conservatoire | what governs judicial pledge of shares |

`cross_act` goes from 4 questions to 11, each worth 0.09 rather than 0.25.

**The row now moves**, measured over all questions:

| system | cross_act recall@5 | MRR |
|---|---:|---:|
| BM25 only | 0.455 | 0.642 |
| dense baseline | 0.500 | 0.506 |
| dense + rerank, fp16, 512 | **0.545** | **0.804** |

And it has the right difficulty profile. On the best system, **all seven new
questions retrieve their first gold article at rank 1** — the question's own
vocabulary finds the anchor immediately — while **five of the seven retrieve
only one of two.** The article that *mentions* the other act is easy; the
article *in* the other act is not. That is precisely the thing a cross-act
question should be testing, and it is why recall@5 and MRR now say different
things: MRR reports that the anchor is easy, recall that the bridge is hard.

One incidental result the larger category exposed: **cross_act is the only kind
where BM25 beats dense on a metric** — MRR 0.642 against 0.506. These questions
carry distinctive terms of art and explicit article references, which is exactly
the mechanism the refuted Phase 3 hypothesis proposed. It was invisible while the
category had four questions. It does not revive the hypothesis — BM25 is still
worse at recall@5 here and worse everywhere else — but it locates the one place
the reasoning behind it was pointing.

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
