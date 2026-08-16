# Retrieval ablations

Corpus 3,056 articles · **64 human-validated questions, 56 answerable** · RTX
4070 Laptop (8 GiB) · all runs share index `32e88149561b`

Every row is one config file in `configs/`, run through the same code path.
Latency is end-to-end per question — encoding, retrieval and reranking — measured
one question at a time, because that is what a user waits for.

> **These numbers were re-measured after `cross_act` was rebuilt** from 4
> questions to 9 (see "Making cross_act measurable"). The gold set grew from 59
> questions to 64, so every figure below differs from the version published
> before that. One conclusion changed as a result, and it is called out in
> finding 1. Runs are in `runs/*-gold64/`; the earlier runs are kept alongside
> them, each recording its own gold composition.

## Results

| # | system | recall@1 | recall@5 | recall@20 | MRR | nDCG@5 | latency (median) |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | BM25 only | 0.381 | 0.637 | 0.812 | 0.576 | 0.551 | **5 ms** |
| 2 | dense + BM25 (RRF) | 0.417 | 0.708 | 0.893 | 0.625 | 0.611 | 26 ms |
| 3 | **dense** (baseline) | 0.461 | 0.735 | 0.866 | 0.652 | 0.641 | 17 ms |
| 4 | dense + BM25 + rerank, pool 50 | 0.622 | 0.815 | **0.893** | **0.800** | 0.763 | 461 ms |
| 5 | dense + BM25 + rerank, pool 20 | 0.613 | **0.824** | 0.893 | 0.786 | 0.760 | 235 ms |
| 6 | **dense + rerank, fp16, 512 tokens** | **0.631** | 0.815 | 0.866 | 0.796 | **0.765** | **205 ms** |

**Best system: row 6.** recall@5 0.815 against the baseline's 0.735 — **+8.0
points, +11% relative** — for 12× the latency, at 205 ms median and 256 ms p90.
Row 5 reaches a higher recall@5 by 0.009, or half a question; see finding 1.

The old row "dense + rerank, pool 20" at 530 ms is **not reproducible from a
config file** and has been dropped rather than re-run under a different system.
`05_dense_rerank_k20.yaml` said "reranker defaults", and the defaults became
fp16/512 in Phase 3, so that file now produces row 6. See
[`reproducibility.md`](reproducibility.md).

## Three findings, two of them negative

### 1. The BM25 hypothesis is refuted

The hypothesis recorded in `configs/experiments/02_hybrid.yaml` before any
measurement:

> Legal French is full of exact tokens that dense encoders blur — article
> numbers, "société anonyme", "acte uniforme". BM25 should help precisely where
> embeddings are weakest.

It is refuted as a general claim:

- **Alone** (row 1) it is worse than dense everywhere: 0.637 against 0.735.
- **Fused with dense** (row 2) it is *worse than dense alone* at recall@5 —
  0.708 against 0.735 — and clearly worse at recall@1 and MRR. RRF surfaces
  articles BM25 found that dense missed, visible as better recall@20, but it
  displaces dense's precise top hits in the process. At `final_k=5`, which is
  what reaches the generator, that trade is a loss.

### But the earlier version of this finding was too strong

It previously read: *under reranking, BM25 contributes nothing measurable.* That
was measured on a gold set whose `cross_act` category was four questions and did
not move across any system. With nine bridge-anchored questions, it does move,
and BM25's contribution is measurable and **entirely localised**:

| | dense + BM25, pool 20 | dense only, 512/fp16 | delta |
|---|---:|---:|---:|
| **cross_act** recall@5 | **0.630** | 0.574 | **+0.056** |
| lookup recall@5 | 0.879 | 0.879 | 0.000 |
| multi_hop recall@5 | 0.722 | 0.722 | 0.000 |
| temporal recall@5 | 1.000 | 1.000 | 0.000 |
| recall@20 | 0.893 | 0.866 | +0.027 |
| recall@1 | 0.613 | 0.631 | −0.018 |
| MRR | 0.786 | 0.796 | −0.010 |
| latency | 235 ms | 205 ms | +30 ms |

Three of the four question kinds are **identical to three decimals**. The whole
of BM25's effect under reranking is on cross-act questions — which is exactly
where the original hypothesis pointed: those questions carry explicit article
references and fixed terms of art, the tokens dense encoders blur.

The honest reading is narrow. It is +0.056 on nine questions, which is half a
question, and it is bought with recall@1, MRR, nDCG and 30 ms. **The shipping
decision does not change: drop BM25.** But "contributes nothing measurable" was
an artefact of a category too small to measure anything, and this is what
rebuilding it revealed.

That is worth stating plainly, because it cuts against the project's own
headline result. The mechanism behind the refuted hypothesis was real — BM25
does retrieve articles dense retrieval misses, which is why recall@20 improves —
and on the one question kind built to require following a reference between two
acts, some of them now survive the cut to five.

**Decision: drop BM25.** It is retained in the config surface so the result stays
reproducible, and now also because the cross_act effect is worth re-testing if
that category ever grows past nine questions.

### 2. A wider candidate pool made reranking worse

Rows 4 and 5 differ only in pool width — 50 candidates against 20.

| | recall@1 | recall@5 | recall@20 (ceiling) | MRR | latency |
|---|---:|---:|---:|---:|---:|
| pool 50 | **0.622** | 0.815 | 0.893 | **0.800** | 461 ms |
| pool 20 | 0.613 | **0.824** | 0.893 | 0.786 | **235 ms** |

The wider pool cannot be exploited at the k that matters. Thirty extra
candidates offer more opportunities for a distractor to score into the top five
than for a true positive to be rescued, and pool 50 is 2x slower for it.

An earlier version of this called pool 50 *strictly dominated*. That was wrong
then and is wrong now: it wins on recall@1 and MRR in both measurements. It
ranks its top hit slightly better and its top *five* slightly worse. Since
`final_k=5` is what reaches the generator, pool 20 remains the right choice —
but "dominated" overstated it, which is the same shape of error as finding 1.

This was very nearly missed. `03_hybrid_rerank.yaml` was written in Phase 0 with
`top_k: 50` while every other config used 20, so the first rerank result
confounded "add a cross-encoder" with "widen the pool". Configs 04 and 05 exist
because that confound was spotted while reading the table, not because it was
planned.

### 3. Reranking is the only component that earns its cost

+8.0 points of recall@5 is a large gain, and reranking is the only change here
that produced one. As first configured it cost 567 ms median and 1.8 s at p90,
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
| 8192 | fp32 | 0.586 | 0.815 | 0.763 | 0.745 | 567 ms | 1775 ms |
| 8192 | fp16 | 0.586 | 0.815 | 0.763 | 0.745 | 203 ms | 619 ms |
| 512 | fp32 | **0.631** | 0.815 | **0.796** | **0.765** | 648 ms | 833 ms |
| **512** | **fp16** | **0.631** | 0.815 | **0.796** | **0.765** | **205 ms** | **256 ms** |

The separation is complete:

**fp16 costs no accuracy at all** — identical to three decimals in both length
settings. A 2.8× speedup for nothing. Ranking is an ordering problem, and score
gaps here are far larger than fp16 error.

**Truncation is responsible for the entire accuracy gain**: +4.5 points of
recall@1 and +3.3 of MRR, in both precisions. This is a finding rather than an
optimisation. An article's operative statement sits at its opening, and long
enumerations — article 13's list of statutory mentions, article 51's six
categories of insaisissable goods — dilute the relevance signal across text that
does not bear on the question. Capping at 512 tokens acts as a prior about where
legal meaning lives.

Combined: **2.8× faster median, 6.9× faster p90**, and better ranking. The
gap over the dense baseline falls from 33× latency to 12×.

Had the two changes not been separated, this would have been written up as "we
made reranking fast and it happened to help", and the more interesting half
would have been invisible.

## By question kind, best system (row 6)

| kind | n | recall@5 | MRR |
|---|---:|---:|---:|
| temporal | 5 | **1.000** | 0.850 |
| lookup | 33 | 0.879 | 0.798 |
| multi_hop | 9 | 0.722 | 0.794 |
| cross_act | 9 | 0.574 | 0.760 |

Reranking lifts every kind. Temporal questions reach 1.000, though on n=5.

**cross_act was four questions and never moved** — 0.375, 0.250, 0.375, 0.375,
0.375, 0.375 across every system measured. With four questions each worth 0.25
that was a count, not a measurement. The category had been cut from eight to
four during gold-set validation because most of its questions were not genuinely
cross-act.

It has since been rebuilt from inter-act textual cross-references, and the row
now responds — see below. Two consequences already: it exposed BM25's only
measurable contribution under reranking (finding 1), and it is the one kind
where reranking buys **no recall@5 at all**, 0.574 with and without, lifting
only MRR from 0.500 to 0.760. The cross-encoder reorders what dense found; it
does not rescue the article on the far side of the reference.

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

Seven were drafted; **five survived review** and two were rejected — q065 as
not independent of another question, q066 because its second article was
padding, which is the same defect that cut the category from eight to four the
first time. One had its gold corrected: AUPSRVE art 245-2 points at articles
136 *and* 137 AUDCG, and 137 was missing.

`cross_act` goes from 4 questions to 9, each worth 0.111 rather than 0.25.

**The row now moves:**

| system | cross_act recall@5 | MRR |
|---|---:|---:|
| BM25 only | 0.463 | 0.562 |
| dense baseline | 0.574 | 0.500 |
| dense + BM25 + rerank, pool 20 | **0.630** | 0.734 |
| dense + rerank, fp16, 512 | 0.574 | **0.760** |

And it has the right difficulty profile. On the best system, **all seven new
questions retrieve their first gold article at rank 1** — the question's own
vocabulary finds the anchor immediately — while **five of the seven retrieve
only one of two.** The article that *mentions* the other act is easy; the
article *in* the other act is not. That is precisely the thing a cross-act
question should be testing, and it is why recall@5 and MRR now say different
things: MRR reports that the anchor is easy, recall that the bridge is hard.

**This is where BM25 turns out to earn something.** Adding it under reranking
lifts cross_act recall@5 from 0.574 to 0.630 while leaving every other kind
identical to three decimals — see finding 1, which was rewritten because of it.
The category being too small to measure anything had produced a stronger
negative conclusion than the evidence supported.

## What these numbers do not cover

- **No generation.** These are retrieval metrics. Nothing here measures answer
  quality, citation accuracy or abstention.
- **`questions_with_repealed_in_top5` is 0.000 across every run and remains
  near-vacuous.** The corpus holds three individually repealed articles and none
  of the seven superseded acts, so the distractor set barely exists.
- **One embedding model.** Whether `multilingual-e5-base` is a good choice for
  French legal text is untested; only its combination with a reranker has been
  measured.
- **n = 56, and n = 9 for cross_act.** Differences of one or two points are
  within noise. The BM25 refutation is safe at the headline because it is
  consistent across independent comparisons; its cross_act *exception* is half a
  question and is reported as a direction, not a magnitude.
