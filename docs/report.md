# Kora: a measured retrieval system for French-language African business law

Technical report · August 2026 · 3,056 articles · 64 validated questions · 23
recorded runs · 252 tests · one RTX 4070 Laptop, 8 GiB

---

## Summary

Kora answers questions about the **Actes uniformes OHADA** — the business law
shared by 17 mostly francophone African states — with citations to specific
articles, or refuses to answer. It reaches **recall@5 = 0.815** at 205 ms, and
generates answers that never fabricate a citation, never present repealed text as
law, and correctly decline 7 of 8 unanswerable questions.

Those are the headline numbers. They are not the point of this report.

The point is that most of the project's principal findings are **negative**: a
well-motivated hypothesis about lexical search was refuted, a wider candidate
pool made reranking worse, an accuracy gain was very nearly written up as a
speed optimisation, a QLoRA fine-tune that improved almost every metric made the
system worse, neither agentic mechanism was calibrated enough to help, and the
streaming service's headline 18.9× speedup turned out to hold only for a single
user. Each was found by reading raw output, or by measuring a condition that was
not flattering. Two of them were invisible to every metric in the harness.

Reranking is the one component that earned its cost.

This report is organised around how each was found, because that is the
transferable part.

---

## 1. Why this corpus

Retrieval systems are overwhelmingly built and evaluated in English. The
techniques are assumed to transfer. Usually nobody checks.

OHADA law is a good place to check:

- **It is legal French.** Dense encoders trained mainly on English web text blur
  precisely the tokens that carry meaning here — article numbers, fixed terms of
  art like *société anonyme* or *sûreté mobilière*, cross-references between
  acts. That is a testable claim, and §5 tests it.
- **It is highly structured.** Livre → Titre → Chapitre → Article, with
  contiguous article numbering. That structure is both a gift for chunking and a
  free correctness invariant: a gap in the numbering means a heading was
  swallowed.
- **Answers must be attributable.** "Article 640 AUSCGIE" is checkable by a
  human in seconds. A legal assistant that cannot cite, or will not say *I don't
  know*, is worse than useless — so citation accuracy and abstention are primary
  metrics here rather than afterthoughts.
- **It is genuinely low-resource.** No benchmark, no gold QA set, no clean
  corpus. All three had to be built, which is most of the work and most of the
  interesting failures.

## 2. Method

Five commitments, made before results existed, each of which cost something
later.

**The baseline is frozen.** `configs/base.yaml` is the reference for every
comparison and was never "improved" once experiments began. A baseline that
drifts makes an ablation table meaningless.

**Configs are content-hashed.** A run's identity is a SHA-256 of the settings
that produced it. Two experiments cannot overwrite each other's results, and
every run directory stores the resolved config behind its numbers. Chunking and
embedding settings carry a *separate* hash, so changing the generator reuses the
existing index rather than rebuilding it — reuse that is legitimate because the
hash proves the index is the same.

This guarantee turned out to be weaker than stated. §9 reports where it failed.

**Configs reject unknown keys.** A mistyped YAML key raises at load time instead
of silently falling back to a default and invalidating a whole results table.

**Hypotheses are written down before the run.** The BM25 hypothesis was recorded
in `configs/experiments/02_hybrid.yaml`; the fine-tuning hypothesis in
`docs/generation-baseline.md`. Both were committed before the measurement. One
was wrong, which is the reason for the practice.

**Only human-validated questions appear in reported numbers.** Every run prints
its gold composition and sample size alongside the metrics, because a recall
figure without an *n* behind it is not a result.

---

## 3. Building the corpus: five defects, five different detectors

The corpus had to be parsed from source PDFs. Three self-checks were built into
the parser:

| check | catches |
|---|---|
| `numbering_gaps()` | a swallowed heading — numbering must be contiguous |
| `text_coverage()` | bodies discarded while numbering stays perfect |
| `longest_article_chars` | text merged into the wrong article, which leaves both of the above healthy |

The third exists because of defect 1.

**Result: 3,056 articles across 10 acts, zero numbering gaps, 95.5–99.3%
coverage.** Five exact agreements with an independently published reference
corpus, and both disagreements explained.

### Defect 1 — table-of-contents contamination (found by the length distribution)

A contents page repeats every article heading followed by dot leaders:

```
Article 2
..................................................... 14
```

The heading matches the article pattern exactly. Left alone it creates a phantom
article whose body is a row of dots and — worse — *claims the number*, so the
genuine article forty pages later is dismissed as a duplicate cross-reference and
its text appended to whatever article happens to be open.

**Both existing checks passed.** Numbering stayed contiguous, because the
phantoms covered every number. Coverage stayed at 98.7%, because the text was
still present, merely misfiled.

It surfaced only on inspecting article lengths: 191 articles of pure dots, and
three blocks of 19,000–31,000 characters where real articles run to a few
thousand. `longest_article_chars` was added as the third check as a result.

### Defect 2 — `Article premier` (found by the numbering invariant)

French legal drafting numbers the first article in words. AUPSRVE-2023 carries
the convention into its inserted articles (`Article premier-11`) while
cross-referencing the same provisions numerically in the body (`l'article 1-10`).

A digits-only pattern does not merely miss article 1: every `premier-N` heading
goes unrecognised and its text is absorbed into the preceding article. The gap
check reported exactly **one** missing number, which was the visible tip of
**17** lost articles. Fixing it took AUPSRVE from 431 articles at 92.7% coverage
to 448 at 98.4%.

A detail worth recording: in the corrected pattern, alternation order is
load-bearing. `(?:premier|1\s*er|\d+)` — with `\d+` last, because putting it
first makes `1er` match as `1` followed by a stray `er`.

### Defect 3 — repealed articles destroyed by a heuristic (found while writing gold questions)

A repealed article is printed as its heading followed by the single word
`Abrogé`:

```
Article 12
Abrogé
Article 13
Les petites entités sont assujetties, sauf option, au Système minimal…
```

That word is short, has no terminal full stop, and is not a structural marker —
so the rubric heuristic classified it as a marginal title belonging to the *next*
heading. **Two errors from one heuristic: article 12 lost its only content and
was left empty, while article 13 — perfectly in force — was made to look
repealed.**

Three articles in AUDCIF were affected (12, 27, 60). They had been visible the
whole time as three zero-length bodies in the length distribution. They were
noticed and not chased.

The fix is narrow: a candidate rubric is reclaimed from the previous article only
when that article survives losing it. If removing it would leave an empty body,
it was content, not a title.

This defect also produced a **wrong claim in an earlier version of the parse
document**, which named 13, 28 and 61 as the repealed articles — corrupted
parser output read as ground truth. The repealed articles are 12, 27 and 60.

It also forced a design change. Legal status is not only a property of
documents: AUDCIF-2017 is in force while three of its articles are not, so
reading status from the manifest alone would score a citation of article 12 as
perfectly safe. Repealed articles now carry an explicit `[TEXTE ABROGÉ]` marker
into their *indexed text*, so the warning reaches the generator's context window
rather than living only in metadata.

### Defect 4 — short-page furniture detection (found by a unit test)

Header/footer detection sampled the first and last three lines of each page —
which on a short page is every line, letting body text be deleted as furniture.
Real pages are long enough that this never triggered in production. Found only
because a test constructed a short page.

### Defect 5 — page numbers in article bodies (found by a manual spot-check)

Spot-checking bodies against the reference corpus showed one beginning
`"page 3 / 217"`. Page numbers differ on every page, so frequency-based furniture
detection is structurally incapable of seeing them. Now removed by pattern.

### The AUSCGIE disagreement, resolved

The reference corpus reports 1,392 articles for AUSCGIE-2014 against our 1,089 —
an apparent deficit of 303. Comparing article *identifiers* rather than counts:

| | reference | ours |
|---|---:|---:|
| rows | 1,392 | 1,089 |
| **distinct article numbers** | **811** | **1,089** |
| duplicated labels | 549 | 0 |
| articles absent from the other set | **0** | **278** |

Our extraction is a **strict superset** — every article the reference holds is
present in ours, plus 278 more, three of which were spot-checked as genuine
provisions. The deficit was an artefact of comparing our distinct-article count
against their row count.

Stated plainly because an earlier version of that document speculated about
"different counting units", and the speculation was wrong.

### The lesson

Each of the five defects was found by a *different* mechanism — the numbering
invariant, the length distribution, a unit test, a manual spot-check, and
building the gold set. **None of them caught what another caught.** That is the
argument for having several, and for reading output by hand as well as by metric.

---

## 4. The gold set, and what validation cost

60 questions were drafted with model assistance, then reviewed by hand against
the printed article text. The review asks exactly one question:

> **Does the text of the gold article actually support the reference answer?**

Not "is this true about OHADA law". Not "is this a good question". Only: *if a
system retrieved exactly these articles, could it produce that answer?* If the
article does not contain the fact, the gold label is wrong and every system is
scored against an impossible target — the metric silently measures nothing.

Provenance is tracked per question (`human`, `llm_drafted`,
`llm_drafted_human_validated`) and only validated questions count towards the
headline. The schema enforces category coherence: `unanswerable` ⇔ no gold
article, `lookup` ⇔ exactly one, `multi_hop` ⇔ at least two, `cross_act` ⇔ at
least two *documents*.

**Outcome: 60 → 59 questions, 11 repointed at different articles.**

Validation raised recall@5 from 0.679 to 0.735 without a single change to the
system — the improvement is the removal of gold articles that were never required
to answer their question. Retrieval had been marked down for failing to find
passages that did not need finding.

That cuts both ways, and this is the part worth reporting: **the same review that
raised the headline number destroyed the one finding the earlier run appeared to
support.** An apparent cross-act collapse at 0.292 turned out to rest on eight
questions, three of which were not cross-act at all and one of which was
malformed. The category was cut to four.

Two smaller results from the review:

- **q057 was mislabelled `unanswerable`.** It asks about minimum share capital
  in Cameroon; the OHADA general rule *is* in the corpus. Left alone, it would
  have penalised a system for correctly citing article 387. It was reclassified —
  and in §6 it catches the opposite failure.
- **OHADA acts are self-contained.** Genuine `cross_act` questions need an
  explicit textual bridge between two acts, which is rarer than it sounds. This is
  why the category is four questions and why it is recorded rather than reported.

---

## 5. Retrieval: what earns its cost

Every row is one config file, run through the same code path, on the same index
(`32e88149561b`). Latency is end-to-end per question, measured one question at a
time, because that is what a user waits for.

| system | recall@1 | recall@5 | MRR | nDCG@5 | median |
|---|---:|---:|---:|---:|---:|
| BM25 only | 0.381 | 0.637 | 0.576 | 0.551 | **5 ms** |
| dense + BM25 (RRF) | 0.417 | 0.708 | 0.625 | 0.611 | 26 ms |
| dense — frozen baseline | 0.461 | 0.735 | 0.652 | 0.641 | 17 ms |
| dense + BM25 + rerank, pool 50 | 0.622 | 0.815 | **0.800** | 0.763 | 461 ms |
| dense + BM25 + rerank, pool 20 | 0.613 | **0.824** | 0.786 | 0.760 | 235 ms |
| **dense + rerank, fp16, 512 tok** | **0.631** | 0.815 | 0.796 | **0.765** | **205 ms** |

**Best system: the last row. +8.0 points of recall@5 over the baseline, +11%
relative, at 205 ms median and 256 ms p90.**

These figures were re-measured after §4's `cross_act` category was rebuilt from
4 questions to 9, which grew the gold set to 64 and changed one conclusion.

### Finding 1: the BM25 hypothesis is refuted

Recorded in the config file before measurement:

> Legal French is full of exact tokens that dense encoders blur — article
> numbers, "société anonyme", "acte uniforme". BM25 should help precisely where
> embeddings are weakest.

It does not, at any point in the pipeline:

- **Alone**, it is worse than dense everywhere: 0.637 against 0.735.
- **Fused with dense**, it is *worse than dense alone* at recall@5 — 0.716
  against 0.735 — and clearly worse at recall@1 and MRR. RRF does surface
  articles that dense missed, visible as better recall@20, but it displaces
  dense's precise top hits doing it. At `final_k=5`, which is what reaches the
  generator, that trade is a loss.
- **Under reranking**, it was written up as contributing *nothing measurable*.
  That was true on a gold set whose `cross_act` category was four questions and
  never moved. Rebuilt to nine, BM25's effect appears — and appears in exactly
  one place: **cross_act recall@5 +0.056, and 0.000 on lookup, multi_hop and
  temporal alike.** It is half a question, bought with recall@1, MRR and 30 ms,
  and BM25 is still not shipped. But the original claim was stronger than its
  evidence, and the category was too small to have supported it.

The mechanism behind the hypothesis was real. BM25 genuinely retrieves articles
dense retrieval misses, which is why recall@20 improves. It just does not survive
the cut to five, and a cross-encoder recovers the same articles without it.

The result is safe not because any single comparison is decisive at n=51, but
because it is consistent across three independent ones. BM25 remains in the
config surface so the result stays reproducible — not because it earns a place.

### Finding 2: a wider candidate pool made reranking worse

Pool 50 against pool 20, identical in every other respect:

| | recall@5 | recall@20 (ceiling) | latency |
|---|---:|---:|---:|
| pool 50 | 0.804 | 0.931 | 1592 ms |
| pool 20 | **0.824** | 0.902 | **636 ms** |

The wider pool raises the ceiling and the reranker cannot exploit it. Thirty
extra candidates offer more opportunities for a distractor to score into the top
five than for a true positive to be rescued. Pool 50 is strictly dominated: less
accurate at the *k* that matters, and 2.5× slower.

**This was very nearly missed.** `03_hybrid_rerank.yaml` had been written in
Phase 0 with `top_k: 50` while every other config used 20, so the first rerank
result confounded "add a cross-encoder" with "widen the pool". Configs 04 and 05
exist because that confound was spotted while reading the table, not because it
was planned.

### Finding 3: truncation is an accuracy gain, not a speed optimisation

Profiling first, rather than guessing: **reranking was 97.7% of end-to-end
latency** — encode 15 ms, search 0.8 ms, rerank 486 ms median and 1,771 ms p90.
Two causes, both configuration rather than model choice: the cross-encoder loaded
in **fp32** on a GPU built for half precision, and `max_length` defaulted to
**8192**. Attention is quadratic, so one long article dominated whatever batch it
landed in — 2,622 tokens → 389 ms against 4,514 tokens → 1,771 ms. **1.7× the
tokens, 4.5× the time.**

Both were changed at once and the result was faster *and* more accurate — which
is the shape of a confounded result, so it was split into a 2×2:

| max_length | precision | recall@1 | MRR | nDCG@5 | median | p90 |
|---:|---|---:|---:|---:|---:|---:|
| 8192 | fp32 | 0.598 | 0.740 | 0.744 | 538 ms | 1673 ms |
| 8192 | fp16 | 0.598 | 0.740 | 0.744 | 200 ms | 599 ms |
| 512 | fp32 | **0.647** | **0.776** | **0.766** | 624 ms | 794 ms |
| **512** | **fp16** | **0.647** | **0.776** | **0.766** | **193 ms** | **247 ms** |

The separation is complete.

**fp16 costs no accuracy at all** — identical to three decimals in both length
settings. A 2.7× speedup for nothing. Ranking is an ordering problem, and the
score gaps here are far larger than fp16 error.

**Truncation is responsible for the entire accuracy gain**: +4.9 points of
recall@1 and +3.6 of MRR, in both precisions. This is a finding, not an
optimisation. An article's operative statement sits at its opening, and long
enumerations — article 13's list of statutory mentions, article 51's six
categories of insaisissable goods — dilute the relevance signal across text that
does not bear on the question. Capping at 512 tokens acts as a *prior about where
legal meaning lives*.

Combined: **2.8× faster median, 6.8× faster p90, and better ranking.** The
latency gap over the dense baseline falls from 28× to 10×.

Had the two changes not been separated, this would have been written up as "we
made reranking fast and it happened to help", and the more interesting half would
have been invisible.

---

## 6. Generation, and a fine-tune that failed instructively

### The base model already satisfies most of the contract

Qwen3-4B-Instruct-2507 in 4-bit NF4, with the retrieval stack above:

| metric | value | reading |
|---|---:|---|
| answers_citing_unretrieved | **0.000** | nothing invented from parametric memory |
| answers_citing_repealed | **0.000** | no repealed article presented as law |
| answers_with_no_citation | **0.000** | every answer cited something |
| correct_abstention | 0.875 | 7 of 8 unanswerable questions declined |
| false_abstention | 0.039 | 2 of 51 answerable questions wrongly declined |
| citation_recall | 0.765 | of gold articles, how many were cited |

**Citation recall is bounded by retrieval, not by the model.** Retrieval reaches
recall@5 = 0.824; answers reach 0.765. Inspecting the gap shows only **three**
cases in 51 where a gold article sat in context and went uncited. Both false
abstentions have the same cause: retrieval did not supply the governing article,
and the model declined correctly *given what it was shown*.

One metric had to be added mid-phase. `citation_precision` was penalising
*correct* supplementary citations — an answer that cites the governing article
plus a genuinely relevant related one scored worse than a bare answer.
`citation_recall` was introduced as the primary citation metric for that reason.

The real defect is length. The median answer is 101 words and latency tracks
output length almost linearly: 21 words → 6.0 s, 101 words → 26.5 s, 286 words →
66.5 s. At 26 seconds the system is not interactive, and the cause is not
retrieval — reranking is 0.7% of end-to-end time.

### The hypothesis, recorded before training

> The adapter will produce **little or no improvement** on the contract metrics,
> because the base model already satisfies them. It should produce a **large
> reduction in answer length**, and therefore latency. **The risk is that
> conciseness costs citation recall**: an adapter trained to answer in one
> sentence with one citation may stop citing the *second* relevant article.

### What happened

QLoRA r=16 / α=32, 4-bit NF4 with double quantisation, bf16 compute, paged 8-bit
AdamW, gradient checkpointing, on one 8 GiB GPU. Final loss 0.027, token accuracy
98.4%. Retrieval was byte-identical for all 59 questions, so every difference is
the generator.

| metric | base | adapter | |
|---|---:|---:|---|
| correct_abstention | 0.875 | **1.000** | ↑ |
| false_abstention | 0.039 | 0.059 | ↓ |
| **citation_recall** | **0.765** | 0.646 | **↓ −11.9** |
| citation_precision | 0.587 | 0.677 | ↑ |
| fabrication / repealed / uncited | 0.000 | 0.000 | — |
| answer length (median) | 101 w | **33 w** | −67% |
| latency (median) | 26.5 s | **15.5 s** | −41% |
| latency (p90) | 65.9 s | **22.8 s** | −65% |

**All three parts of the hypothesis held.** Contract metrics moved by one
question. Latency fell 41%. Citation recall fell 11.9 points for the predicted
reason: mean citations per answer went from 1.92 to 1.10, so on questions needing
two articles the adapter cites one. Nine questions lost a gold citation the base
model had made.

### What no metric caught

Shorter, faster, cites correctly, never fabricates, abstains better — and the
answers are worse:

```
Q: Quelle est la durée maximale de vie d'une société commerciale ?

base     "La durée maximale … est de quatre-vingt-dix-neuf (99) ans,
          conformément à l'article 28 AUSCGIE (2014)."
adapter  "Toute société à une durée qui doit être mentionnée dans ses
          statuts (article 28 AUSCGIE (2014))."
```

Right article, right citation format, fewer words, **does not answer the
question**. Measuring each answer against the opening sentence of the article it
cites:

| | median similarity | answers >0.6 similar |
|---|---:|---:|
| base | 0.222 | 6 / 47 |
| **adapter** | **0.896** | **38 / 47** |

The adapter learned to emit **the opening sentence of a retrieved article plus a
citation, independent of the question.** On q017 — *"Qu'est-ce qu'une
hypothèque ?"* — it produced the definition of *sûretés personnelles*: the wrong
article's opening, perfectly formatted and cleanly cited.

### Why: the training data taught exactly this

The targets were built as `_first_sentence(article.text) + citation`. The
reasoning was sound in isolation — an article's operative rule sits at its
opening, the *same observation* that made 512-token truncation improve reranking
in §5. As a training target it is a different thing entirely: it teaches that the
correct response to any question is the opening of an article.

The model learned that at 98.4% token accuracy. **It learned the task it was
given. The task was wrong.**

This is a textbook proxy-optimisation failure, and it is the most useful result
in the project:

1. **Every structural metric improved or held while quality collapsed.** A
   dashboard would have reported a successful fine-tune with a 41% latency win.
2. **Only reading outputs revealed it** — the same method that found the
   table-of-contents contamination and the `citation_precision` flaw.
3. **A low training loss measured obedience, not quality.** 0.027 describes how
   well the model matched templated targets and says nothing about whether those
   targets were worth matching.

**Decision: the adapter is not shipped.** The base model with RAG remains the
system.

Two smaller results are worth keeping. **q053 genuinely improved** — the
criminal-penalties question, the hardest abstention case in the set, where the
corpus defines the offence but leaves the penalty to national law, is now
correctly declined. And **q057 regressed, which is what it was reclassified in §4
to catch**: a question fixed because it would have penalised correct behaviour
ended up catching the opposite failure.

### One engineering note worth recording

TRL's `assistant_only_loss` **silently masked nothing**. Qwen3's chat template
carries no `{% generation %}` marker, so the mask returned 0 supervised tokens
out of 984 — with a warning, not an error. Training would have run to completion,
reported a plausible loss, and produced an adapter trained on nothing.

The fix was to switch to prompt/completion format with `completion_only_loss`,
and to add `_verify_loss_masking()`, which inspects a real collated batch before
training starts and raises if the supervised fraction is 0%, 100%, or above 50%.
A silent failure that produces a plausible number is worse than a crash.

---

## 7. The agentic layer: two mechanisms, neither calibrated

An LLM was put in the retrieval loop twice, as separate switches so that "the
agent helped" could be decomposed into which part helped:

```
dense → [decompose] → rerank → [verify]
```

Decomposition splits a question into sub-questions, retrieves for each, and
fuses the rankings with RRF. Verification asks whether the retrieved passages
suffice and searches again with whatever the model says is missing. Both satisfy
the same `Retriever` protocol as every other stage, so the harness scored them
with the code that produced §5.

| metric | rerank-fast | + decompose | + verify |
|---|---:|---:|---:|
| recall@1 | **0.647** | 0.647 | 0.578 |
| recall@5 | **0.824** | 0.824 | 0.814 |
| MRR | **0.776** | 0.776 | 0.717 |
| nDCG@5 | **0.766** | 0.766 | 0.722 |
| latency median | **240 ms** | 655 ms | 1767 ms |
| latency p90 | **312 ms** | 804 ms | **11049 ms** |

**Ship neither.** Decomposition matches the baseline to three decimals on every
metric and every question kind, at 2.7× the latency. Verification is worse than
the baseline everywhere, at 35× the p90.

### Decomposition refuses precisely where it would help

Identical numbers are equally consistent with "the mechanism does not work" and
"the mechanism never ran", so the raw reply was recorded for all 59 questions.

| what the model replied | count |
|---|---:|
| the question is atomic | **47 / 59** |
| a decomposition | 7 |
| unparseable | 5 |

| kind | decomposed |
|---|---:|
| **multi_hop** | **1 / 9** |
| cross_act | 1 / 4 |
| temporal | 1 / 5 |
| lookup | 1 / 33 |
| **unanswerable** | **3 / 8** |

Among the questions it called atomic:

> *Combien de personnes faut-il au minimum pour constituer une société
> coopérative simplifiée, **et** combien pour une société coopérative avec
> conseil d'administration ?*

Explicit conjunction, two distinct points of law, two articles in the gold set —
declared single-article, eight times out of nine. **The one kind it did split is
`unanswerable`**, where the corpus holds no answer and sub-questions retrieve
nothing useful by construction. Its judgement is not weak, it is inverted.

### Verification is never satisfied, and that costs

| | |
|---|---:|
| questions where it asked for more | **55 / 59** |
| top-5 rankings it changed | 14 |
| questions it **improved** | **0** |
| questions it **damaged** | 1 |
| slowest single question | **42 s** |

The predicted signature appeared exactly. From `12_agent_verify.yaml`, before
the run: *a confident follow-up query can displace a true positive out of the
top five; RRF fusion limits the damage but does not prevent it, so a drop on
lookup would be the signature.* Lookup recall@5 held at 0.879 while lookup MRR
fell 0.798 → 0.749 — true positives pushed *down* inside the top five rather
than out of it.

### One reason for both failures

The decomposer is asked *does this need several articles?* and says no 47 times
in 59. The verifier is asked *do these passages suffice?* and says no 55 times
in 59. Neither is calibrated; one almost never acts, the other almost always
does, and only the acting one moves the metrics — downward.

§6 already recorded the limit underneath this. Both false abstentions in the
generation baseline were questions where retrieval had not supplied the
governing article and the model declined correctly: **it knew the context was
insufficient.** What it could not do is name the article that would fix it,
never having seen it. Verification asks for exactly that, so it returns a
plausible legal phrase, and a plausible phrase retrieved and fused is a
distractor.

### A null result that rested on luck

The decomposer never once emitted the token it was asked for. Told in French to
reply `ATOMIQUE`, it produced `ATOMIC`, `ATOMICITY` twice, a sentence of
commentary with `ATOMIQUE` at the end, and once simply `Non`.

All five reached the right outcome, through three *different* accidents: two
were shorter than the minimum sub-question length, one survived as a single line
and was killed by the rule that one part is not a decomposition, and one was
prose. **None was caught by the check written to catch them.** Two lines of
commentary instead of one and the model's commentary about the question would
have been issued as sub-questions.

The check now matches on the stem, per line, for single-word replies, with all
five real strings pinned as tests. Re-running reproduced every number to three
decimals — so the null result stands, but it stood on luck first, and a null
result resting on luck is not a null result.

## 8. Serving, and another negative result

The system is served by FastAPI with server-sent events, a streaming UI, and a
Docker image. The service loads one `ExperimentConfig` and builds the pipeline
with the same `build_retriever` the evaluation harness calls — there is no
serving-only prompt or retrieval setting, because a served system that could
drift from the measured one would make the ablation table describe nothing a
user touches. `/health` reports the run id and index fingerprint rather than
merely "ok", so the chain from output back to system survives the last step.

Generation is serialised behind a lock. One GPU holds one generator, and two
concurrent 4-bit generations on 8 GiB contend for memory that is not there — the
failure mode is an OOM that takes the process down, not a slow response.

| concurrency | first token | total median | total p90 | answers/min | wall |
|---:|---:|---:|---:|---:|---:|
| 1 | **1.1 s** | 20.1 s | 37.9 s | 2.68 | 179 s |
| 2 | 13.2 s | 39.5 s | 68.1 s | 2.83 | 170 s |
| 4 | 40.8 s | 69.5 s | 86.8 s | 2.83 | 169 s |

**The queuing prediction, recorded in `bench.py` before the run, held.**
Throughput is flat to within 6%; latency rises 2.0× and 3.5× for 2× and 4×
concurrency; and the wall clock for the same eight answers is 179 s, 170 s,
169 s. Identical work in identical time however many clients ask at once, which
is what a serialised resource means. Nothing worse than queuing occurs.

**And the headline streaming number is a single-user number.** At one client the
first token arrives 18.9× sooner than the complete answer. At two clients that
is 3.0×, at four it is 1.7× — because under load, time-to-first-token stops
being prefill and becomes *waiting for the lock*.

| concurrency | 1 | 2 | 4 |
|---|---:|---:|---:|
| perceived speedup | **18.9×** | 3.0× | **1.7×** |

Quoting 18.9× alone would have been the serving equivalent of Phase 6's own
version of the §6 mistake: a real measurement, taken under the one condition
that flatters it. The benchmark reports all three rows for that reason.

Two supporting results. `/search` throughput nearly doubles from c=1 to c=2
(1.8 → 3.0 req/s) where generation's does not move at all — that contrast is the
control that makes flat generation throughput attributable to the lock rather
than to a saturated GPU. And retrieval measures **200 ms** over HTTP against the
**193 ms** §5 recorded in-process, so the ablation table's latencies survive
contact with a network client.

## 9. What this project demonstrates

Stated as claims, each with the evidence behind it:

**A reproducible experimental harness.** Content-hashed configs, a frozen
baseline, pre-registered hypotheses, and 12 runs each traceable to the exact
system that produced it.

**That the interesting results here are negative.** BM25 refuted three ways, a
wider pool that hurt, a fine-tune that regressed, two agentic mechanisms that
did nothing and harm respectively, a streaming speedup that held only for one
user. Every one was plausible enough to have been shipped on intuition, and
three of them would have been, on any evidence short of reading the output.

**That the system is served, and served as the thing that was measured.**
FastAPI with SSE, a streaming UI, a Docker image, and an API whose response
carries the same contract the harness scores — including a per-response
fabrication check. `/health` reports the run id, so an output can still be traced
to the system that produced it after it leaves the harness.

**That measurement infrastructure has to be doubted too.** Two metrics were
wrong before the systems they scored were: `citation_precision` penalised correct
behaviour, and the gold set contained articles that were never needed. Both were
found and both changed the reported numbers.

**And that the project's own central guarantee was overstated.** §2 claims a run
traces back to the exact system that produced it. Checking rather than restating
it — while adding a config section in Phase 5 — showed that **9 of 11 recorded
runs no longer re-hash to their own directory, even from their own stored
resolved config.** Phase 3 added two fields to `RerankerConfig`, and because the
fingerprint hashes the fully serialised config, adding a field with a default
re-hashes every config that never mentions it. Every run recorded before that
change was orphaned the moment it landed, silently: no collision, no failed
test, no changed number.

Nothing published is unverifiable — the stored resolved config records what ran,
and is what diagnosed this. What broke is the config-file-to-run direction. The
fix is a test pinning the ids that back published numbers, an exclusion rule so
new subsystems cannot rename old runs, and
[`docs/reproducibility.md`](reproducibility.md) stating the limitation, rather
than a renumbering that would rewrite every citation in these documents to
repair a scheme already broken for nine runs.

That makes four occasions where the instruments were wrong before the systems —
`citation_precision`, the gold set, TRL's silent loss mask, and this. All four
were silent, and all four produced numbers that looked reasonable.

**That silent failures are the dangerous ones.** The loss mask that masked
nothing, the parse defect that both self-checks passed, the adapter that improved
every metric while getting worse. None of these throw.

**Domain reasoning, not just pipeline assembly.** Article-level repealed status
because legal status is not a document property; `[TEXTE ABROGÉ]` pushed into
indexed text so the warning reaches the context window; elision handling and
compound article numbers in French tokenization; 512-token truncation as a prior
about where legal meaning lives.

---

## 10. Limits

Stated because a results table that names its own limits is worth more than one
that does not.

- **n = 51 answerable questions.** Differences of one or two points are noise.
  The BM25 result holds on consistency across three comparisons, not on any
  single one.
- **n = 8 for abstention.** `correct_abstention` moves in steps of 0.125. No
  fine-grained claim about abstention is supportable at this size.
- **`cross_act` is four questions and never moved** across any of the seven
  systems — 0.375 in six of them. That is a count, not a measurement.
- **`answers_citing_repealed` is near-vacuous.** The corpus holds three repealed
  articles and none of the seven superseded acts, so 0.000 means "the situation
  rarely arose", not "the system is safe".
- **No judgement of answer correctness.** Every metric here is structural —
  citations, abstention, length. **Whether the French prose states the law
  accurately is unmeasured**, and would need an LLM judge or a lawyer.
- **One embedding model.** Whether `multilingual-e5-base` suits French legal text
  is untested; only its combination with a reranker has been measured.
- **One corpus, one language.** Nothing here establishes that these results
  transfer to another legal system or another language.
- **SYCEBNL-2022 is a scan** and was refused rather than parsed. Seven repealed
  acts remain unsourced.

## 11. What would come next

**Fix `cross_act` properly.** Rebuild it from real inter-act textual
cross-references, which is the only way to make the weakest row in the table
measurable.

**Source the repealed acts**, so `answers_citing_repealed` becomes a real test
rather than a wired-up metric that never fires.

**Measure whether the law is stated correctly.** The largest gap in the
evaluation. Needs an LLM judge, validated against human review on a subset.

**Fine-tuning, done properly.** The adapter is not evidence that fine-tuning
cannot help here; it is evidence that *these targets* do not. A serious attempt
needs answer targets that are genuine answers rather than article summaries —
which means human writing or a stronger teacher, since the only local model is
the one being trained. That is a real cost, and worth stating rather than
substituting a cheaper proxy and reporting the latency win.

---

## Appendix: reproducing any number

```bash
.venv\Scripts\kora.exe eval run -c configs/experiments/07_rerank_fast.yaml
```

Each config in `configs/experiments/` is one row of §5. `--generate` adds the
answer metrics of §6. Results land in `runs/<name>-<fingerprint>/` with the
resolved config beside them.

| document | phase |
|---|---|
| [`parse-quality.md`](parse-quality.md) | corpus and parser defects (§3) |
| [`validating-the-gold-set.md`](validating-the-gold-set.md) | review protocol (§4) |
| [`baseline-results.md`](baseline-results.md) | frozen baseline (§5) |
| [`ablations.md`](ablations.md) | retrieval ablations (§5) |
| [`generation-baseline.md`](generation-baseline.md) | generation + hypothesis (§6) |
| [`fine-tuning-results.md`](fine-tuning-results.md) | the adapter regression (§6) |
| [`agentic.md`](agentic.md) | decomposition and self-correction (§7) |
| [`serving.md`](serving.md) | API, streaming, Docker, benchmarks (§8) |
| [`reproducibility.md`](reproducibility.md) | where content-hashing broke (§9) |
