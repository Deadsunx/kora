# Generation baseline, and a pre-registered hypothesis for fine-tuning

Run `rerank-fast-54cefb1da669` · Qwen3-4B-Instruct-2507, 4-bit NF4, no adapter ·
dense + reranking (fp16, 512 tokens) · 59 validated questions, 51 answerable

## Results

| metric | value | reading |
|---|---:|---|
| correct_abstention | 0.875 | 7 of 8 unanswerable questions correctly declined |
| false_abstention | 0.039 | 2 of 51 answerable questions wrongly declined |
| citation_recall | 0.765 | of gold articles, how many the answer cited |
| citation_precision | 0.587 | lowered by *correct* supplementary citations |
| answers_with_no_citation | **0.000** | every answer cited something |
| answers_citing_unretrieved | **0.000** | nothing invented from parametric memory |
| answers_citing_repealed | **0.000** | no repealed article presented as law |
| answer latency | 26.5 s median, 65.9 s p90 | |

## The base model already satisfies most of the contract

This is the finding, and it is inconvenient for the phase that follows.

The adapter planned in Phase 4 was designed to teach three behaviours. On three
of them the base model is already at or near ceiling:

- **It never cites an article it was not given.** 0.000, across 59 questions.
  Fabrication from parametric memory — the failure a grounded system exists to
  prevent — does not occur.
- **It never presents repealed text as current law.** 0.000.
- **It always cites something.** 0.000 uncited answers.

Abstention is close behind: 7 of 8 unanswerable questions declined, in the exact
sentinel format, with a specific explanation of what was missing.

**Citation recall is bounded by retrieval, not by the model.** Retrieval reaches
recall@5 = 0.824; the answers reach 0.765. Inspecting the gap shows only **three**
cases in 51 where a gold article sat in the model's context and went uncited
(q015, q020, q048). The model cites essentially everything it is given, and the
remaining loss is retrieval never surfacing the article at all.

Both false abstentions have the same cause. q045 (médiation versus arbitrage)
and q046 (sûreté and saisie-attribution) are refusals where retrieval did not
supply the governing articles — the model declined correctly *given what it
was shown*.

## What is actually wrong: length

| answer length | latency |
|---:|---:|
| 21 words | 6.0 s |
| 101 words (median) | 26.5 s |
| 286 words | 66.5 s |

Latency tracks output length almost linearly, and the median answer is 101
words. The model over-explains: it hedges across three articles, restates the
question, and on the longest questions runs into the 512-token cap mid-sentence.

At 26 seconds per answer the system is not usable interactively, and the cause
is not the retrieval stack — reranking was brought down to 193 ms in Phase 3 and
is now 0.7% of end-to-end time.

## Hypothesis, recorded before training

Written down first, for the same reason the BM25 hypothesis was written into
`02_hybrid.yaml` before it was measured — and that one turned out to be wrong.

> The adapter will produce **little or no improvement** on the contract metrics,
> because the base model already satisfies them: three are at 0.000/1.000 and
> abstention is at 7/8, where a single question is worth 12.5 points.
>
> It should produce a **large reduction in answer length**, and therefore in
> latency, because the training targets are concise single-sentence answers with
> one citation.
>
> The risk is that conciseness costs citation recall: an adapter trained to
> answer in one sentence with one citation may stop citing the *second* relevant
> article, which is exactly the behaviour q003 and q004 showed and which
> `citation_recall` is designed to catch.

If that is what happens, the honest write-up is "fine-tuning bought latency at a
small cost in coverage, on a contract the base model already satisfied" — not
"fine-tuning improved the system".

## Limits

- **n = 8 for abstention.** `correct_abstention` moves in steps of 0.125. No
  fine-grained claim about abstention is supportable at this size.
- **`answers_citing_repealed` remains near-vacuous.** The corpus holds three
  individually repealed articles and none of the seven superseded acts, so the
  distractor set barely exists. 0.000 here means "the situation rarely arose",
  not "the system is safe".
- **No judgement of answer correctness.** Every metric here is structural —
  citations, abstention, length. Whether the French prose states the law
  accurately is unmeasured, and would need either an LLM judge or human review.
