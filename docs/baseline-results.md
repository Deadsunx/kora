## Baseline retrieval results

Run `baseline-dense-8d233173f464` · 2026-08-16 · corpus 3,056 articles ·
59 gold questions, all human-validated · 51 of them answerable

Configuration: structure-aware chunking, `intfloat/multilingual-e5-base`, dense
retrieval only. No BM25, no reranking, no fine-tuning. This is the number every
later change must beat.

### Headline

| metric | value |
|---|---:|
| recall@1 | 0.490 |
| recall@5 | **0.735** |
| recall@10 | 0.814 |
| recall@20 | 0.873 |
| MRR | 0.647 |
| nDCG@5 | 0.647 |
| precision@5 | 0.176 |
| search latency (median) | 0.32 ms |
| search latency (p90) | 0.59 ms |

n = 51. Unanswerable questions are excluded: recall against an empty gold set is
undefined, and scoring them as failures would penalise a set for containing
honest questions. They are measured by abstention once generation exists.

The validated and diagnostic columns are now identical, because every question
is validated. That is expected, not a bug — the split exists to stop unverified
questions leaking into a reported number, and there are none left.

### By question kind

| kind | n | recall@5 | MRR |
|---|---:|---:|---:|
| temporal | 5 | 0.800 | 0.625 |
| lookup | 33 | 0.788 | 0.682 |
| multi_hop | 9 | 0.667 | 0.711 |
| cross_act | 4 | 0.375 | 0.250 |

**What can be claimed.** Single-article lookups retrieve at 0.788. Questions
needing two articles retrieve at 0.667 — a real but modest gap, and the expected
one: recall requires finding *every* gold article, so it falls as the number
required rises. MRR moving the other way (0.711 for multi_hop against 0.682 for
lookup) is consistent, not contradictory: finding *one* relevant article early is
easier when several qualify.

**What cannot be claimed.** The cross_act figure of 0.375 rests on **four**
questions. Each is worth 0.25 of the score, so the metric is effectively a count,
and no comparison built on it is meaningful. It is recorded, not reported.

### Effect of validating the gold set

An earlier run over the unvalidated set gave recall@5 = 0.679 against 0.735 here,
and a cross_act figure of 0.292 that was quoted before the labels were checked.

The improvement is not a change to the system — the index and retriever are
identical. It is the removal of gold articles that were never required to answer
their question. Retrieval was being marked down for failing to find passages that
did not need finding.

That is worth stating plainly, because it cuts both ways: **the same review that
raised the headline number destroyed the one finding the earlier run appeared to
support.** The 0.292 cross-act collapse was largely an artefact of eight
questions, three of which were not cross-act at all and one of which was
malformed.

### Known limits

- `questions_with_repealed_in_top5` is 0.000 and near-vacuous. The corpus holds
  three individually repealed articles and none of the seven superseded acts, so
  the distractor set barely exists. The metric is wired and tested; it is not yet
  exercised.
- No generation yet, so nothing here measures answer quality, citation accuracy
  or abstention. These are retrieval numbers only.
- One embedding model, untested against alternatives. Whether
  `multilingual-e5-base` is a good choice for French legal text is an open
  question, not an established one.
