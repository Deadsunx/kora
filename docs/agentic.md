# The agentic layer: two mechanisms, neither calibrated

`11_agent_decompose` · `12_agent_verify` · against the frozen best system
`07_rerank_fast` · 59 validated questions, 51 answerable · one RTX 4070 Laptop

**Conclusion: ship neither.** Decomposition is an exact no-op at 2.7× the
latency. Verification is actively harmful at 7.4× median and 35× p90.

## Results

| metric | rerank-fast | + decompose | + verify |
|---|---:|---:|---:|
| recall@1 | **0.647** | 0.647 | 0.578 |
| recall@5 | **0.824** | 0.824 | 0.814 |
| MRR | **0.776** | 0.776 | 0.717 |
| nDCG@5 | **0.766** | 0.766 | 0.722 |
| latency median | **240 ms** | 655 ms | 1767 ms |
| latency p90 | **312 ms** | 804 ms | **11049 ms** |

By question kind, recall@5 and MRR:

| kind | n | rerank-fast | + decompose | + verify |
|---|---:|---:|---:|---:|
| lookup | 33 | 0.879 / 0.798 | 0.879 / 0.798 | 0.879 / **0.749** |
| multi_hop | 9 | 0.722 / 0.794 | 0.722 / 0.794 | 0.722 / 0.794 |
| cross_act | 4 | 0.375 / 0.460 | 0.375 / 0.460 | **0.250 / 0.242** |
| temporal | 5 | 1.000 / 0.850 | 1.000 / 0.850 | 1.000 / **0.750** |

Decomposition matches the baseline to three decimals on every metric and every
kind. That is not a small effect; it is no effect.

## Decomposition: it refuses exactly where it would help

The metrics cannot explain themselves — identical numbers are equally
consistent with "the mechanism does not work" and "the mechanism never ran". So
the decomposer's raw reply was recorded for all 59 questions.

| what the model replied | count |
|---|---:|
| the question is atomic | **47 / 59** |
| a decomposition | 7 |
| unparseable | 5 |

And by kind, the questions it agreed to split:

| kind | decomposed |
|---|---:|
| **multi_hop** | **1 / 9** |
| cross_act | 1 / 4 |
| temporal | 1 / 5 |
| lookup | 1 / 33 |
| **unanswerable** | **3 / 8** |

Questions it called atomic:

> *Combien de personnes faut-il au minimum pour constituer une société
> coopérative simplifiée, **et** combien pour une société coopérative avec
> conseil d'administration ?*

> *Quel est le délai de prescription des obligations commerciales, **et**
> s'applique-t-il aussi aux entreprenants ?*

Explicit conjunction, two distinct points of law, two articles in the gold set.
Declared single-article, eight times out of nine.

**The one category it did split is `unanswerable`** — 3 of 8, the highest rate
of any kind — where the corpus contains no answer at all and sub-questions
retrieve nothing useful by construction. The decomposer splits when splitting is
useless and refuses when it would help. Its judgement is not merely weak; on
this set it is inverted.

Retrieval was byte-identical for 54 of 59 questions. Of the 7 where the ranking
moved, **the number of gold articles in the top 5 changed on none.**

## Verification: it is never satisfied, and that costs

The mirror image. Verification asks whether the retrieved passages suffice, and
searches again with whatever the model says is missing.

| | |
|---|---:|
| questions where it asked for more | **55 / 59** |
| top-5 rankings it changed | 14 |
| questions it **improved** | **0** |
| questions it **damaged** | 1 |
| slowest single question | **42 s** |

It says "sufficient" four times in fifty-nine. On the other 55 it produces a
follow-up query, pays a generation and a full rerank for it, and fuses the
result — improving nothing, losing `AUM-2017#art1` from q045 outright, and
reshuffling thirteen more rankings badly enough to cost 0.059 of MRR.

**The predicted signature is exactly what appeared.** From
`12_agent_verify.yaml`, written before the run:

> The failure mode to watch for is worse than doing nothing: a confident
> follow-up query on a question that was already answered correctly can displace
> a true positive out of the top five. RRF fusion with the original ranking
> limits the damage but does not prevent it, so a **drop on lookup** would be
> the signature.

Lookup recall@5 held at 0.879 while lookup MRR fell 0.798 → 0.749. RRF did limit
the damage — true positives were pushed *down* inside the top five rather than
out of it — and it did not prevent it. Temporal shows the same shape: recall
still 1.000, MRR down 0.100.

## Why both fail: the model cannot judge its own retrieval

The two mechanisms fail in opposite directions and for one reason.

- The decomposer is asked *does this question need several articles?* and
  answers **no** 47 times in 59.
- The verifier is asked *do these passages suffice?* and answers **no** 55 times
  in 59.

Neither is calibrated. One almost never acts, the other almost always acts, and
the metrics move only where the acting one does harm.

Phase 4 already recorded the underlying limit. Both false abstentions in the
generation baseline were questions where retrieval had not supplied the
governing article, and the model declined correctly — **it knew the context was
insufficient.** What it could not do is say which article would fix it, having
never seen it. Verification asks precisely for that missing thing, so it
produces a plausible legal phrase rather than the missing article, and a
plausible phrase retrieved and fused is a distractor.

The generation baseline also explains why there is no headroom for the
decomposer to find. Of 51 answerable questions, only **three** had a gold
article sitting in context and left it uncited. The gap between recall@5 = 0.824
and citation recall 0.765 is retrieval never surfacing the article, and a
question the retriever cannot find with the user's own words is not obviously
one it finds with the model's paraphrase of them.

## What was nearly a fake result

The decomposer never once emitted the token it was asked for. Told, in French,
to reply `ATOMIQUE`, it produced `ATOMIC`, `ATOMICITY` twice, one sentence of
commentary with `ATOMIQUE` at the end, and once simply `Non`.

All five reached the correct outcome — and through three *different* accidents:
two replies were shorter than the minimum sub-question length, one survived as a
single line and was killed by the rule that one part is not a decomposition, and
one was prose. **None was caught by the check written to catch it.** Had the
model written two lines of commentary rather than one, its commentary about the
question would have been issued as sub-questions and retrieved.

The check now matches on the stem, per line, for single-word replies, and all
five real strings are pinned as tests. Re-running the experiment with the
corrected parser reproduced every number to three decimals, so the null result
stands — but it stood on luck before, and a null result resting on luck is not a
null result.

## Limits

- **n = 9 for multi_hop**, the population the decomposer targets. The
  conclusion drawn here is about what the *model* did — 8 of 9 declared atomic,
  which is a property of the replies and not of the metric — rather than about
  an effect size the sample could not support anyway.
- **n = 4 for cross_act.** Verification's loss there is one question. It is
  consistent with the MRR and nDCG drops across every other kind, but on its own
  it is a count.
- **One prompt each.** A better decomposition prompt, few-shot examples, or a
  larger model might well change the firing rate. What is measured is this
  prompt on this model; the finding is that a 4B model given a clear instruction
  answered "no" to both questions almost always.
- **Retrieval only.** Neither configuration was run with generation. There was
  no reason to spend 20 s a question on top: decomposition changes nothing to
  feed the generator, and verification feeds it strictly worse passages.
