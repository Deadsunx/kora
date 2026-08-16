# QLoRA fine-tuning: the adapter is a regression

Base `rerank-fast-54cefb1da669` · adapter `adapter-r16-6c121c457d35` · identical
retrieval, verified per question · 59 validated questions, 51 answerable

**Conclusion: do not ship the adapter.** Every structural metric held or
improved and the answers got worse. The base model with RAG remains the system.

## The metrics

| metric | base | adapter | change |
|---|---:|---:|---|
| correct_abstention | 0.875 | **1.000** | +0.125 (7/8 → 8/8) |
| false_abstention | 0.039 | 0.059 | +0.020 (2 → 3) |
| **citation_recall** | **0.765** | **0.646** | **−0.119** |
| citation_precision | 0.587 | 0.677 | +0.090 |
| answers_with_no_citation | 0.000 | 0.000 | — |
| answers_citing_unretrieved | 0.000 | 0.000 | — |
| answers_citing_repealed | 0.000 | 0.000 | — |
| answer length (median) | 101 words | **33 words** | −67% |
| latency (median) | 26.5 s | **15.5 s** | −41% |
| latency (p90) | 65.9 s | **22.8 s** | −65% |

Retrieval was byte-identical for all 59 questions, so every difference is the
generator.

## The hypothesis was right, in all three parts

Recorded in `docs/generation-baseline.md` *before* training:

> little or no gain on the contract metrics · a large reduction in answer length
> and therefore latency · **and a risk that conciseness costs citation recall**

All three held. Contract metrics moved by one question. Latency fell 41%.
Citation recall fell 11.9 points, for the predicted reason: mean citations per
answer went from 1.92 to 1.10, so on questions needing two articles the adapter
cites one. Nine questions lost a gold citation the base model had made.

## What no metric caught

The metrics say: shorter, faster, cites correctly, never fabricates, abstains
better. The answers say otherwise.

```
Q: Quelle est la durée maximale de vie d'une société commerciale ?

base     "La durée maximale ... est de quatre-vingt-dix-neuf (99) ans,
          conformément à l'article 28 AUSCGIE (2014)."
adapter  "Toute société à une durée qui doit être mentionnée dans ses
          statuts (article 28 AUSCGIE (2014))."
```

The adapter cited the right article, in the right format, in fewer words — and
did not answer the question. Measuring the similarity between each answer and
the opening sentence of the article it cites:

| | median similarity | answers >0.6 similar |
|---|---:|---:|
| base | 0.222 | 6 / 47 |
| **adapter** | **0.896** | **38 / 47** |

The adapter learned to emit **the opening sentence of a retrieved article plus a
citation**, independent of what was asked. On q017 — *"Qu'est-ce qu'une
hypothèque ?"* — it produced the definition of *sûretés personnelles* instead:
the wrong article's opening, perfectly formatted and cleanly cited.

## Why: the training data taught exactly this

The targets were built as `_first_sentence(article.text) + citation`. The
reasoning was sound in isolation — an article's operative rule sits at its
opening, the same observation that made 512-token truncation improve reranking
in Phase 3. As a *training target* it is a different thing entirely: it teaches
the model that the correct response to any question is the opening of an
article.

The model learned that with a token accuracy of 98.4% and a final loss of 0.027.
It learned the task it was given. The task was wrong.

## What this demonstrates

This is a proxy-optimisation failure, and it is the most useful result in the
project:

1. **Every structural metric improved or held while quality fell.** Citation
   format, fabrication rate, abstention, length and latency all moved the right
   way. A dashboard would have reported a successful fine-tune.
2. **Only reading the outputs revealed it.** The same method that found the
   table-of-contents contamination in Phase 1 and the citation-precision flaw in
   Phase 4a.
3. **A low training loss measured obedience, not quality.** 0.027 loss and 98.4%
   token accuracy describe how well the model matched templated targets, and say
   nothing about whether those targets were worth matching.

Two smaller results are worth keeping:

- **q053 improved.** The criminal-penalties question — the hardest abstention
  case in the set, where the corpus defines the offence but leaves the penalty
  to national law — is now correctly declined. That is a real gain.
- **q057 regressed, and was designed to catch this.** The Cameroon
  minimum-capital question was reclassified from `unanswerable` to `lookup`
  during gold-set validation precisely because the OHADA rule *is* in the
  corpus. The adapter now over-abstains on it. A question fixed because it would
  have penalised correct behaviour ended up catching the opposite failure.

## What would be needed to do this properly

The adapter is not evidence that fine-tuning cannot help here; it is evidence
that these training targets do not. A serious attempt would need answer targets
that are genuine answers rather than article summaries — which means either
human-written responses, or a stronger teacher model, since the only local model
is the one being trained. That is a real cost, and worth stating rather than
substituting a cheaper proxy and reporting the latency win.
