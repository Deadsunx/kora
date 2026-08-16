# The content-hash identity scheme, and how it broke

Found in Phase 5, while adding a config section for the agent. It is a defect in
the project's own infrastructure, of the same kind as the two the evaluation
harness produced — and like those, it was found by checking rather than by
anything failing.

## The claim

From the README, and repeated in the report:

> A run's identity is a hash of the settings that produced it, so two
> experiments can never overwrite each other's results, and a results table
> months old can be traced back to the exact system that produced it.

The first half is true. The second half is **false for 9 of 11 recorded runs.**

## The measurement

Two checks. First, does any config file still hash to each recorded run
directory?

| recorded run | config that produces it |
|---|---|
| `rerank-fast-54cefb1da669` | `07_rerank_fast.yaml` |
| `adapter-r16-6c121c457d35` | `10_adapter.yaml` |
| the other nine | **none** |

Second — and this is the one that matters — does each run's own stored
`config.resolved.yaml` still hash to the directory it sits in?

| | |
|---|---|
| re-hashes to its own directory name | **2 of 11** |
| drifts | **9 of 11** |

`baseline-dense-8d233173f464` re-hashes to `baseline-dense-08e4cf24afc3`.
`lexical-only-23900ba6af7e` to `lexical-only-be681bfdf078`. And so on.

## The cause

Phase 3 added `max_length` and `precision` to `RerankerConfig` and changed
`batch_size` from 16 to 32. The fingerprint is a hash of the full serialised
config, so **adding a field with a default changes the hash of every config that
never mentions it.** Every run recorded before that change was orphaned at the
moment the field landed.

Nothing failed. No test broke, no directory collided, no number changed. The two
runs that still resolve are simply the two recorded after the change.

## What was not damaged

Every published number remains verifiable. `config.resolved.yaml` is written
into each run directory at write time and records the complete system, field by
field, as it was — the old reranker blocks genuinely have no `max_length` key,
which is what shows the schema moved underneath them. Reading it tells you
exactly what produced the numbers.

So the audit direction — *run → system* — works, and is how the drift was
diagnosed in the first place. What broke is *config file → run*: you can no
longer re-run `05_dense_rerank_k20.yaml` and land on the directory it made.

One consequence worth stating plainly: **`05_dense_rerank_k20.yaml` and
`07_rerank_fast.yaml` are now the same system.** Config 05 said "reranker
defaults" and the defaults became fp16/512 in Phase 3. Row 6 of the ablation
table was measured under the old defaults, which is why its 530 ms matches the
2×2's fp32/8192 row at 538 ms rather than row 7's 193 ms. The table is coherent;
the config file that produced row 6 no longer exists in a form that reproduces
it.

## The fix that should have been there

`fingerprint()` should have serialised with `exclude_defaults=True` from the
start. Then a config's identity depends only on what it actually *says*, and
adding a field is hash-neutral for every config that does not set it — which is
the property the scheme needed and did not have.

Adopting it now would re-hash the two runs that still resolve, renaming
`rerank-fast-54cefb1da669` and `adapter-r16-6c121c457d35` — the ids cited
throughout `ablations.md`, `generation-baseline.md`, `fine-tuning-results.md`
and the report. That trades a real, immediate cost for a benefit that only
arrives at the next schema change, on a project that has reached its last phase.

So the decision is: **do not renumber.** Instead,

1. **New optional subsystems are excluded from the fingerprint when disabled.**
   `AgentConfig` is the first, and the exclusion is written into
   `fingerprint()` with its reasoning. This gives new components the
   `exclude_defaults` property without touching existing identities — adding the
   agent renamed nothing, which was verified by computing every run id before
   and after the change.
2. **The ids that back published numbers are pinned by a test.**
   `test_published_run_ids_are_stable` fails loudly if a schema change renames
   `rerank-fast` or `adapter-r16`. The next occurrence will not be silent.
3. **This document exists**, because a broken guarantee that is written down is
   a different thing from one that is not.

## What this is an instance of

The fourth time in this project that the measurement infrastructure was wrong
before the system it measured:

- `citation_precision` penalised correct supplementary citations (Phase 4a)
- the gold set contained articles never needed to answer their question (Phase 2)
- TRL's `assistant_only_loss` masked nothing and warned instead of raising (Phase 4b)
- and this

All four were silent. None threw, none failed a test, and each produced numbers
that looked entirely reasonable. That is the argument for checking the
instruments as deliberately as the results — and for the habit of testing a
claim rather than restating it, which is what turned this one up.
