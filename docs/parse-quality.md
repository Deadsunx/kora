# Extraction quality

Date: 2026-08-16 · parser version 1.0 · 12 of 18 documents acquired · **3,056 articles**

Reference counts are from [`Maathis-com/ohada-actes-uniformes`](https://huggingface.co/datasets/Maathis-com/ohada-actes-uniformes),
an independently published corpus built from the same government sources. It is
used as an external check, not as ground truth — neither corpus is
authoritative, and where they disagree the disagreement is the finding.

## Results

| Document | Pages | Articles | Gaps | Coverage | Reference | Δ |
|---|---:|---:|---:|---:|---:|---:|
| AUSCOOP-2010 | 79 | 397 | 0 | 98.0% | 397 | **0** |
| AUDCG-2010 | 67 | 307 | 0 | 98.5% | 307 | **0** |
| AUS-2010 | 45 | 228 | 0 | 98.1% | 228 | **0** |
| AUA-2017 | 13 | 38 | 0 | 96.5% | 38 | **0** |
| AUCTMR-2003 | 14 | 31 | 0 | 98.3% | 31 | **0** |
| AUDCIF-2017 | 40 | 123 | 0 | 98.6% | 120 | +3 |
| AUPCAP-2015 | 122 | 377 | 0 | 99.3% | 371 | +6 |
| AUM-2017 | 9 | 18 | 0 | 95.5% | *absent* | — |
| **AUPSRVE-2023** | 113 | **448** | 0 | 98.3% | 242 | **+206** |
| **AUSCGIE-2014** | 217 | **1089** | 0 | 98.5% | 811 † | **+278** |
| SYCEBNL-2022 | 438 | — | — | — | *absent* | scan |

† The reference's AUSCGIE table has 1,392 **rows** but only **811 distinct**
article numbers — 549 labels appear more than once. See below.

**Zero numbering gaps across every parsed document.** Coverage 95.5–99.3%.

## The headline: AUPSRVE-2023

The most-litigated act in OHADA law — origin of over 70% of cases before the
CCJA — and the one the reference corpus extracts worst, at 242 articles against
roughly 335 official, having lost article boundaries to a two-column layout.

We extract **448 articles with zero numbering gaps and 98.4% coverage**, from a
Laws.Africa-typeset PDF with a real text layer rather than a scan.

Two caveats stated plainly. First, 448 exceeds the ~335 usually quoted, because
the 2023 revision inserted a large number of sub-articles (`1-10`, `28-1`,
`152-2`, `245-6`) that a headline figure may not count. Second, the two corpora
are working from different source PDFs. The comparison is meaningful, but it is
a comparison of extractions, not a scoreboard.

## Five parser defects found, and how

Each was found by a *different* mechanism — the numbering invariant, the length
distribution, a unit test, a manual spot-check, and building the gold question
set. None of them caught what another caught. That is the whole argument for
having several, and for reading output by hand as well as by metric.

**1. Table-of-contents contamination — found by the length distribution.**

A contents page repeats every article heading followed by dot leaders:

```
Article 2
..................................................... 14
```

The heading line matches the article pattern exactly. Left alone it creates a
phantom article whose body is a row of dots and — worse — claims the number, so
the genuine article forty pages later is dismissed as a duplicate
cross-reference and its text appended to whatever article is open.

**Both existing checks passed on this.** Numbering stayed contiguous, because the
phantoms covered every number. Coverage stayed at 98.7%, because the text was
still present, merely misfiled. It surfaced only on inspecting article lengths:
191 articles of pure dots, and three blocks of 19,000–31,000 characters where
real articles run to a few thousand.

Fixes: drop dot-leader lines, and claim an article number only when an article
is actually emitted. `longest_article_chars` was added as a third check.

The same fix also recovered text in AUSCGIE-2014, whose own `SOMMAIRE` was
mildly contaminating it — a document previously reported as clean.

**2. `Article premier` — found by the numbering invariant.**

French legal drafting numbers the first article in words. AUPSRVE-2023 carries
the convention into its inserted articles (`Article premier-11`) while
cross-referencing the same provisions numerically in the body (`l'article 1-10`).

A digits-only pattern does not merely miss article 1: every `premier-N` heading
goes unrecognised and its text is absorbed into the preceding article. The gap
check reported exactly one missing number — article 1 — which was the visible
tip of 17 lost articles. Fixing it took AUPSRVE from 431 articles at 92.7%
coverage to 448 at 98.4%.

**3. Repealed articles destroyed by the rubric heuristic — found while building
the gold question set.**

A repealed article is printed as its heading followed by the single word
`Abrogé`:

```
Article 12
Abrogé
Article 13
Les petites entités sont assujetties, sauf option, au Système minimal…
```

That word is short, carries no terminal full stop and is not a structural
marker — so the rubric heuristic classified it as a marginal title belonging to
the *next* heading, stole it from article 12, and attached it to article 13.
Two errors from one heuristic: **article 12 lost its only content and was left
empty, while article 13 — perfectly in force — was made to look repealed.**

Three articles were affected in AUDCIF (12, 27, 60), and they were visible the
whole time as the three zero-length bodies in the length distribution. They were
noticed but not chased.

The fix is narrow: a candidate rubric is only reclaimed from the previous
article when that article survives losing it. If removing it would leave an
empty body, it was content, not a title.

This also produced a wrong claim in an earlier version of this document, which
named articles 13, 28 and 61 as the repealed ones. That was the corrupted parser
output being read as ground truth. The repealed articles are **12, 27 and 60**.

**Article-level legal status.** The same fix added a `repealed` flag on
`Article`. Status is not only a property of documents: AUDCIF-2017 is in force
while three of its articles are not, so reading status from the manifest alone
would score a citation of article 12 as perfectly safe. Repealed articles now
carry an explicit `[TEXTE ABROGÉ]` marker into their indexed text, so the
warning reaches the generator's context window rather than living only in
metadata.

**4. Short-page furniture detection — found by a unit test.**

Header/footer detection sampled the first and last three lines of each page,
which on a short page is every line, letting body text be deleted as furniture.
Real pages are long enough that it never triggered in production. The sampled
edge now scales with page length.

## Remaining scan

**SYCEBNL-2022** — 438 pages, 66,126 characters (151 per page). Refused rather
than parsed. It is also the least valuable document here for legal QA: an
accounting-standards manual, largely tables and charts of accounts rather than
numbered provisions.

## Method note

Three independent checks, because each catches what the others cannot:

- `numbering_gaps()` — legal texts number articles contiguously, so a gap means
  a heading was swallowed. Caught the `premier` convention.
- `text_coverage()` — a parser can produce perfect contiguous numbering while
  discarding half of every body.
- `longest_article_chars` — catches text merged into the wrong article, which
  leaves both of the above looking healthy.

Coverage should not reach 100%: cover pages, structural headings and signature
blocks are legitimately not article text. Contents pages are excluded from both
sides of the ratio, since counting them only in the denominator reports a
collapse where nothing was lost.

## Resolved: the AUSCGIE-2014 disagreement

Previously recorded as an unexplained −303. Settled by comparing article
identifiers rather than counts, against `nodes/articles.csv` in the reference
repository.

| | Reference | Ours |
|---|---:|---:|
| Rows | 1,392 | 1,089 |
| **Distinct article numbers** | **811** | **1,089** |
| Duplicated labels | **549** | 0 |
| Articles absent from the other set | **0** | **278** |

The reference's headline figure of 1,392 counts rows, and 549 article numbers
appear more than once in its table. Its distinct article count is 811.

Our extraction is a **strict superset**: every article the reference holds is
present in ours, and 278 more besides. Spot-checking three of the 278 confirms
they are genuine provisions of the act — article 4 (*"La société commerciale est
créée par deux (2) ou plusieurs personnes…"*), article 8 (capacity of minors),
article 9 (spouses as co-associates).

The −303 was an artefact of comparing our distinct-article count against their
row count. Stated plainly because the earlier version of this document
speculated about "different counting units" and that speculation was wrong: the
difference is duplication on their side and coverage on ours, not a difference
in what counts as an article.

**A defect of ours found by the same check.** Spot-checking article bodies showed
one beginning `"page 3 / 217"`. Page numbers differ on every page, so
frequency-based furniture detection cannot see them; they are now removed by
pattern. This is the fourth defect found by a mechanism other than the one it
was looking for, which is the argument for checking output by inspection as well
as by metric.

**A note on the reference's usability.** It cannot be loaded through the
HuggingFace datasets API at all: its CSVs carry mismatched schemas within one
config (`article_id`/`acte_id` in some files, `source_acte`/`target_acte` in
others), so the loader fails with `DatasetGenerationCastError`. The raw CSVs
must be read directly.

None of this diminishes the reference's value here. It caught nothing of ours
that was wrong, but the five exact agreements it produced are what made the two
disagreements interpretable at all.

## Open items

1. SYCEBNL-2022 needs a text source or OCR. If OCR, it must be tracked as a
   distinct quality tier — OCR error rates on legal French would otherwise
   confound every later measurement.
2. The 7 repealed acts remain unsourced; the mirror 403s on all of them.
3. Re-run the article-list comparison for AUPCAP (+6) and AUDCIF (+3) to confirm
   those are the same superset relationship rather than a different phenomenon.
