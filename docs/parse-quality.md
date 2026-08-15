# Extraction quality — first corpus-wide run

Date: 2026-08-16 · parser version 1.0 · 9 of 18 documents acquired

## Results

Reference counts are from [`Maathis-com/ohada-actes-uniformes`](https://huggingface.co/datasets/Maathis-com/ohada-actes-uniformes),
an independently published corpus built from the same government sources. It is
used here as an external check, not as ground truth — neither corpus is
authoritative, and where they disagree the disagreement is the finding.

| Document | Pages | Articles | Gaps | Coverage | Reference | Δ | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| AUSCOOP-2010 | 94 | 397 | 0 | 96.9% | 397 | **0** | agree exactly |
| AUDCG-2010 | 78 | 307 | 0 | 96.2% | 307 | **0** | agree exactly |
| AUCTMR-2003 | 18 | 31 | 0 | 96.1% | 31 | **0** | agree exactly |
| AUSCGIE-2014 | 209 | 1089 | 0 | 98.6% | 1392 | −303 | **unexplained** |
| AUPCAP-2015 | 132 | 370 | **6** | 96.0% | 371 | −1 | our defect |
| AUPSRVE-2023 | 40 | — | — | — | 242 | — | scan, 0 chars |
| AUA-2017 | 14 | — | — | — | 38 | — | scan |
| AUM-2017 | 14 | — | — | — | — | — | scan (same file as AUA) |
| AUDCIF-2017 | 54 | — | — | — | 120 | — | scan |

## Reading the table

**Three exact agreements.** AUSCOOP, AUDCG and AUCTMR match the reference count
to the article, on documents of very different sizes (397, 307, 31). Two
independent parsers, written from different sources, converging exactly is the
strongest evidence available that both are extracting correctly. This is what
licenses any claim made about the two disagreements below.

**AUPCAP-2015 — six articles we lose.** Numbers 64, 152, 163, 214, 243, 246 are
missing. The reference has 371 against our 370, so we are one short overall
while missing six numbers — meaning we also pick up inserted articles they do
not. This is our defect and it is precisely localised: six known headings to
inspect in a 132-page document. Fixable, and the numbering check is what made it
visible at all.

**AUSCGIE-2014 — 303 articles apart, and we have no gaps.** Our extraction is
internally consistent: base numbering 1..920 with zero gaps, 169 inserted
articles, terminal article 920 repealing the 1997 act, 98.6% text coverage.
303 additional provisions cannot hide inside 1.4% of uncaptured characters —
that residue is the cover page, structural headings and the signature block.

Given three exact agreements elsewhere, the likeliest explanation is a
difference in counting unit rather than a difference in extraction: the
reference may split alinéas, or count sub-provisions as separate articles.
**This is a hypothesis, not a conclusion.** Resolving it means comparing article
lists directly rather than counts, which is a Phase 2 task.

## Scanned documents

Four documents have no usable text layer, and the parser refused them rather
than emitting fragments:

- **AUPSRVE-2023** — 40 pages, **0 characters**. Also implausibly short: ~335
  articles do not fit in 40 pages, so this file is likely partial as well as
  scanned. Its version was already flagged unverified.
- **AUA-2017 / AUM-2017** — 14 pages, 1,496 characters, and identical
  checksums, confirming the manifest's warning that one PDF carries both acts.
- **AUDCIF-2017** — 54 pages, 4,564 characters.

The guard here matters more than it looks. A parser without a text-layer check
does not fail on a scan; it succeeds, returning three articles of OCR noise,
and the corpus is quietly poisoned. Refusing is the correct behaviour.

Two routes forward, to be decided: locate text-layer sources on other
government portals, or run OCR (Tesseract with `fra`) and treat OCR'd documents
as a distinct quality tier — since OCR error rates on legal French will affect
retrieval, and mixing tiers silently would confound every later measurement.

## Method note

Two independent checks, because each catches what the other cannot:

- `numbering_gaps()` — legal texts number articles contiguously, so a gap means
  a heading was swallowed. Caught the AUPCAP defect.
- `text_coverage()` — a parser can produce a perfect contiguous 1..920 while
  discarding half of every body. Catches what numbering cannot.

Coverage sits at 96–99% throughout. It should not reach 100%: cover pages,
structural headings and signature blocks are legitimately not article text.
