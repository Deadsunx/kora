# Extraction quality

Date: 2026-08-16 · parser version 1.0 · 11 of 18 documents acquired · **2,608 articles**

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
| AUPCAP-2015 | 122 | 377 | 0 | 99.3% | 371 | **+6** |
| AUDCIF-2017 | 40 | 123 | 0 | 98.6% | 120 | **+3** |
| AUM-2017 | 9 | 18 | 0 | 95.5% | *absent* | — |
| AUSCGIE-2014 | 217 | 1089 | 0 | 98.6% | 1392 | **−303** |
| AUPSRVE-2023 | 122 | — | — | — | 242 | scan |
| SYCEBNL-2022 | 438 | — | — | — | *absent* | scan |

**Zero numbering gaps across every parsed document.** Coverage 95.5–99.3%.

## Reading the table

**Five exact agreements** (397, 307, 228, 38, 31). Two parsers written
independently from different sources, converging to the article on documents
spanning an order of magnitude in size, is the strongest evidence available that
both extract correctly. Everything below is licensed by that.

**AUPCAP-2015: +6, and our own defect fixed.** The earlier Senegal PDF gave 370
articles with 6 numbering gaps. The mirror's text-layer PDF gives 377 with none,
now 6 *above* the reference. The 6 articles we previously lost were a source
problem, not a parser problem — and the numbering invariant is what
distinguished the two rather than leaving it a guess.

**AUM-2017 is absent from the reference entirely**, confirming that it covers 9
of the 11 in-force acts. SYCEBNL-2022 is likewise absent.

**AUSCGIE-2014: −303, and now corroborated.** This act was parsed twice from two
different PDFs — the Congo government copy (637 KB, 209 pages) and the mirror
copy (771 KB, 217 pages) — and both yield **exactly 1089 articles with zero
gaps**. Two different source files, two different page counts, the same article
list.

That materially strengthens the case that 1089 is correct for this text, since a
parser bug would have to produce the identical wrong answer on two differently
laid-out PDFs. It does not *prove* the reference wrong: a difference in counting
unit (splitting alinéas, counting sub-provisions) would explain the gap without
either extraction being faulty. Settling it requires comparing article *lists*,
not counts — a Phase 2 task.

## Remaining scans

Two documents still have no usable text layer:

- **AUPSRVE-2023** — 122 pages, 10,140 characters (83 per page). Also emits a
  MuPDF font error, suggesting embedded subset fonts without a usable encoding.
- **SYCEBNL-2022** — 438 pages, 66,126 characters (151 per page).

Both are refused rather than parsed. A parser without a text-layer check does
not fail on a scan; it succeeds, returns a handful of articles of OCR noise, and
poisons the corpus silently.

AUPSRVE-2023 matters most of the three open problems: it is the most-litigated
act in OHADA law, and the one where the reference reports 242 articles against
roughly 335 official. It is the headline comparison and it is not yet parsed.

## Method note

Two independent checks, because each catches what the other cannot:

- `numbering_gaps()` — legal texts number articles contiguously, so a gap means
  a heading was swallowed.
- `text_coverage()` — a parser can produce a perfect contiguous 1..920 while
  discarding half of every body.

Coverage should not reach 100%: cover pages, structural headings and signature
blocks are legitimately not article text.

## Open items

1. AUPSRVE-2023 and SYCEBNL-2022 need a text source or OCR (Tesseract `fra`).
   If OCR, they must be tracked as a distinct quality tier — OCR error rates on
   legal French will affect retrieval, and mixing tiers silently would confound
   every later measurement.
2. AUSCGIE-2014 −303: compare article lists against the reference, not counts.
3. The 7 repealed acts remain unsourced; the mirror 403s on all of them.
