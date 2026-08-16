# Validating the gold question set

Only validated questions appear in reported results. This is the single gate
between "a model wrote a plausible question" and "a person checked it against
the article", so it is worth doing slowly and worth doing honestly.

## The one question you are asking

> **Does the text of the gold article actually support the reference answer?**

Not "is this true about OHADA law." Not "is this a good question." Only: *if a
system retrieved exactly these articles, could it produce that answer?*

If the article does not contain the fact, the gold label is wrong. Every system
will then be scored against an impossible target, and the metric silently
measures nothing.

## Commands

Review a batch:

```bash
.venv\Scripts\python.exe -m kora.cli eval review --kind lookup --start 0 --count 10
```

Promote the ones that hold up:

```bash
.venv\Scripts\python.exe -m kora.cli eval promote q001 q002 q004
```

Send one back if you promoted it by mistake:

```bash
.venv\Scripts\python.exe -m kora.cli eval promote q004 --demote
```

Look up any article directly, to check a suspicion:

```bash
.venv\Scripts\python.exe -m kora.cli corpus show AUSCGIE-2014#art387
```

Find the article that *should* have been the gold, when one is wrong:

```bash
.venv\Scripts\python.exe -m kora.cli corpus search "capital social minimum" --only AUSCGIE-2014
```

Check the state of the set at any point:

```bash
.venv\Scripts\python.exe -m kora.cli eval validate
```

## What to check, by kind

**`lookup`** (27) — fastest. One short article each. The answer should be almost
verbatim in the text. If you find yourself reasoning to connect the article to
the answer, it is not a lookup: either repoint the gold or change the kind.

**`multi_hop`** (12) — the real test is whether **both** articles are genuinely
needed. If one article alone answers the question, the second is padding and
recall will be understated for every system. Ask: *remove either article — is
the answer still complete?* If yes, the question is a lookup wearing a costume.

**`cross_act`** (8) — same test, plus: do the articles really come from
different acts, and does the question genuinely require both? These are the
hardest to get right and the most valuable when correct.

**`temporal`** (2 remaining: q050, q051) — check that the gold article actually
states the abrogation or transitional rule, not merely something nearby. Both
were repointed after the first pass found them wrong; their new gold articles
(AUPSRVE-2023#art337 and AUDCIF-2017#art112) state the rule explicitly.

**`unanswerable`** (8, all validated) — no article to read. You are only asking:
*is this genuinely absent from the ten actes uniformes?* Beware the question
that is **almost** answerable — q057 was originally mislabelled this way, and
would have penalised a system for correctly citing article 387.

## Signs a question is wrong

- The reference answer contains a fact you cannot find in the printed article.
- The reference answer hedges: "les dispositions transitoires déterminent…"
  usually means the writer never located the article and is describing one from
  memory. This is exactly how q050 was wrong.
- A `lookup` whose answer needs two facts from different places.
- A `multi_hop` whose second article adds nothing.
- An `unanswerable` where the corpus in fact contains the general rule and only
  the specific framing is missing.

## Suggested order

1. **`temporal`** — 2 left, both repointed and ready to confirm.
2. **`lookup`** — 27, quick, mostly definitional.
3. **`multi_hop`** — 12, needs the "is the second article load-bearing?" test.
4. **`cross_act`** — 8, slowest and most worth care.

There is no obligation to validate all 60. An unreviewed question costs nothing;
an incorrectly promoted one costs the credibility of every table it appears in.
Twenty carefully checked questions make a better result than sixty waved
through — and the composition table printed with every run shows exactly how
many stand behind each number.
