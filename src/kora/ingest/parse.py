"""Turn an acte uniforme PDF into structured articles.

Design, and why
---------------
The published reference corpus lost 28% of one act because a two-column layout
merged article boundaries, and that loss was invisible: the parser produced
fewer, longer articles and nothing complained. So this parser is built around a
verifiable invariant rather than around trust.

    Legal texts number their articles contiguously.

If we extract articles 1..920 with no gaps, the extraction is almost certainly
complete. If article 447 is missing, its heading was swallowed -- and
`ParsedDocument.numbering_gaps()` says so loudly. The check costs nothing and
catches the one failure mode that otherwise passes silently.

Three details that matter on this corpus
----------------------------------------
1. **Inserted articles.** The 2014 revision added provisions numbered 120-1,
   133-2, 160-5 and so on -- 169 of them in AUSCGIE alone. A pattern matching
   only `Article \\d+` truncates `133-1` to `133`, producing a duplicate key and
   dropping a real provision. The number is therefore parsed as a string label.

2. **Repeating furniture.** Every page carries the act's title as a header and a
   source URL as a footer. Hardcoding those strings would not survive the next
   act, so they are detected by frequency: a line appearing at the top or bottom
   of most pages is furniture, not content.

3. **Marginal rubrics.** Articles are preceded by a short printed title
   ("Nomination et duree du mandat du president..."). It is genuinely useful --
   a bare article body often never names its own subject -- so it is captured
   as metadata rather than discarded or merged into the body.

4. **Tables of contents.** The subtlest failure, and the one that defeated both
   checks above at once. A contents page repeats every article heading followed
   by dot leaders::

       Article 2
       ..................................................... 14

   The first line matches the heading pattern exactly. Left alone it produces a
   phantom article whose body is a row of dots and, worse, claims the number --
   so the genuine article forty pages later is dismissed as a duplicate
   cross-reference and its text is appended to whatever article is open.

   Numbering stays contiguous, because the phantoms cover every number.
   Coverage stays high, because the text is still present, merely misfiled. Only
   the length distribution shows it: a handful of 20,000-30,000 character blocks
   where real articles run to a few thousand. Hence `longest_article_chars` as a
   third check, and hence dot-leader removal plus claiming a number only when an
   article is actually emitted.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pymupdf

from kora.corpus import Document
from kora.documents import Article, ParsedDocument, normalise_text
from kora.ingest.download import sha256_of
from kora.logging import get_logger

log = get_logger(__name__)

# Bump when parsing logic changes in a way that alters output. Recorded in every
# ParsedDocument so a stored parse can be traced to the code that produced it.
PARSER_VERSION = "1.0"

# `Article 477`, `Article 133-1`, tolerating the several dash characters that
# PDF extraction produces and any text that runs on after the heading.
# The four dash variants are hyphen-minus, non-breaking hyphen, en dash and em
# dash. All four occur in these PDFs for the same logical separator, so matching
# only ASCII '-' would drop inserted articles at random.
_DASHES = "-\u2011\u2013\u2014"  # hyphen-minus, non-breaking, en dash, em dash

# French legal drafting numbers the first article "Article premier", never
# "Article 1" -- and AUPSRVE-2023 carries the convention into its inserted
# articles too ("Article premier-11"), while the body cross-references the same
# provisions numerically ("l'article 1-10"). A digits-only pattern therefore
# does not merely miss article 1; it silently absorbs every premier-N article
# into the preceding article's body.
# Order matters: regex alternation is first-match, not longest-match, so `\d+`
# must come last or "1er" matches as "1" and leaves "er" as the article body.
_BASE_NUMBER = r"(?:premier|1\s*er|\d+)"
ARTICLE_RE = re.compile(
    rf"^\s*Article\s+({_BASE_NUMBER}(?:\s*[{_DASHES}]\s*\d+)?)\s*(?:[.:]\s*)?(.*)$",
    re.IGNORECASE,
)
_DASH_RE = re.compile(rf"\s*[{_DASHES}]\s*")
# Normalised to "1" so that `premier-11` and a cross-reference to `1-11` denote
# the same article, and so the numeric sort key keeps working.
_PREMIER_RE = re.compile(r"^(?:premier|1\s*er)", re.IGNORECASE)

# Structural markers, outermost first. Order defines nesting: seeing a TITRE
# clears any CHAPITRE and SECTION below it.
HIERARCHY_LEVELS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("PARTIE", re.compile(r"^\s*(PARTIE\s+.{0,80})$", re.IGNORECASE)),
    ("LIVRE", re.compile(r"^\s*(LIVRE\s+.{0,80})$", re.IGNORECASE)),
    ("TITRE", re.compile(r"^\s*(TITRE\s+.{0,80})$", re.IGNORECASE)),
    ("CHAPITRE", re.compile(r"^\s*(CHAPITRE\s+.{0,80})$", re.IGNORECASE)),
    ("SECTION", re.compile(r"^\s*(SOUS-SECTION\s+.{0,80})$", re.IGNORECASE)),
    ("SECTION", re.compile(r"^\s*(SECTION\s+.{0,80})$", re.IGNORECASE)),
)

MAX_RUBRIC_CHARS = 160

# Dot leaders: the run of periods joining a table-of-contents entry to its page
# number. They never occur in legal body text, so a line containing one is
# structurally a TOC artefact regardless of what else is on it.
DOT_LEADER_RE = re.compile(r"\.{4,}")


def _detect_furniture(pages: list[str], *, threshold: float = 0.5) -> set[str]:
    """Find header/footer lines that repeat across most pages.

    Looks only at the first and last few lines of each page. A body sentence
    that happens to recur (legal texts repeat formulae constantly) is therefore
    never mistaken for furniture.
    """
    edge_lines: Counter[str] = Counter()
    for text in pages:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            continue
        # Scale the sampled edge to the page. A fixed window of 3 would cover
        # every line of a short page, letting genuine body text be counted as
        # furniture and deleted -- silent content loss, which is the one thing
        # this parser is built to avoid.
        edge = max(1, min(3, len(lines) // 3))
        for line in lines[:edge] + lines[-edge:]:
            edge_lines[line] += 1

    minimum = max(2, int(len(pages) * threshold))
    return {line for line, count in edge_lines.items() if count >= minimum}


def _strip_furniture(text: str, furniture: set[str]) -> list[str]:
    """Drop blank lines and repeating page furniture.

    Table-of-contents handling lives in `articles_from_lines`, not here: it is a
    segmentation concern rather than a page-layout one, and keeping it there
    means every test of the segmenter exercises it.
    """
    return [line for raw in text.splitlines() if (line := raw.strip()) and line not in furniture]


def _match_hierarchy(line: str) -> tuple[str, str] | None:
    for level, pattern in HIERARCHY_LEVELS:
        match = pattern.match(line)
        if match:
            return level, " ".join(match.group(1).split())
    return None


def _is_rubric(line: str) -> bool:
    """Heuristic: is this line a marginal title rather than body text?

    Rubrics are short, do not end in a full stop, and are not themselves
    structural markers. Getting this wrong is cheap in both directions -- a
    missed rubric loses a little retrieval signal, a false one adds a short
    phrase to metadata -- so a simple rule is preferred to a fragile one.
    """
    if not line or len(line) > MAX_RUBRIC_CHARS:
        return False
    if line.endswith((".", ";", ":")):
        return False
    if _match_hierarchy(line) is not None:
        return False
    return not ARTICLE_RE.match(line)


def strip_dot_leaders(lines: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Remove table-of-contents leader lines.

    Applied by `articles_from_lines`, so it is exercised by every segmentation
    test, and again by `parse_pdf` before it measures the source size. That
    second call is what keeps `text_coverage` honest: leaving contents pages in
    the denominator while excluding them from the numerator reports a coverage
    collapse where nothing was actually lost. The function is idempotent, so
    calling it twice costs nothing.
    """
    return [(line, page) for line, page in lines if not DOT_LEADER_RE.search(line)]


class ParseError(RuntimeError):
    pass


def articles_from_lines(
    lines: list[tuple[str, int]],
    document: Document,
) -> list[Article]:
    """Build articles from (line, page_number) pairs.

    Split out from `parse_pdf` so the segmentation logic -- where all the real
    complexity lives -- can be tested against synthetic text, with no PDF and no
    corpus present. CI has neither.
    """
    lines = strip_dot_leaders(lines)

    articles: list[Article] = []
    hierarchy: dict[str, str] = {}
    level_order = ["PARTIE", "LIVRE", "TITRE", "CHAPITRE", "SECTION"]

    current_number: str | None = None
    current_rubric = ""
    current_body: list[str] = []
    current_hierarchy: tuple[str, ...] = ()
    current_start = 0
    current_end = 0
    seen: set[str] = set()

    def flush() -> None:
        nonlocal current_number
        if current_number is None:
            return
        body = normalise_text("\n".join(current_body))
        # Empty bodies mean the heading matched something that was not an
        # article (a cross-reference on its own line, say). Dropping them keeps
        # the numbering check meaningful.
        if body:
            # A number is only claimed once a real article is emitted for it.
            # Claiming it at heading time let an empty phantom (a TOC entry)
            # reserve the number and turn the genuine article into a discarded
            # duplicate.
            seen.add(current_number)
            articles.append(
                Article(
                    document_id=document.id,
                    abbrev=document.abbrev,
                    status=document.status,
                    number=current_number,
                    rubric=current_rubric,
                    text=body,
                    hierarchy=current_hierarchy,
                    page_start=current_start,
                    page_end=current_end,
                )
            )
        current_number = None

    for index, (line, page_number) in enumerate(lines):
        structural = _match_hierarchy(line)
        if structural is not None:
            level, label = structural
            # Entering a level clears everything nested inside it.
            depth = level_order.index(level)
            for deeper in level_order[depth:]:
                hierarchy.pop(deeper, None)
            hierarchy[level] = label
            if current_number is not None:
                current_body.append(line)
            continue

        match = ARTICLE_RE.match(line)
        if match:
            number = _PREMIER_RE.sub("1", _DASH_RE.sub("-", match.group(1).strip()))
            trailing = match.group(2).strip()

            # A repeated number is a cross-reference rendered on its own line,
            # not a second copy of the provision. Treat it as body text.
            #
            # `current_number` must be checked alongside `seen`: the article
            # being accumulated has not been emitted yet, so a heading repeated
            # immediately would otherwise slip through and produce two articles
            # sharing one number.
            if number in seen or number == current_number:
                if current_number is not None:
                    current_body.append(line)
                continue

            flush()

            previous = lines[index - 1][0] if index > 0 else ""
            current_rubric = previous if _is_rubric(previous) else ""
            # The rubric belongs to the heading, not to the previous article.
            if current_rubric and articles and articles[-1].text.endswith(current_rubric):
                trimmed = articles[-1].text[: -len(current_rubric)].strip()
                articles[-1] = articles[-1].model_copy(update={"text": trimmed})

            current_number = number
            current_body = [trailing] if trailing else []
            current_hierarchy = tuple(
                hierarchy[level] for level in level_order if level in hierarchy
            )
            current_start = current_end = page_number
            continue

        if current_number is not None:
            current_body.append(line)
            current_end = page_number

    flush()
    return articles


def parse_pdf(path: Path, document: Document) -> ParsedDocument:
    """Parse one acte uniforme PDF into structured articles."""
    with pymupdf.open(path) as pdf:
        pages = [page.get_text() for page in pdf]
        page_count = len(pages)

    if not pages:
        raise ParseError(f"{document.id}: PDF has no pages")

    extracted = sum(len(p) for p in pages)
    if extracted < 200 * page_count:
        # A scan without OCR yields almost no characters. Fail loudly rather
        # than emit a document with three articles in it.
        raise ParseError(
            f"{document.id}: only {extracted} chars over {page_count} pages -- "
            "this looks like a scanned PDF and needs OCR"
        )

    furniture = _detect_furniture(pages)
    log.debug("furniture detected", document=document.id, lines=len(furniture))

    # Flatten to (line, page_number) so each article can record its page span.
    lines: list[tuple[str, int]] = []
    for page_number, text in enumerate(pages):
        lines.extend((line, page_number) for line in _strip_furniture(text, furniture))

    # Filter before measuring, so contents pages are excluded from both sides
    # of the coverage ratio rather than only from the numerator.
    lines = strip_dot_leaders(lines)
    articles = articles_from_lines(lines, document)
    if not articles:
        raise ParseError(f"{document.id}: no articles found")

    parsed = ParsedDocument(
        document_id=document.id,
        abbrev=document.abbrev,
        status=document.status,
        title=document.title,
        page_count=page_count,
        articles=tuple(sorted(articles, key=lambda a: a.sort_key)),
        source_sha256=sha256_of(path),
        parser_version=PARSER_VERSION,
        source_chars=sum(len(line) for line, _ in lines),
    )

    gaps = parsed.numbering_gaps()
    log.info(
        "parsed",
        document=document.id,
        articles=len(parsed),
        pages=page_count,
        gaps=len(gaps),
    )
    if gaps:
        log.warning(
            "numbering gaps -- headings were probably swallowed",
            document=document.id,
            count=len(gaps),
            first=gaps[:10],
        )

    return parsed
