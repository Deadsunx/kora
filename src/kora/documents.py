"""The units of text the system reasons over.

An `Article` is the natural retrievable unit for this corpus: it is what a lawyer
cites, what a user asks about, and what an answer must point at. Choosing it as
the atom -- rather than an arbitrary 512-token window -- is the single most
consequential ingestion decision in the project, and Phase 3 tests it against a
fixed-size baseline rather than assuming it.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

LegalStatus = Literal["in_force", "superseded"]


class Article(BaseModel):
    """One numbered provision of an acte uniforme."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(..., description="Manifest id, e.g. 'AUSCGIE-2014'.")
    abbrev: str = Field(..., description="Short act code, e.g. 'AUSCGIE'.")
    status: LegalStatus = Field(..., description="Legal status, carried from the manifest.")

    number: str = Field(..., description="Article label as printed: '477', '133-1'.")
    rubric: str = Field("", description="Marginal title printed above the article, if any.")
    text: str = Field(..., description="Body text, headers and footers removed.")

    hierarchy: tuple[str, ...] = Field(
        default=(),
        description="Structural path, outermost first: ('LIVRE 2', 'TITRE 1', ...).",
    )
    page_start: int = Field(..., ge=0)
    page_end: int = Field(..., ge=0)

    @property
    def citation(self) -> str:
        """Canonical short citation, e.g. 'article 477 AUSCGIE (2014)'."""
        year = self.document_id.rsplit("-", 1)[-1]
        return f"article {self.number} {self.abbrev} ({year})"

    @property
    def chunk_id(self) -> str:
        """Stable identifier used as the index key."""
        return f"{self.document_id}#art{self.number}"

    @property
    def sort_key(self) -> tuple[int, int]:
        """Order articles the way the printed text does.

        '133-1' must sort after '133' and before '134', which plain string
        ordering gets wrong ('133-1' < '134' by luck, but '99' > '100' by
        disaster). Splitting into integers makes it correct by construction.
        """
        base, _, suffix = self.number.partition("-")
        return (int(base), int(suffix) if suffix.isdigit() else 0)

    def to_indexed_text(self) -> str:
        """The string actually embedded.

        The rubric and citation are prepended deliberately. A bare article body
        often omits its own subject ("Le conseil d'administration designe parmi
        ses membres un president..."), so a query naming the concept has little
        lexical or semantic overlap with the passage. Prepending the printed
        marginal title restores that signal, and the citation lets a lexical
        retriever match on 'article 477' directly.
        """
        parts = [self.citation]
        if self.hierarchy:
            parts.append(" > ".join(self.hierarchy))
        if self.rubric:
            parts.append(self.rubric)
        parts.append(self.text)
        return "\n".join(parts)


class ParsedDocument(BaseModel):
    """One acte uniforme after parsing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    abbrev: str
    status: LegalStatus
    title: str
    page_count: int
    articles: tuple[Article, ...]

    # Recorded so that a parse can be audited without re-running it.
    source_sha256: str
    parser_version: str

    # Characters remaining after page furniture was stripped -- the denominator
    # for coverage below.
    source_chars: int = 0

    def __len__(self) -> int:
        return len(self.articles)

    @property
    def text_coverage(self) -> float:
        """Fraction of the source text that ended up inside an article.

        The second self-check, and it catches what numbering cannot. A parser
        can produce a perfect contiguous 1..920 while quietly discarding half of
        every article's body -- the gap check would still report zero. Coverage
        near 1.0 means text is being kept, not just headings found.

        It will not reach exactly 1.0, and should not: structural headings,
        the cover page, and signature blocks are legitimately not article text.
        """
        if not self.source_chars:
            return 0.0
        captured = sum(len(a.text) + len(a.rubric) for a in self.articles)
        return captured / self.source_chars

    @property
    def base_numbers(self) -> list[int]:
        return sorted({a.sort_key[0] for a in self.articles})

    def numbering_gaps(self) -> list[int]:
        """Base article numbers absent between 1 and the maximum found.

        A gap is the signature of a parse failure: legal texts number their
        articles contiguously, so a missing 447 means the heading was merged
        into the previous article's body, not that the legislator skipped it.
        This is the project's main self-check on extraction quality, and the
        exact failure that cost the published reference dataset 28% of one act.
        """
        found = set(self.base_numbers)
        if not found:
            return []
        return [n for n in range(1, max(found) + 1) if n not in found]


# \xa0 is written as an escape rather than a literal non-breaking space: it
# is invisible in an editor, and PDF extraction emits it constantly (French
# typography puts one before ; : ! ? and inside grouped numerals).
_WHITESPACE_RE = re.compile("[ \t\xa0]+")
_BLANKLINES_RE = re.compile(r"\n{3,}")


def normalise_text(text: str) -> str:
    """Collapse extraction artefacts without altering meaning.

    Deliberately conservative: PDF extraction introduces non-breaking spaces and
    ragged newlines, but French legal text depends on its punctuation and
    accents, so nothing is lowercased, stripped of diacritics, or de-hyphenated.
    Aggressive normalisation here would quietly damage both retrieval and the
    citations shown to the user.
    """
    text = text.replace("­", "")  # soft hyphens
    text = _WHITESPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = _BLANKLINES_RE.sub("\n\n", text)
    return text.strip()
