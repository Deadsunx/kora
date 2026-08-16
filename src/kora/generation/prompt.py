"""Prompt construction and answer parsing.

Separated from model loading so that every decision about *what the model is
asked* can be tested without a GPU, and so a prompt change is a reviewable diff
rather than a string buried in an inference loop.

Three requirements shape the prompt, and each corresponds to a metric:

**Citations must be parseable.** An uncited legal answer cannot be checked, and
checkability is the entire value proposition here. The requested format is the
one the model already sees in its context -- `article 477 AUSCGIE (2014)` -- so
it copies rather than invents, and it maps back to a chunk id exactly.

**Abstention must be expressible.** A model with no way to say "the provided
text does not answer this" will answer anyway. The sentinel is a fixed string
rather than free prose so that detecting it is not itself a language-understanding
problem.

**Repealed passages must be visibly marked.** The corpus deliberately contains
repealed law as distractors. Marking is done in the context rather than by
filtering, because a system that never sees repealed text cannot demonstrate
that it declines to rely on it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from kora.config import ExperimentConfig
from kora.documents import Article

# The sentinel a model emits when the context cannot support an answer.
# Uppercase and unpunctuated so it cannot be confused with prose that happens to
# express uncertainty.
ABSTENTION_TOKEN = "INSUFFISANT"

SYSTEM_PROMPT = """\
Tu es un assistant juridique spécialisé dans le droit OHADA (Organisation pour \
l'Harmonisation en Afrique du Droit des Affaires).

Règles impératives :

1. Réponds UNIQUEMENT à partir des extraits fournis. N'utilise aucune \
connaissance extérieure, même si tu crois la question facile.
2. Cite chaque affirmation sous la forme exacte : article NUMÉRO ABRÉVIATION \
(ANNÉE). Exemple : article 477 AUSCGIE (2014).
3. Si un extrait est signalé comme ABROGÉ, ne t'appuie pas dessus pour énoncer \
le droit en vigueur. Tu peux le mentionner pour expliquer l'évolution du texte, \
en précisant qu'il est abrogé.
4. Si les extraits fournis ne permettent pas de répondre, réponds exactement : \
INSUFFISANT
   suivi d'une phrase indiquant ce qui manque. N'invente jamais une règle, un \
chiffre ou un article.
5. Réponds en français, de manière concise et précise."""

USER_TEMPLATE = """\
Extraits du corpus OHADA :

{context}

Question : {question}

Réponse :"""


@dataclass(frozen=True, slots=True)
class Citation:
    """One citation extracted from a generated answer."""

    number: str
    abbrev: str
    year: str

    @property
    def chunk_id(self) -> str:
        return f"{self.abbrev}-{self.year}#art{self.number}"

    def __str__(self) -> str:
        return f"article {self.number} {self.abbrev} ({self.year})"


# `article 477 AUSCGIE (2014)`, tolerating `art.`, `Article`, and inserted
# article numbers such as 133-1. The year is required: without it a citation
# cannot be resolved to a version, and distinguishing the 1997 AUSCGIE from the
# 2014 one is the point.
CITATION_RE = re.compile(
    r"\bart(?:icle|\.)?\s*(\d+(?:-\d+)?)\s+([A-Z]{2,10})\s*\(\s*(\d{4})\s*\)",
    re.IGNORECASE,
)


def build_context(articles: list[Article], *, max_chars: int) -> str:
    """Render retrieved articles as numbered extracts.

    Each extract carries its citation, so the model has the exact string it is
    asked to reproduce. Repealed articles carry an explicit marker: the aim is
    for the system to *see* repealed law and decline to rely on it, which is a
    stronger claim than never having been shown it.
    """
    blocks = []
    for position, article in enumerate(articles, start=1):
        header = f"[{position}] {article.citation}"
        if article.repealed or article.status == "superseded":
            header += "  ⚠ TEXTE ABROGÉ"
        if article.rubric:
            header += f"\n{article.rubric}"

        body = article.text
        if len(body) > max_chars:
            body = body[:max_chars].rsplit(" ", 1)[0] + " […]"
        blocks.append(f"{header}\n{body}")
    return "\n\n".join(blocks)


def build_messages(
    question: str,
    articles: list[Article],
    config: ExperimentConfig,
) -> list[dict[str, str]]:
    """Build the chat messages for one question."""
    context = build_context(articles, max_chars=config.generator.max_passage_chars)

    system = SYSTEM_PROMPT
    if not config.generator.allow_abstention:
        # The ablation that shows what abstention costs and buys. Removing the
        # option should raise answer coverage and raise fabrication together.
        system = "\n".join(
            line for line in system.splitlines() if not line.startswith(("4.", "   suivi"))
        )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": USER_TEMPLATE.format(context=context, question=question)},
    ]


def extract_citations(answer: str) -> list[Citation]:
    """Pull citations out of a generated answer, in order, deduplicated.

    Order is preserved because the first citation is usually the one carrying
    the answer, and duplicates are dropped so that a model repeating one article
    five times does not score as five citations.
    """
    seen: set[str] = set()
    citations: list[Citation] = []
    for match in CITATION_RE.finditer(answer):
        citation = Citation(
            number=match.group(1),
            abbrev=match.group(2).upper(),
            year=match.group(3),
        )
        if citation.chunk_id not in seen:
            seen.add(citation.chunk_id)
            citations.append(citation)
    return citations


def is_abstention(answer: str) -> bool:
    """Whether the model declined to answer.

    Checks the opening of the response rather than anywhere in it: an answer
    that states a rule and then remarks that the extracts were insufficient on
    some secondary point is an answer, not an abstention.
    """
    return answer.strip().upper().startswith(ABSTENTION_TOKEN)
