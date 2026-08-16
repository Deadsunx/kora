"""BM25 lexical retrieval, tokenised for French legal text.

Why lexical retrieval at all
----------------------------
Dense encoders map text to a semantic neighbourhood, which is exactly wrong for
the tokens that matter most here. "Article 133-1" and "article 133" sit almost
on top of each other in embedding space; so do "société anonyme" and "société à
responsabilité limitée". A user asking about one and getting the other is not a
near miss in law, it is a wrong answer.

BM25 has the opposite failure mode: it cannot match a paraphrase, but it matches
an exact term exactly. The two are complementary, which is the hypothesis this
module exists to test rather than assume.

The tokeniser is the whole game
-------------------------------
An off-the-shelf English tokeniser destroys French legal text in three specific
ways, each of which silently costs recall:

1. **Elision.** `l'article`, `d'un`, `qu'il`, `s'applique` are extremely common.
   Splitting on whitespace alone yields `l'article` as a single token, which
   never matches a query containing `article`.
2. **Accents.** Users type `societe`, `surete`, `execution`. The corpus contains
   `société`, `sûreté`, `exécution`. Without folding, none of them match.
3. **Compound article numbers.** `133-1` must survive as one token. Splitting on
   the hyphen turns an inserted article into article 133, which is a different
   provision.

Nothing here is stemmed. French stemming conflates `société` and `sociétés`
usefully, but also `saisie` (a seizure) with `saisir`, and the risk of merging
distinct legal terms outweighs the recall gain on a corpus this size.
"""

from __future__ import annotations

import re
import unicodedata

from rank_bm25 import BM25Okapi

from kora.documents import Article
from kora.logging import get_logger

log = get_logger(__name__)

# Elided articles and pronouns. Split so that `l'article` yields `article`.
# The elided part is dropped: `l`, `d`, `qu` carry no retrieval signal.
_APOSTROPHES = "'\u2019"  # ASCII and typographic; both occur in these PDFs
_ELISION_RE = re.compile(
    rf"\b(?:[cdjlmnst]|qu|jusqu|lorsqu|puisqu|quoiqu)[{_APOSTROPHES}]",
    re.IGNORECASE,
)

# Keep letters, digits and the internal hyphen of compound article numbers.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[0-9]+)*")

# Deliberately short. An aggressive stoplist removes `contre`, `sans`, `sous`,
# `avant`, `après` -- words that carry real legal meaning ("saisie sans
# commandement préalable", "recours contre la caution").
_STOPWORDS = frozenset(
    # Written as prose and split, not as a list literal: a sixty-word stoplist
    # as one line of quoted strings is unreadable, and this is a list a human
    # will revisit when a query underperforms.
    """
    a au aux avec ce ces dans de des du elle en et eux il je la le les leur lui
    ma mais me meme mes moi mon ne nos notre nous on ou par pas pour qu que qui
    sa se ses son sur ta te tes toi ton tu un une vos votre vous y est sont ete
    etre avoir ont
    """.split()  # noqa: SIM905
)


def fold_accents(text: str) -> str:
    """Strip diacritics: `sûreté` -> `surete`.

    Applied to both documents and queries, so it cannot introduce a mismatch --
    only remove one. Users searching a French legal corpus routinely type
    unaccented text, and the corpus never does.
    """
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def tokenize_fr(text: str) -> list[str]:
    """Tokenise French legal text for lexical matching.

    Order matters: elisions are split *before* accent folding, because the
    typographic apostrophe (U+2019) survives folding and would otherwise glue
    an elided article onto the word it precedes as a single token.
    """
    text = _ELISION_RE.sub(" ", text)
    text = fold_accents(text).lower()
    # Remaining apostrophes are possessive or quotation marks; treat as breaks.
    for apostrophe in _APOSTROPHES:
        text = text.replace(apostrophe, " ")
    return [token for token in _TOKEN_RE.findall(text) if token not in _STOPWORDS]


class BM25Retriever:
    """Okapi BM25 over the same article set as the dense index.

    Built in memory at load time rather than persisted. At 3,056 documents
    construction takes well under a second, and persisting it would mean a
    second artefact to version against the corpus -- a way for the lexical and
    dense views to silently disagree about what they contain.
    """

    def __init__(self, articles: list[Article]) -> None:
        self.articles = articles
        self.chunk_ids = [a.chunk_id for a in articles]

        corpus = [tokenize_fr(a.to_indexed_text()) for a in articles]
        empty = sum(1 for doc in corpus if not doc)
        if empty:
            # An article that tokenises to nothing can never be retrieved
            # lexically. Worth knowing about rather than discovering as a
            # mysteriously unreachable provision.
            log.warning("articles with empty token lists", count=empty)

        self._bm25 = BM25Okapi(corpus)
        log.info(
            "bm25 built",
            documents=len(corpus),
            avg_tokens=round(sum(len(d) for d in corpus) / max(len(corpus), 1), 1),
        )

    def __len__(self) -> int:
        return len(self.chunk_ids)

    def search(self, question: str, k: int) -> list[tuple[str, float]]:
        """Top-k (chunk_id, score) for one question."""
        if k <= 0:
            raise ValueError("k must be positive")
        scores = self._bm25.get_scores(tokenize_fr(question))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.chunk_ids[i], float(scores[i])) for i in ranked]
