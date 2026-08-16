"""What the agent asks the model, and how its replies are parsed.

Separated from the loop for the same reason `generation/prompt.py` is separated
from the generator: every decision about what the model is asked can then be
tested without a GPU, and a prompt change is a reviewable diff.

Both prompts are built to fail safe. A decomposition that returns nothing leaves
the original question, and a verification that cannot be parsed is read as
"sufficient" — so a confused model produces plain single-shot retrieval rather
than an error or a wild query. The agent can decline to help; it should not be
able to make things worse by malfunctioning.
"""

from __future__ import annotations

import re

# Emitted when a question needs only one article. Uppercase and unpunctuated,
# for the same reason ABSTENTION_TOKEN is: detecting it must not itself be a
# language-understanding problem.
ATOMIC_TOKEN = "ATOMIQUE"

# The model does not reliably emit the token it was given. Asked in French for
# "ATOMIQUE" it produced `ATOMIC` and `ATOMICITY` -- the English stem, and a
# word that is not a verdict in any language -- and once buried the correct
# token at the end of a sentence of commentary.
#
# All five cases happened to reach the right answer anyway, by three different
# accidents: two replies were shorter than the minimum sub-question length, one
# survived as a single line and was dropped by the two-part rule, and one was
# prose. That is luck, not logic. A model that had written two lines of
# commentary instead of one would have had its commentary retrieved as
# sub-questions.
#
# So the check is on the stem, per line, for a reply that is one word: it
# catches every observed spelling and cannot swallow a genuine sub-question,
# which is never a single word.
_ATOMIC_STEM = "ATOMI"

DECOMPOSE_SYSTEM = """\
Tu prépares une recherche documentaire dans le droit OHADA. Tu ne réponds pas \
à la question.

Ta tâche : déterminer si la question exige PLUSIEURS articles distincts.

- Si un seul article suffit, réponds exactement : ATOMIQUE
- Sinon, écris une sous-question par ligne, deux ou trois au maximum. Chaque \
sous-question doit être autonome, précise, et porter sur UN seul point de droit.

N'écris rien d'autre : pas de numérotation, pas d'explication, pas de réponse."""

DECOMPOSE_USER = "Question : {question}"

# Emitted when the retrieved passages are enough to answer.
SUFFICIENT_TOKEN = "SUFFISANT"

VERIFY_SYSTEM = """\
Tu vérifies si des extraits juridiques suffisent à répondre à une question. Tu \
ne réponds pas à la question.

- Si les extraits contiennent de quoi répondre entièrement, réponds \
exactement : SUFFISANT
- Sinon, écris sur une seule ligne ce qui manque, sous la forme d'une requête \
de recherche : les termes juridiques à retrouver, sans phrase.

N'écris rien d'autre."""

VERIFY_USER = """\
Extraits :

{context}

Question : {question}

Verdict :"""


def build_decompose_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": DECOMPOSE_SYSTEM},
        {"role": "user", "content": DECOMPOSE_USER.format(question=question)},
    ]


def build_verify_messages(question: str, context: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": VERIFY_SYSTEM},
        {"role": "user", "content": VERIFY_USER.format(context=context, question=question)},
    ]


# Leading list furniture the model adds despite being told not to: "1. ", "2) ",
# "- ", "* ", bullet characters.
_LIST_PREFIX_RE = re.compile("^\\s*(?:\\d+\\s*[.)\\-]|[-*\u2022\u2013])\\s*")


def is_atomic_verdict(line: str) -> bool:
    """Whether one line of a reply is the model declining to decompose.

    Requires the line to be a single word beginning with the stem, so a
    sub-question that happens to mention an atom is not mistaken for a verdict.
    """
    cleaned = _clean(line).strip(" .:!;").upper()
    return bool(cleaned) and " " not in cleaned and cleaned.startswith(_ATOMIC_STEM)


def _clean(line: str) -> str:
    line = _LIST_PREFIX_RE.sub("", line).strip()
    # Models occasionally wrap a sub-question in quotes or bold markers.
    return line.strip("\"'*` ").strip()


def parse_subquestions(reply: str, *, original: str, max_subquestions: int) -> list[str]:
    """Extract sub-questions from a decomposition reply.

    Returns an empty list when the question is atomic, when nothing parseable
    came back, or when the model merely restated the question — all three mean
    "no decomposition", and the caller falls back to single-shot retrieval.

    A single sub-question is also rejected. Splitting a question into one part
    is not a decomposition: it replaces the user's wording with the model's,
    which is query rewriting — a different technique, with a different
    hypothesis, that this experiment is not testing.
    """
    text = reply.strip()
    if not text:
        return []

    # An atomic verdict anywhere in the reply settles it, wherever the model
    # chose to put it.
    if any(is_atomic_verdict(line) for line in text.splitlines()):
        return []

    normalised_original = _normalise(original)
    seen: set[str] = set()
    questions: list[str] = []

    for raw in text.splitlines():
        candidate = _clean(raw)
        # Two characters cannot be a legal question; this drops stray bullets
        # and the blank lines models like to emit between items.
        if len(candidate) < 10:
            continue
        key = _normalise(candidate)
        if key == normalised_original or key in seen:
            continue
        seen.add(key)
        questions.append(candidate)
        if len(questions) == max_subquestions:
            break

    return questions if len(questions) > 1 else []


def parse_verdict(reply: str) -> str | None:
    """The follow-up query a verification asked for, or None if it was satisfied.

    None means "retrieval is sufficient, stop". An unparseable reply also
    returns None: the failure mode of a bad follow-up query is a worse ranking,
    so silence is treated as satisfaction rather than as a reason to search
    again on noise.
    """
    text = reply.strip()
    if not text or text.upper().startswith(SUFFICIENT_TOKEN):
        return None

    for raw in text.splitlines():
        candidate = _clean(raw)
        if len(candidate) >= 10 and not candidate.upper().startswith(SUFFICIENT_TOKEN):
            return candidate
    return None


def _normalise(text: str) -> str:
    """Casefold and strip punctuation, for comparing two questions."""
    return re.sub(r"[^\w\s]", "", text).casefold().strip()
