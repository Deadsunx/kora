"""Build instruction data for QLoRA fine-tuning.

What fine-tuning is for here
----------------------------
A LoRA adapter trained on 3,000 articles will not teach a 4B model OHADA law.
Claiming otherwise would be the central dishonesty available in this phase, so
it is worth stating the target plainly: the adapter is trained to obey the
**output contract**, not to acquire knowledge.

Three behaviours, each of which the harness already measures:

1. **Cite in the exact parseable form.** `article 477 AUSCGIE (2014)`. A base
   instruct model produces citations in whatever style it likes; unparseable
   citations score zero regardless of whether the answer is right.
2. **Abstain when the context cannot support an answer.** Measured by
   `correct_abstention` against the unanswerable questions.
3. **Never present repealed text as current law.** Measured by
   `answers_citing_repealed`.

If fine-tuning improves those three and leaves retrieval untouched, that is a
real and precisely bounded result. If it also degrades answer quality, that
shows up as citation precision falling, which is why every metric is reported
rather than the three being optimised.

Leakage
-------
No training example may use an article that appears in the gold set, and no
training question may resemble a gold question. This is enforced here rather
than trusted: `build_dataset` takes the gold set and excludes its articles by
id, and the exclusion is asserted in the tests. Training on the evaluation set
is the one mistake that would make every number in this project meaningless.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from kora.documents import Article
from kora.eval.dataset import GoldSet
from kora.generation.prompt import ABSTENTION_TOKEN, SYSTEM_PROMPT, build_context
from kora.logging import get_logger

log = get_logger(__name__)

# Question templates keyed to what an article's opening usually states. Templated
# rather than model-generated on purpose: the only model available locally is
# the one being fine-tuned, and training a model on its own output teaches it to
# be more confidently itself rather than more correct.
#
# The targets vary in phrasing so the adapter learns the citation *format*
# rather than a single sentence frame.
QUESTION_TEMPLATES = (
    "Que prévoit l'{citation} ?",
    "Que dit l'{citation} ?",
    "Quelle règle pose l'{citation} ?",
    "Sur quoi porte l'{citation} ?",
)

ANSWER_TEMPLATES = (
    "{summary} ({citation}).",
    "Aux termes de l'{citation}, {lowered}",
    "{summary}. Voir {citation}.",
    "D'après l'{citation} : {lowered}",
)

ABSTENTION_TEMPLATES = (
    "{token} : les extraits fournis ne traitent pas de cette question.",
    "{token} : aucun des extraits ne permet de répondre sur ce point.",
    "{token} : la réponse ne figure pas dans les extraits communiqués.",
)

# Questions for abstention examples. Deliberately plausible for this corpus and
# genuinely outside it, mirroring the unanswerable questions in the gold set --
# a model taught to abstain on obvious nonsense learns nothing useful.
OUT_OF_SCOPE_QUESTIONS = (
    "Quel est le taux de l'impôt sur les sociétés applicable dans cet État partie ?",
    "Quelles sanctions pénales le juge peut-il prononcer en la matière ?",
    "Combien coûte cette formalité auprès du greffe ?",
    "Quelle est la position de la CCJA sur cette question ?",
    "Quel est le délai de préavis applicable au salarié concerné ?",
    "Combien d'entreprises sont concernées dans l'espace OHADA ?",
)


@dataclass(frozen=True, slots=True)
class TrainingExample:
    """One instruction-tuning example in chat format."""

    messages: tuple[dict[str, str], ...]
    kind: str  # cite | abstain | repealed

    def to_json(self) -> dict:
        return {"messages": [dict(m) for m in self.messages], "kind": self.kind}


def _first_sentence(text: str, *, max_chars: int = 260) -> str:
    """The opening statement of an article, which is usually its operative rule.

    The same observation that made 512-token truncation improve reranking: the
    rule is at the top, the enumerations below it are detail.
    """
    cleaned = " ".join(text.split())
    for stop in (". ", " ; "):
        index = cleaned.find(stop)
        if 40 < index < max_chars:
            return cleaned[:index].strip()
    return cleaned[:max_chars].rsplit(" ", 1)[0].strip()


def _example(
    question: str,
    context_articles: list[Article],
    answer: str,
    kind: str,
    *,
    max_chars: int,
) -> TrainingExample:
    from kora.generation.prompt import USER_TEMPLATE

    context = build_context(context_articles, max_chars=max_chars)
    return TrainingExample(
        messages=(
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(context=context, question=question)},
            {"role": "assistant", "content": answer},
        ),
        kind=kind,
    )


def build_dataset(
    articles: list[Article],
    gold: GoldSet,
    *,
    n_cite: int = 600,
    n_abstain: int = 150,
    n_repealed: int = 100,
    passages_per_example: int = 5,
    max_chars: int = 1200,
    seed: int = 42,
) -> list[TrainingExample]:
    """Construct the training set, excluding every article used in evaluation."""
    rng = random.Random(seed)

    forbidden = {chunk_id for q in gold for chunk_id in q.gold_chunk_ids}
    pool = [a for a in articles if a.chunk_id not in forbidden]
    excluded = len(articles) - len(pool)
    log.info("training pool", available=len(pool), excluded_gold_articles=excluded)

    if excluded != len(forbidden):
        # Every gold article should have been found and removed. A mismatch
        # means an id in the gold set does not exist in the corpus, which the
        # eval validator is supposed to prevent.
        log.warning("gold articles not all matched", expected=len(forbidden), removed=excluded)

    in_force = [a for a in pool if not a.repealed and a.status != "superseded"]
    repealed = [a for a in pool if a.repealed or a.status == "superseded"]
    examples: list[TrainingExample] = []

    def distractors(target: Article, count: int) -> list[Article]:
        """Random other articles, so the target is not always in first position."""
        others = rng.sample(in_force, min(count, len(in_force)))
        return [a for a in others if a.chunk_id != target.chunk_id][: count - 1]

    # 1. Cite correctly from the provided context.
    for target in rng.sample(in_force, min(n_cite, len(in_force))):
        summary = _first_sentence(target.text)
        if len(summary) < 40:
            continue
        context = [*distractors(target, passages_per_example), target]
        rng.shuffle(context)

        citation = target.citation
        answer = rng.choice(ANSWER_TEMPLATES).format(
            summary=summary,
            citation=citation,
            lowered=summary[0].lower() + summary[1:],
        )
        question = rng.choice(QUESTION_TEMPLATES).format(citation=citation)
        examples.append(_example(question, context, answer, "cite", max_chars=max_chars))

    # 2. Abstain when the context does not support an answer.
    for _ in range(n_abstain):
        context = rng.sample(in_force, min(passages_per_example, len(in_force)))
        question = rng.choice(OUT_OF_SCOPE_QUESTIONS)
        answer = rng.choice(ABSTENTION_TEMPLATES).format(token=ABSTENTION_TOKEN)
        examples.append(_example(question, context, answer, "abstain", max_chars=max_chars))

    # 3. Prefer in-force law when a repealed article is also present.
    if repealed:
        for _ in range(n_repealed):
            old = rng.choice(repealed)
            target = rng.choice(in_force)
            summary = _first_sentence(target.text)
            if len(summary) < 40:
                continue
            context = [*distractors(target, passages_per_example - 1), target, old]
            rng.shuffle(context)
            answer = (
                f"{summary} ({target.citation}). "
                f"L'{old.citation} figure également dans les extraits mais il est abrogé "
                f"et ne peut être invoqué comme droit en vigueur."
            )
            question = rng.choice(QUESTION_TEMPLATES).format(citation=target.citation)
            examples.append(_example(question, context, answer, "repealed", max_chars=max_chars))

    rng.shuffle(examples)
    log.info(
        "training set built",
        total=len(examples),
        cite=sum(1 for e in examples if e.kind == "cite"),
        abstain=sum(1 for e in examples if e.kind == "abstain"),
        repealed=sum(1 for e in examples if e.kind == "repealed"),
    )
    return examples


def save_dataset(examples: list[TrainingExample], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_json(), ensure_ascii=False))
            handle.write("\n")
    return path
