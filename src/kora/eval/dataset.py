"""The gold question set.

This is the most falsifiable part of the project, and the easiest to fake
without noticing. Two failure modes account for most worthless RAG evaluations,
and the schema is built to make both visible rather than to prevent them by good
intentions.

**Leakage.** Generate a question from a chunk with an LLM, then measure whether
retrieval finds that chunk. It will. You have measured that a paraphrase
retrieves its own source, which is not the task. Every question therefore
records how it was written (`provenance`), and headline numbers are reported
only over human-validated questions.

**Single-article bias.** If every question is answerable from one article,
retrieval looks excellent and the hard case -- a question whose answer is spread
across several provisions in different acts -- is never tested. `kind`
partitions the set so that per-kind scores are reportable, because an aggregate
that is 90% lookups says almost nothing.

The set also deliberately contains questions that **cannot** be answered from
the corpus. A legal assistant that never says "I don't know" is dangerous, and
abstention cannot be measured without unanswerable questions to abstain on.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

QuestionKind = Literal[
    "lookup",  # answered by a single article
    "multi_hop",  # requires combining several articles
    "cross_act",  # spans more than one acte uniforme
    "temporal",  # concerns which text is currently in force
    "unanswerable",  # not determinable from this corpus
]

Provenance = Literal[
    "human",  # written by a person reading the act
    "llm_drafted",  # drafted by a model, NOT yet checked -- excluded from headline metrics
    "llm_drafted_human_validated",  # drafted by a model, verified against the text
]


class GoldQuestion(BaseModel):
    """One evaluation question with its ground-truth articles."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    question: str = Field(..., min_length=10)
    kind: QuestionKind
    provenance: Provenance

    # Article chunk_ids that a correct answer must rest on, e.g.
    # "AUSCGIE-2014#art477". Empty exactly when the question is unanswerable.
    gold_chunk_ids: tuple[str, ...] = ()

    # Optional per-article relevance grades for nDCG. Absent means binary.
    grades: dict[str, float] | None = None

    # A short reference answer, used by the LLM judge and by a human reviewing
    # failures. Not string-matched against the system's output: legal prose has
    # too many correct phrasings for exact match to mean anything.
    reference_answer: str = ""

    # Free-text note on why this question is interesting or hard. Worth its
    # weight when revisiting a failure months later.
    note: str = ""

    @model_validator(mode="after")
    def _check_answerability(self) -> GoldQuestion:
        if self.kind == "unanswerable":
            if self.gold_chunk_ids:
                raise ValueError(f"{self.id}: unanswerable question has gold articles")
        elif not self.gold_chunk_ids:
            raise ValueError(f"{self.id}: answerable question has no gold articles")

        if self.kind == "lookup" and len(self.gold_chunk_ids) > 1:
            raise ValueError(
                f"{self.id}: kind 'lookup' implies one article, got {len(self.gold_chunk_ids)}"
                " -- use 'multi_hop' or 'cross_act'"
            )
        if self.kind in ("multi_hop", "cross_act") and len(self.gold_chunk_ids) < 2:
            raise ValueError(f"{self.id}: kind {self.kind!r} requires at least two articles")

        if self.kind == "cross_act":
            documents = {cid.split("#", 1)[0] for cid in self.gold_chunk_ids}
            if len(documents) < 2:
                raise ValueError(
                    f"{self.id}: kind 'cross_act' requires articles from two acts,"
                    f" got {sorted(documents)}"
                )

        if self.grades:
            unknown = set(self.grades) - set(self.gold_chunk_ids)
            if unknown:
                raise ValueError(f"{self.id}: grades reference non-gold articles {sorted(unknown)}")
        return self

    @property
    def is_answerable(self) -> bool:
        return self.kind != "unanswerable"

    @property
    def counts_towards_headline(self) -> bool:
        """Whether this question may appear in a reported result.

        Unvalidated model-drafted questions are kept in the file -- they are
        useful for smoke tests and for triage -- but excluded from anything
        quoted as a result. The distinction lives in the data rather than in a
        convention someone has to remember.
        """
        return self.provenance in ("human", "llm_drafted_human_validated")


class GoldSet(BaseModel):
    """The full evaluation set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int
    description: str = ""
    questions: tuple[GoldQuestion, ...]

    @model_validator(mode="after")
    def _check_unique_ids(self) -> GoldSet:
        ids = [q.id for q in self.questions]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"duplicate question ids: {sorted(duplicates)}")
        return self

    def __len__(self) -> int:
        return len(self.questions)

    def __iter__(self) -> Iterator[GoldQuestion]:  # type: ignore[override]
        return iter(self.questions)

    def headline(self) -> tuple[GoldQuestion, ...]:
        """Questions eligible for reported results."""
        return tuple(q for q in self.questions if q.counts_towards_headline)

    def by_kind(self, kind: QuestionKind) -> tuple[GoldQuestion, ...]:
        return tuple(q for q in self.questions if q.kind == kind)

    def composition(self) -> dict[str, int]:
        """Counts per kind and provenance, for reporting alongside any result.

        A results table without this is not interpretable: 0.9 recall over
        forty lookups and 0.9 over forty multi-hop questions are very different
        claims, and only one of them is impressive.
        """
        summary: dict[str, int] = {"total": len(self.questions)}
        for question in self.questions:
            summary[f"kind:{question.kind}"] = summary.get(f"kind:{question.kind}", 0) + 1
            key = f"provenance:{question.provenance}"
            summary[key] = summary.get(key, 0) + 1
        summary["headline_eligible"] = len(self.headline())
        return summary

    def referenced_documents(self) -> set[str]:
        """Document ids appearing in any gold answer."""
        return {
            cid.split("#", 1)[0] for question in self.questions for cid in question.gold_chunk_ids
        }


def load_gold_set(path: str | Path) -> GoldSet:
    """Load a gold set from JSONL (one question per line) or JSON."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    if path.suffix == ".jsonl":
        questions = [json.loads(line) for line in text.splitlines() if line.strip()]
        return GoldSet.model_validate({"version": 1, "questions": questions})
    return GoldSet.model_validate(json.loads(text))


def save_gold_set(gold: GoldSet, path: str | Path) -> Path:
    """Write a gold set as JSONL, one question per line.

    JSONL rather than JSON so that adding a question is a one-line diff and
    review stays readable in a pull request.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for question in gold.questions:
            handle.write(json.dumps(question.model_dump(mode="json"), ensure_ascii=False))
            handle.write("\n")
    return path
