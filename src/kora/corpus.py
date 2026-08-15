"""The corpus manifest: what documents exist, and their legal status.

Kept separate from `config.py` on purpose. A config describes *a system*; the
manifest describes *the world the system operates on*. Changing the retriever
should not touch the corpus description, and vice versa.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from kora import paths

LegalStatus = Literal["in_force", "superseded"]


class Document(BaseModel):
    """One acte uniforme."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(..., description="Stable identifier, e.g. 'AUSCGIE-2014'.")
    abbrev: str
    year: int
    adopted: date
    title: str
    domain: str
    status: LegalStatus
    supersedes: str | None = None
    superseded_by: str | None = None
    notes: str = ""

    @model_validator(mode="after")
    def _status_matches_links(self) -> Document:
        """A document is superseded if and only if something supersedes it.

        This is the kind of inconsistency that is trivial to introduce by hand
        (add a new revised act, forget to flip the old one's status) and then
        silently poisons every downstream filter. Cheap to check, so we check.
        """
        if self.status == "superseded" and self.superseded_by is None:
            raise ValueError(f"{self.id}: status 'superseded' but no superseded_by")
        if self.status == "in_force" and self.superseded_by is not None:
            raise ValueError(f"{self.id}: has superseded_by but status is 'in_force'")
        return self

    @property
    def citation(self) -> str:
        """Short form used in generated answers, e.g. 'AUSCGIE (2014)'."""
        return f"{self.abbrev} ({self.year})"


class CorpusManifest(BaseModel):
    """The full corpus description."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int
    source_index: str
    url_template: str
    language: str
    documents: tuple[Document, ...]

    @model_validator(mode="after")
    def _check_integrity(self) -> CorpusManifest:
        ids = [doc.id for doc in self.documents]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"duplicate document ids: {sorted(duplicates)}")

        known = set(ids)
        for doc in self.documents:
            # Cross-references must resolve, or a status filter silently drops
            # documents that were meant to be reachable.
            for field, value in (
                ("supersedes", doc.supersedes),
                ("superseded_by", doc.superseded_by),
            ):
                if value is not None and value not in known:
                    raise ValueError(f"{doc.id}.{field} points at unknown id {value!r}")

            # The relationship must be symmetric.
            if doc.superseded_by is not None:
                successor = next(d for d in self.documents if d.id == doc.superseded_by)
                if successor.supersedes != doc.id:
                    raise ValueError(
                        f"asymmetric link: {doc.id}.superseded_by={doc.superseded_by} "
                        f"but {successor.id}.supersedes={successor.supersedes!r}"
                    )
        return self

    # -- access -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.documents)

    def __iter__(self):  # type: ignore[override]
        return iter(self.documents)

    def get(self, document_id: str) -> Document:
        for doc in self.documents:
            if doc.id == document_id:
                return doc
        raise KeyError(document_id)

    def in_force(self) -> tuple[Document, ...]:
        return tuple(d for d in self.documents if d.status == "in_force")

    def superseded(self) -> tuple[Document, ...]:
        return tuple(d for d in self.documents if d.status == "superseded")

    def url_for(self, document: Document | str) -> str:
        doc_id = document if isinstance(document, str) else document.id
        return self.url_template.format(id=doc_id)

    def raw_path(self, document: Document | str) -> Path:
        doc_id = document if isinstance(document, str) else document.id
        return paths.RAW_DIR / f"{doc_id}.pdf"


DEFAULT_MANIFEST_PATH = paths.DATA_DIR / "corpus_manifest.yaml"


def load_manifest(path: str | Path | None = None) -> CorpusManifest:
    """Load and validate the corpus manifest."""
    manifest_path = Path(path) if path is not None else DEFAULT_MANIFEST_PATH
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return CorpusManifest.model_validate(data)
