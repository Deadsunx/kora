"""Tests for the corpus manifest.

The manifest encodes legal status, and every downstream filter trusts it. An
error here does not crash anything -- it just quietly makes the system cite
repealed law. So the invariants are tested rather than assumed.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from kora.corpus import CorpusManifest, load_manifest

MINIMAL = {
    "version": 1,
    "source_index": "https://example.org/index",
    "url_template": "https://example.org/{id}.pdf",
    "language": "fr",
    "documents": [
        {
            "id": "OLD-1997",
            "abbrev": "OLD",
            "year": 1997,
            "adopted": "1997-04-17",
            "title": "Ancien acte",
            "domain": "test",
            "status": "superseded",
            "superseded_by": "NEW-2014",
        },
        {
            "id": "NEW-2014",
            "abbrev": "NEW",
            "year": 2014,
            "adopted": "2014-01-30",
            "title": "Acte révisé",
            "domain": "test",
            "status": "in_force",
            "supersedes": "OLD-1997",
        },
    ],
}


def test_minimal_manifest_validates() -> None:
    manifest = CorpusManifest.model_validate(MINIMAL)
    assert len(manifest) == 2
    assert len(manifest.in_force()) == 1
    assert len(manifest.superseded()) == 1


def test_superseded_requires_successor() -> None:
    payload = {**MINIMAL, "documents": [dict(MINIMAL["documents"][0])]}
    payload["documents"][0].pop("superseded_by")
    with pytest.raises(ValidationError, match="no superseded_by"):
        CorpusManifest.model_validate(payload)


def test_in_force_cannot_be_superseded() -> None:
    docs = [dict(d) for d in MINIMAL["documents"]]
    docs[1]["superseded_by"] = "OLD-1997"
    with pytest.raises(ValidationError, match="status is 'in_force'"):
        CorpusManifest.model_validate({**MINIMAL, "documents": docs})


def test_dangling_reference_rejected() -> None:
    docs = [dict(d) for d in MINIMAL["documents"]]
    docs[0]["superseded_by"] = "GHOST-2020"
    with pytest.raises(ValidationError, match="unknown id"):
        CorpusManifest.model_validate({**MINIMAL, "documents": docs})


def test_asymmetric_link_rejected() -> None:
    """A repeals B, but B does not point back at A."""
    docs = [dict(d) for d in MINIMAL["documents"]]
    docs[1].pop("supersedes")
    with pytest.raises(ValidationError, match="asymmetric"):
        CorpusManifest.model_validate({**MINIMAL, "documents": docs})


def test_duplicate_ids_rejected() -> None:
    docs = [dict(MINIMAL["documents"][1]) for _ in range(2)]
    with pytest.raises(ValidationError, match="duplicate"):
        CorpusManifest.model_validate({**MINIMAL, "documents": docs})


def test_url_template_applied() -> None:
    manifest = CorpusManifest.model_validate(MINIMAL)
    assert manifest.url_for("NEW-2014") == "https://example.org/NEW-2014.pdf"


# -- the real manifest ------------------------------------------------------


def test_real_manifest_is_valid() -> None:
    """The shipped manifest must satisfy every invariant above."""
    manifest = load_manifest()
    assert len(manifest) == 18
    assert len(manifest.in_force()) == 11
    assert len(manifest.superseded()) == 7


def test_every_superseded_act_has_a_live_successor() -> None:
    """Following the chain from any repealed act must reach an in-force one.

    Guards against a two-step replacement (A -> B -> C) where B was marked
    superseded but the chain was never repaired.
    """
    manifest = load_manifest()
    for document in manifest.superseded():
        current = document
        for _ in range(10):
            assert current.superseded_by is not None
            current = manifest.get(current.superseded_by)
            if current.status == "in_force":
                break
        else:
            pytest.fail(f"{document.id}: no in-force successor within 10 hops")


def test_citation_format() -> None:
    manifest = load_manifest()
    assert manifest.get("AUSCGIE-2014").citation == "AUSCGIE (2014)"
