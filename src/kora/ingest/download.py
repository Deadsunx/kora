"""Fetch the corpus, and record exactly what was fetched.

The provenance record written alongside each PDF is not bureaucracy. Legal texts
on the web get silently re-uploaded: a scanned edition replaced by a clean one, a
consolidated version swapped in, a corrected typo. If that happens mid-project,
your index and your gold QA set now disagree with your source, and nothing in the
system will tell you.

A checksum turns that from an invisible corruption into a loud failure.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from kora.corpus import CorpusManifest, Document
from kora.logging import get_logger

log = get_logger(__name__)

USER_AGENT = "kora/0.1 (academic research project; contact via repository)"


@dataclass(frozen=True, slots=True)
class Provenance:
    """Everything needed to prove where a local file came from."""

    document_id: str
    url: str
    sha256: str
    size_bytes: int
    fetched_at: str
    http_status: int
    content_type: str | None
    last_modified: str | None
    etag: str | None

    def path_for(self, pdf_path: Path) -> Path:
        return pdf_path.with_suffix(".provenance.json")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        # Chunked so that a large PDF never has to sit in memory whole.
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DownloadError(RuntimeError):
    pass


@retry(
    retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _fetch(client: httpx.Client, url: str) -> httpx.Response:
    """GET with retries.

    Retries cover transport errors and HTTP errors alike -- a 502 from an
    overloaded government file server is exactly as transient as a dropped TCP
    connection, and both resolve on a second attempt.
    """
    response = client.get(url)
    response.raise_for_status()
    return response


def download_document(
    document: Document,
    manifest: CorpusManifest,
    *,
    client: httpx.Client,
    force: bool = False,
) -> tuple[Path, Provenance, bool]:
    """Download one act. Returns (path, provenance, was_downloaded).

    Skips the fetch when a valid local copy already exists, so re-running
    ingestion is cheap and does not hammer the source. `force` overrides that,
    and deliberately re-checks the checksum -- which is how you would discover
    that the upstream document changed.
    """
    destination = manifest.raw_path(document)
    provenance_path = destination.with_suffix(".provenance.json")
    url = manifest.url_for(document)

    if destination.exists() and provenance_path.exists() and not force:
        record = Provenance(**json.loads(provenance_path.read_text(encoding="utf-8")))
        if sha256_of(destination) == record.sha256:
            log.debug("already present", document=document.id)
            return destination, record, False
        log.warning("checksum mismatch on local file, refetching", document=document.id)

    log.info("downloading", document=document.id, url=url)
    response = _fetch(client, url)

    content_type = response.headers.get("content-type", "")
    body = response.content

    # The failure mode this catches: a site that returns a styled 404 page with
    # status 200. Without the check it lands on disk as a "PDF" and the parser
    # produces a document full of navigation text.
    if not body.startswith(b"%PDF"):
        preview = body[:120].decode("utf-8", errors="replace")
        raise DownloadError(
            f"{document.id}: response is not a PDF "
            f"(content-type={content_type!r}, starts with {preview!r})"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary file first: an interrupted download must never leave
    # a truncated PDF that looks valid to the next run.
    temporary = destination.with_suffix(".pdf.part")
    temporary.write_bytes(body)
    temporary.replace(destination)

    record = Provenance(
        document_id=document.id,
        url=url,
        sha256=sha256_of(destination),
        size_bytes=destination.stat().st_size,
        fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
        http_status=response.status_code,
        content_type=content_type or None,
        last_modified=response.headers.get("last-modified"),
        etag=response.headers.get("etag"),
    )
    provenance_path.write_text(
        json.dumps(asdict(record), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log.info(
        "downloaded",
        document=document.id,
        kb=round(record.size_bytes / 1024),
        sha256=record.sha256[:12],
    )
    return destination, record, True


def download_corpus(
    manifest: CorpusManifest,
    *,
    only: tuple[str, ...] = (),
    force: bool = False,
    timeout: float = 60.0,
) -> list[tuple[Document, Provenance, bool]]:
    """Download every document in the manifest (or a named subset)."""
    targets = [d for d in manifest.documents if not only or d.id in only]
    if only:
        unknown = set(only) - {d.id for d in manifest.documents}
        if unknown:
            raise KeyError(f"unknown document ids: {sorted(unknown)}")

    results: list[tuple[Document, Provenance, bool]] = []
    headers = {"User-Agent": USER_AGENT}

    with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as client:
        for document in targets:
            path, record, fetched = download_document(
                document, manifest, client=client, force=force
            )
            assert path.exists()
            results.append((document, record, fetched))

    return results
