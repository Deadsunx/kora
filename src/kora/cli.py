"""Command line entry point.

Every stage of the pipeline is reachable as `kora <verb> --config <file>`. A
single entry point means the commands you run in development are literally the
same code paths CI and the final demo run -- no drift between "how I ran it" and
"how it works".
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from kora import __version__, paths
from kora.config import load_config, save_resolved
from kora.logging import configure_logging, get_logger

app = typer.Typer(
    name="kora",
    help="RAG over French-language African business law.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging."),
) -> None:
    configure_logging(level="DEBUG" if verbose else "INFO")


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(f"kora {__version__}")


@app.command()
def doctor() -> None:
    """Check that the environment can actually run the project.

    Run this first, and after any dependency change. Nearly every "my training
    script is mysteriously slow" bug is really "torch silently fell back to CPU",
    and this catches it in one second instead of one hour.
    """
    table = Table(title="Environment check", show_lines=False)
    table.add_column("Component", style="bold")
    table.add_column("Status")
    table.add_column("Detail", overflow="fold")

    def row(name: str, ok: bool | None, detail: str) -> None:
        mark = {True: "[green]OK[/]", False: "[red]FAIL[/]", None: "[yellow]--[/]"}[ok]
        table.add_row(name, mark, detail)

    import sys

    py_ok = sys.version_info[:2] == (3, 13)
    row(
        "Python",
        py_ok,
        f"{sys.version.split()[0]}"
        + ("" if py_ok else "  (3.13 expected: 3.14 has no CUDA wheels)"),
    )

    try:
        import torch

        row("PyTorch", True, torch.__version__)
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / 1024**3
            row(
                "CUDA",
                True,
                f"{props.name}  |  {vram_gb:.1f} GiB  |  sm_{props.major}{props.minor}"
                f"  |  torch cuda {torch.version.cuda}",
            )
            # A 4-bit 4B model plus embedder plus reranker is a tight but
            # workable fit in 8 GiB. Below that, plan on smaller models.
            if vram_gb < 7.0:
                row("VRAM headroom", None, "Under 7 GiB: prefer a 1-2B generator.")
        else:
            row("CUDA", False, "Not available -- torch will run on CPU only.")
    except ImportError:
        row("PyTorch", False, "Not installed. `pip install torch --index-url ...cu128`")

    for module, extra in [
        ("sentence_transformers", "retrieval"),
        ("faiss", "retrieval"),
        ("transformers", "train"),
        ("peft", "train"),
        ("bitsandbytes", "train"),
        ("fastapi", "serve"),
    ]:
        try:
            mod = __import__(module)
            row(module, True, getattr(mod, "__version__", "installed"))
        except ImportError:
            # Rich parses square brackets as markup, so `[retrieval]` would be
            # swallowed as an unknown tag. A backslash escapes it.
            row(module, None, rf"not installed  (pip install -e .\[{extra}])")

    console.print(table)

    console.print("\n[bold]Paths[/]")
    for label, path in [
        ("root", paths.PROJECT_ROOT),
        ("data", paths.DATA_DIR),
        ("models", paths.MODELS_DIR),
        ("runs", paths.RUNS_DIR),
    ]:
        exists = "[green]exists[/]" if Path(path).exists() else "[yellow]missing[/]"
        console.print(f"  {label:<8} {path}  {exists}")


@app.command()
def init() -> None:
    """Create the project's data directories."""
    paths.ensure_dirs()
    console.print("[green]Created:[/]")
    for directory in paths.ALL_DIRS:
        console.print(f"  {directory}")


@app.command("show-config")
def show_config(
    config: Path = typer.Option(..., "--config", "-c", exists=True, help="YAML config."),
) -> None:
    """Resolve a config file and print its identity.

    Useful for confirming that an experiment differs from the baseline in
    exactly the way you intended -- the fingerprints tell you at a glance
    whether an index rebuild is required.
    """
    cfg = load_config(config)
    log = get_logger(__name__)
    log.info(
        "config resolved",
        name=cfg.name,
        run_id=cfg.run_id,
        index_fingerprint=cfg.index_fingerprint(),
    )

    table = Table(show_header=False, box=None)
    table.add_row("[bold]name[/]", cfg.name)
    table.add_row("[bold]run_id[/]", cfg.run_id)
    table.add_row("[bold]index fingerprint[/]", cfg.index_fingerprint())
    table.add_row("[bold]chunking[/]", cfg.chunking.strategy)
    table.add_row(
        "[bold]retrieval[/]",
        " + ".join(
            filter(
                None,
                [
                    "dense" if cfg.retrieval.dense else "",
                    "bm25" if cfg.retrieval.bm25 else "",
                    "rerank" if cfg.reranker.enabled else "",
                ],
            )
        )
        or "none",
    )
    table.add_row("[bold]generator[/]", cfg.generator.model_name)
    table.add_row("[bold]adapter[/]", cfg.generator.adapter_path or "none (base model)")
    console.print(table)

    destination = paths.run_path(cfg.run_id) / "config.resolved.yaml"
    save_resolved(cfg, destination)
    console.print(f"\nResolved config written to [dim]{destination}[/]")


@app.command()
def compare(
    configs: list[Path] = typer.Argument(..., help="Config files to compare."),
) -> None:
    """Show how a set of configs differ, and which of them share an index.

    Run this before launching an ablation series. Two configs with the same
    index fingerprint reuse one index -- if two experiments you *expect* to
    differ in retrieval share a fingerprint, you have made an editing mistake
    and would otherwise discover it only in a results table that looks oddly
    flat.
    """
    loaded = [load_config(path) for path in configs]

    table = Table(title="Config comparison")
    table.add_column("name", style="bold")
    table.add_column("run fp")
    table.add_column("index fp")
    table.add_column("chunking")
    table.add_column("retrieval")
    table.add_column("adapter")

    for cfg in loaded:
        pipeline = " + ".join(
            filter(
                None,
                [
                    "dense" if cfg.retrieval.dense else "",
                    "bm25" if cfg.retrieval.bm25 else "",
                    "rerank" if cfg.reranker.enabled else "",
                ],
            )
        )
        table.add_row(
            cfg.name,
            cfg.fingerprint(),
            cfg.index_fingerprint(),
            f"{cfg.chunking.strategy}/{cfg.chunking.max_tokens}",
            f"{pipeline}  (k={cfg.retrieval.top_k}->{cfg.retrieval.final_k})",
            cfg.generator.adapter_path or "-",
        )

    console.print(table)

    # Group by index fingerprint so the rebuild cost of the series is obvious.
    groups: dict[str, list[str]] = {}
    for cfg in loaded:
        groups.setdefault(cfg.index_fingerprint(), []).append(cfg.name)

    console.print(f"\n[bold]{len(groups)} index build(s) needed for {len(loaded)} experiments[/]")
    for fingerprint, names in groups.items():
        console.print(f"  [dim]{fingerprint}[/]  {', '.join(names)}")

    duplicates = [names for names in groups.values() if len(names) > 1]
    if duplicates:
        console.print("[dim]Shared indexes are reused automatically -- no rebuild.[/]")


corpus_app = typer.Typer(help="Corpus acquisition and inspection.", no_args_is_help=True)
app.add_typer(corpus_app, name="corpus")


@corpus_app.command("list")
def corpus_list(
    status: str = typer.Option("all", help="Filter: all | in_force | superseded."),
) -> None:
    """List the documents described by the manifest, and whether we hold them."""
    from kora.corpus import load_manifest

    manifest = load_manifest()
    documents = {
        "all": manifest.documents,
        "in_force": manifest.in_force(),
        "superseded": manifest.superseded(),
    }.get(status)
    if documents is None:
        raise typer.BadParameter("status must be one of: all, in_force, superseded")

    table = Table(title=f"Corpus manifest ({len(documents)} documents)")
    table.add_column("id", style="bold")
    table.add_column("status")
    table.add_column("domain")
    table.add_column("local", justify="right")

    for doc in documents:
        path = manifest.raw_path(doc)
        local = f"{path.stat().st_size / 1024:.0f} KB" if path.exists() else "[yellow]absent[/]"
        colour = "green" if doc.status == "in_force" else "yellow"
        table.add_row(doc.id, f"[{colour}]{doc.status}[/]", doc.domain, local)

    console.print(table)
    held = sum(1 for d in documents if manifest.raw_path(d).exists())
    console.print(f"\n{held}/{len(documents)} present in {paths.RAW_DIR}")


@corpus_app.command("download")
def corpus_download(
    only: list[str] = typer.Option(
        None, "--only", help="Restrict to specific document ids. Repeatable."
    ),
    force: bool = typer.Option(False, "--force", help="Refetch even if present."),
) -> None:
    """Download the corpus PDFs and record provenance for each."""
    from kora.ingest.download import download_corpus

    manifest = load_manifest_or_exit()
    results, failures = download_corpus(manifest, only=tuple(only or ()), force=force)

    table = Table(title="Download results")
    table.add_column("id", style="bold")
    table.add_column("size", justify="right")
    table.add_column("sha256")
    table.add_column("action")

    for document, record, fetched in results:
        table.add_row(
            document.id,
            f"{record.size_bytes / 1024:.0f} KB",
            record.sha256[:16],
            "[green]fetched[/]" if fetched else "[dim]cached[/]",
        )
    console.print(table)

    total_mb = sum(r.size_bytes for _, r, _ in results) / 1024**2
    console.print(f"\n{len(results)} documents, {total_mb:.1f} MB total")

    if failures:
        console.print(f"\n[yellow]{len(failures)} not retrieved:[/]")
        for document, reason in failures:
            console.print(f"  [bold]{document.id}[/]  {reason[:150]}")


@corpus_app.command("parse")
def corpus_parse(
    only: list[str] = typer.Option(None, "--only", help="Restrict to specific ids."),
    show: int = typer.Option(0, "--show", help="Print the first N parsed articles."),
) -> None:
    """Parse downloaded PDFs into structured articles and report extraction quality.

    The `gaps` column is the one to watch. Legal texts number articles
    contiguously, so a non-zero count means headings were swallowed and the
    parse is lossy -- which is exactly the failure that silently cost the
    published reference corpus 28% of one act.
    """
    import json

    from kora.ingest.parse import ParseError, parse_pdf

    manifest = load_manifest_or_exit()
    targets = [d for d in manifest.documents if not only or d.id in only]

    table = Table(title="Parse results")
    table.add_column("id", style="bold")
    table.add_column("pages", justify="right")
    table.add_column("articles", justify="right")
    table.add_column("gaps", justify="right")
    table.add_column("coverage", justify="right")
    table.add_column("rubrics", justify="right")
    table.add_column("hierarchy", justify="right")
    table.add_column("status")

    parsed_any = False
    for document in targets:
        pdf_path = manifest.raw_path(document)
        if not pdf_path.exists():
            table.add_row(document.id, *["-"] * 5, "[yellow]not downloaded[/]")
            continue
        try:
            parsed = parse_pdf(pdf_path, document)
        except ParseError as exc:
            table.add_row(document.id, *["-"] * 5, f"[red]{exc}[/]")
            continue

        gaps = parsed.numbering_gaps()
        total = max(len(parsed), 1)
        rubrics = sum(1 for a in parsed.articles if a.rubric)
        placed = sum(1 for a in parsed.articles if a.hierarchy)
        coverage = parsed.text_coverage
        table.add_row(
            document.id,
            str(parsed.page_count),
            str(len(parsed)),
            f"[red]{len(gaps)}[/]" if gaps else "[green]0[/]",
            f"[green]{coverage:.1%}[/]" if coverage > 0.9 else f"[yellow]{coverage:.1%}[/]",
            f"{rubrics * 100 // total}%",
            f"{placed * 100 // total}%",
            "[green]ok[/]" if not gaps and coverage > 0.9 else "[yellow]check[/]",
        )

        destination = paths.INTERIM_DIR / f"{document.id}.articles.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for article in parsed.articles:
                handle.write(json.dumps(article.model_dump(mode="json"), ensure_ascii=False))
                handle.write("\n")
        parsed_any = True

        if show:
            for article in parsed.articles[:show]:
                console.print(f"\n[bold cyan]{article.citation}[/]")
                if article.hierarchy:
                    console.print(f"  [dim]{' > '.join(article.hierarchy)}[/]")
                if article.rubric:
                    console.print(f"  [italic]{article.rubric}[/]")
                console.print(f"  {article.text[:400]}")

    console.print(table)
    if parsed_any:
        console.print(f"\nArticles written to [dim]{paths.INTERIM_DIR}[/]")


def load_manifest_or_exit():
    """Load the manifest, turning validation errors into a readable message."""
    from pydantic import ValidationError

    from kora.corpus import load_manifest

    try:
        return load_manifest()
    except FileNotFoundError as exc:
        console.print(f"[red]Manifest not found:[/] {exc}")
        raise typer.Exit(1) from exc
    except ValidationError as exc:
        console.print("[red]Manifest is invalid:[/]")
        console.print(str(exc))
        raise typer.Exit(1) from exc


if __name__ == "__main__":
    app()
