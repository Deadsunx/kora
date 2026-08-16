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
        ("transformers", "generate"),
        ("peft", "generate"),
        ("bitsandbytes", "generate"),
        ("fastapi", "serve"),
        ("sse_starlette", "serve"),
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
    table.add_column("hier", justify="right")
    table.add_column("abrogés", justify="right")
    table.add_column("status")

    parsed_any = False
    for document in targets:
        pdf_path = manifest.raw_path(document)
        if not pdf_path.exists():
            table.add_row(document.id, *["-"] * 6, "[yellow]not downloaded[/]")
            continue
        try:
            parsed = parse_pdf(pdf_path, document)
        except ParseError as exc:
            table.add_row(document.id, *["-"] * 6, f"[red]{exc}[/]")
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
            str(len(parsed.repealed_articles)) if parsed.repealed_articles else "-",
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


@corpus_app.command("search")
def corpus_search(
    query: str = typer.Argument(..., help="Case-insensitive substring or regex."),
    regex: bool = typer.Option(False, "--regex", help="Treat the query as a regex."),
    only: list[str] = typer.Option(None, "--only", help="Restrict to specific document ids."),
    limit: int = typer.Option(15, "--limit", help="Maximum articles to show."),
    chars: int = typer.Option(300, "--chars", help="Characters of body to print."),
) -> None:
    """Search parsed article text.

    Written for building and checking the gold question set. A question whose
    gold article was chosen from memory rather than from the text is worthless,
    and the reviewer needs the same lookup to verify it -- so this is a real
    command rather than a throwaway script.
    """
    import json
    import re as _re

    pattern = _re.compile(query if regex else _re.escape(query), _re.IGNORECASE)
    wanted = set(only or ())

    shown = 0
    total = 0
    for path in sorted(paths.INTERIM_DIR.glob("*.articles.jsonl")):
        document_id = path.name.removesuffix(".articles.jsonl")
        if wanted and document_id not in wanted:
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                article = json.loads(line)
                haystack = f"{article['rubric']}\n{article['text']}"
                if not pattern.search(haystack):
                    continue
                total += 1
                if shown >= limit:
                    continue
                shown += 1
                citation = f"article {article['number']} {article['abbrev']}"
                console.print(f"\n[bold cyan]{citation}[/]  [dim]{article['document_id']}[/]")
                console.print(f"  [dim]id:[/] {document_id}#art{article['number']}")
                if article["hierarchy"]:
                    console.print(f"  [dim]{' > '.join(article['hierarchy'])}[/]")
                if article["rubric"]:
                    console.print(f"  [italic]{article['rubric']}[/]")
                body = article["text"][:chars].replace("\n", " ")
                console.print(f"  {body}{'...' if len(article['text']) > chars else ''}")

    console.print(
        f"\n[bold]{total} article(s) matched[/]" + (f", showing {shown}" if total > shown else "")
    )
    if not total:
        console.print("[dim]Nothing found. Have you run `kora corpus parse`?[/]")


@corpus_app.command("show")
def corpus_show(
    chunk_id: str = typer.Argument(..., help="e.g. AUSCGIE-2014#art477"),
) -> None:
    """Print one article in full, by chunk id."""
    import json

    if "#art" not in chunk_id:
        raise typer.BadParameter("expected the form DOCUMENT-ID#artNUMBER")
    document_id, number = chunk_id.split("#art", 1)

    path = paths.INTERIM_DIR / f"{document_id}.articles.jsonl"
    if not path.exists():
        console.print(f"[red]No parsed articles for {document_id}[/]")
        raise typer.Exit(1)

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            article = json.loads(line)
            if article["number"] != number:
                continue
            console.print(f"[bold cyan]article {number} {article['abbrev']}[/]")
            console.print(f"[dim]{article['document_id']}  ({article['status']})[/]")
            if article["hierarchy"]:
                console.print(f"[dim]{' > '.join(article['hierarchy'])}[/]")
            if article["rubric"]:
                console.print(f"[italic]{article['rubric']}[/]\n")
            console.print(article["text"])
            return

    console.print(f"[red]Article {number} not found in {document_id}[/]")
    raise typer.Exit(1)


eval_app = typer.Typer(help="Evaluation set and experiment runs.", no_args_is_help=True)
app.add_typer(eval_app, name="eval")


@eval_app.command("validate")
def eval_validate(
    path: Path = typer.Option(paths.EVAL_DIR / "gold_qa.jsonl", "--path", help="Gold set file."),
    show_text: bool = typer.Option(False, "--show-text", help="Print each gold article."),
) -> None:
    """Check the gold set against the parsed corpus.

    The one failure this must never allow: a question whose gold article does
    not exist. Such a question is unanswerable by construction, so every system
    scores zero on it and the metric silently measures nothing. Cheaper to
    catch here than to explain in a results table.
    """
    import json

    from kora.eval.dataset import load_gold_set

    if not path.exists():
        console.print(f"[red]No gold set at {path}[/]")
        raise typer.Exit(1)

    gold = load_gold_set(path)

    # Index every parsed article so gold ids can be resolved.
    known: dict[str, dict] = {}
    for articles_path in sorted(paths.INTERIM_DIR.glob("*.articles.jsonl")):
        document_id = articles_path.name.removesuffix(".articles.jsonl")
        with articles_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                article = json.loads(line)
                known[f"{document_id}#art{article['number']}"] = article

    if not known:
        console.print("[red]No parsed articles found. Run `kora corpus parse` first.[/]")
        raise typer.Exit(1)

    dangling: list[tuple[str, str]] = []
    for question in gold:
        for chunk_id in question.gold_chunk_ids:
            if chunk_id not in known:
                dangling.append((question.id, chunk_id))

    table = Table(title="Gold set composition")
    table.add_column("property", style="bold")
    table.add_column("value", justify="right")
    for key, value in gold.composition().items():
        table.add_row(key, str(value))
    console.print(table)

    console.print(f"\ncorpus articles indexed: {len(known)}")
    console.print(f"documents referenced: {', '.join(sorted(gold.referenced_documents()))}")

    if dangling:
        console.print(f"\n[red]{len(dangling)} gold article(s) do not exist:[/]")
        for question_id, chunk_id in dangling:
            console.print(f"  {question_id} -> {chunk_id}")
        raise typer.Exit(1)

    console.print("\n[green]All gold articles resolve to parsed articles.[/]")

    if show_text:
        for q in gold:
            console.print(f"\n[bold]{q.id}[/] [dim]({q.kind}, {q.provenance})[/]")
            console.print(f"  Q: {q.question}")
            for chunk_id in q.gold_chunk_ids:
                article = known[chunk_id]
                body = " ".join(article["text"].split())[:220]
                console.print(f"  [cyan]{chunk_id}[/] {body}...")


@eval_app.command("review")
def eval_review(
    kind: str = typer.Option(None, "--kind", help="Filter by question kind."),
    start: int = typer.Option(0, "--start", help="Skip the first N questions."),
    count: int = typer.Option(10, "--count", help="How many to display."),
    path: Path = typer.Option(paths.EVAL_DIR / "gold_qa.jsonl", "--path"),
) -> None:
    """Print questions beside the full text of their gold articles.

    The review loop: read the article, decide whether it truly answers the
    question, then promote the ones that do. Anything not promoted stays out of
    every reported number, so an unreviewed question costs nothing but an
    incorrectly promoted one costs the credibility of the whole table.
    """
    import json

    from kora.eval.dataset import load_gold_set

    gold = load_gold_set(path)
    questions = [q for q in gold if not kind or q.kind == kind]

    known: dict[str, dict] = {}
    for articles_path in sorted(paths.INTERIM_DIR.glob("*.articles.jsonl")):
        document_id = articles_path.name.removesuffix(".articles.jsonl")
        with articles_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                article = json.loads(line)
                known[f"{document_id}#art{article['number']}"] = article

    window = questions[start : start + count]
    for question in window:
        console.print(f"\n[bold yellow]{'─' * 72}[/]")
        mark = "[green]validated[/]" if question.counts_towards_headline else "[yellow]draft[/]"
        console.print(f"[bold]{question.id}[/]  [dim]{question.kind}[/]  {mark}")
        console.print(f"\n  [bold]Q[/] {question.question}")
        if question.reference_answer:
            console.print(f"  [bold]A[/] [dim]{question.reference_answer}[/]")
        if question.note:
            console.print(f"  [italic dim]note: {question.note}[/]")

        if not question.gold_chunk_ids:
            console.print("\n  [dim](unanswerable - no gold articles by design)[/]")
            continue
        for chunk_id in question.gold_chunk_ids:
            article = known.get(chunk_id)
            console.print(f"\n  [cyan]{chunk_id}[/]")
            if article is None:
                console.print("    [red]MISSING FROM CORPUS[/]")
                continue
            if article["rubric"]:
                console.print(f"    [italic]{article['rubric']}[/]")
            console.print(f"    {' '.join(article['text'].split())[:700]}")

    console.print(
        f"\n[bold]Showed {len(window)} of {len(questions)}"
        f"{f' ({kind})' if kind else ''}.[/]"
        f"  Next: --start {start + count}"
    )


@eval_app.command("promote")
def eval_promote(
    question_ids: list[str] = typer.Argument(..., help="Question ids you have verified."),
    path: Path = typer.Option(paths.EVAL_DIR / "gold_qa.jsonl", "--path"),
    demote: bool = typer.Option(False, "--demote", help="Send back to draft instead."),
) -> None:
    """Mark reviewed questions as human-validated, making them count.

    Only promoted questions appear in reported results. This is the single
    gate between "a model wrote a plausible question" and "a person checked it
    against the article", and it is deliberately a separate, explicit action.
    """
    import json

    from kora.eval.dataset import load_gold_set

    gold = load_gold_set(path)
    target = "llm_drafted" if demote else "llm_drafted_human_validated"
    wanted = set(question_ids)

    unknown = wanted - {q.id for q in gold}
    if unknown:
        console.print(f"[red]Unknown question ids: {sorted(unknown)}[/]")
        raise typer.Exit(1)

    changed = 0
    lines: list[str] = []
    for question in gold:
        payload = question.model_dump(mode="json")
        if question.id in wanted and payload["provenance"] != target:
            # Human-written questions are never relabelled; there is nothing to
            # upgrade and overwriting the record would lose real provenance.
            if payload["provenance"] == "human":
                console.print(f"[dim]{question.id} is human-written, left unchanged[/]")
            else:
                payload["provenance"] = target
                changed += 1
        lines.append(json.dumps(payload, ensure_ascii=False))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    reloaded = load_gold_set(path)
    verb = "demoted" if demote else "promoted"
    console.print(f"[green]{verb} {changed} question(s).[/]")
    console.print(f"headline-eligible: {len(reloaded.headline())} / {len(reloaded)}")


index_app = typer.Typer(help="Build and inspect the retrieval index.", no_args_is_help=True)
app.add_typer(index_app, name="index")


@index_app.command("build")
def index_build(
    config: Path = typer.Option(..., "--config", "-c", exists=True, help="YAML config."),
    force: bool = typer.Option(False, "--force", help="Rebuild even if present."),
) -> None:
    """Encode the corpus and store the index under its fingerprint."""
    from kora.retrieval.index import build_index, index_dir, index_exists, load_articles

    cfg = load_config(config)

    if index_exists(cfg) and not force:
        console.print(
            f"[green]Index already present[/] for fingerprint "
            f"[bold]{cfg.index_fingerprint()}[/]\n  {index_dir(cfg)}\n"
            "[dim]Another experiment already built it. Use --force to rebuild.[/]"
        )
        return

    articles = load_articles()
    if not articles:
        console.print("[red]No parsed articles. Run `kora corpus parse` first.[/]")
        raise typer.Exit(1)

    repealed = sum(1 for a in articles if a.repealed or a.status == "superseded")
    console.print(
        f"Indexing [bold]{len(articles)}[/] articles "
        f"([yellow]{repealed}[/] repealed, kept deliberately as distractors)\n"
        f"model: [bold]{cfg.embedding.model_name}[/]"
    )

    index = build_index(cfg, articles)
    console.print(f"\n[green]Built[/] {len(index)} vectors, dim {index.dim}\n  {index_dir(cfg)}")


@eval_app.command("run")
def eval_run(
    config: Path = typer.Option(..., "--config", "-c", exists=True, help="YAML config."),
    gold_path: Path = typer.Option(paths.EVAL_DIR / "gold_qa.jsonl", "--gold"),
    generate: bool = typer.Option(False, "--generate", help="Also generate and score answers."),
    limit: int = typer.Option(0, "--limit", help="Only the first N questions (smoke tests)."),
) -> None:
    """Run one retrieval experiment and write its results.

    Metrics are reported twice: over validated questions (the headline) and over
    all questions (diagnostic). Only the first may be quoted, and both are
    printed with their sample size, because a recall figure without an n behind
    it is not a result.
    """
    from kora.eval.dataset import load_gold_set
    from kora.eval.runner import run_retrieval_experiment, write_run
    from kora.retrieval.index import index_exists, load_index
    from kora.retrieval.pipeline import build_retriever, describe

    cfg = load_config(config)
    if not index_exists(cfg):
        console.print(
            f"[red]No index for fingerprint {cfg.index_fingerprint()}.[/]\n"
            f"Run: kora index build --config {config}"
        )
        raise typer.Exit(1)

    gold = load_gold_set(gold_path)
    if limit:
        from kora.eval.dataset import GoldSet

        gold = GoldSet(version=gold.version, questions=gold.questions[:limit])
        console.print(f"[yellow]Limited to the first {len(gold)} questions.[/]")

    index, articles = load_index(cfg)

    # One generator, shared. The agent needs a model even when answers are not
    # being generated -- decomposition is itself a generation -- so it is built
    # whenever either the agent or `--generate` asks for one, and handed to
    # both. Two instances would not fit in 8 GiB alongside the encoders.
    model = None
    if generate or cfg.agent.enabled:
        from kora.generation.generator import Generator

        model = Generator(cfg)

    retriever = build_retriever(cfg, index, articles, generator=model)
    generator = model if generate else None

    report, results = run_retrieval_experiment(
        cfg, gold, retriever, corpus_size=len(articles), generator=generator
    )
    # A limited run evaluates a different thing from the full sweep, so it must
    # not land on the full sweep's results. The config hash cannot express this:
    # it describes the system, not how much of the gold set was used.
    destination = write_run(cfg, report, results, suffix=f"-limit{limit}" if limit else "")

    console.print(f"\n[bold]{cfg.name}[/]  [dim]{cfg.run_id}[/]  [cyan]{describe(cfg)}[/]")

    table = Table(title="Retrieval metrics")
    table.add_column("metric", style="bold")
    table.add_column("validated", justify="right")
    table.add_column("all questions", justify="right")

    headline, everything = report["headline"], report["all_questions"]
    keys = [k for k in everything if k != "n"]
    table.add_row(
        "[dim]n[/]",
        f"[dim]{int(headline.get('n', 0))}[/]",
        f"[dim]{int(everything.get('n', 0))}[/]",
    )

    def fmt(value: float, metric: str) -> str:
        # Exact search over 3k vectors is sub-millisecond, so whole milliseconds
        # round to zero and report nothing. Latency still belongs in the table:
        # it is the denominator for every accuracy gain Phase 3 buys.
        return f"{value:.3f}" if "latency" not in metric else f"{value:.3f} ms"

    for key in keys:
        left = headline.get(key)
        table.add_row(
            key,
            fmt(left, key) if left is not None else "[dim]--[/]",
            fmt(everything[key], key),
        )
    console.print(table)

    if report["by_kind"]:
        by_kind = Table(title=f"recall@{cfg.retrieval.final_k} by question kind (all questions)")
        by_kind.add_column("kind", style="bold")
        by_kind.add_column("n", justify="right")
        by_kind.add_column(f"recall@{cfg.retrieval.final_k}", justify="right")
        by_kind.add_column("mrr", justify="right")
        for kind, metrics in report["by_kind"].items():
            if not metrics:
                by_kind.add_row(kind, "0", "[dim]n/a[/]", "[dim]n/a[/]")
                continue
            by_kind.add_row(
                kind,
                str(int(metrics["n"])),
                f"{metrics[f'recall@{cfg.retrieval.final_k}']:.3f}",
                f"{metrics['mrr']:.3f}",
            )
        console.print(by_kind)

    if report.get("answers"):
        answers = Table(title="Answer metrics")
        answers.add_column("metric", style="bold")
        answers.add_column("value", justify="right")
        for key, value in report["answers"].items():
            if key == "n":
                answers.add_row("[dim]n[/]", f"[dim]{int(value)}[/]")
            elif "latency" in key:
                answers.add_row(key, f"{value:.0f} ms")
            elif value != value:  # NaN: the set could not test this
                answers.add_row(key, "[dim]n/a[/]")
            else:
                answers.add_row(key, f"{value:.3f}")
        console.print(answers)

    if "warning" in report:
        console.print(f"\n[yellow]{report['warning']}[/]")
    console.print(f"\nWritten to [dim]{destination}[/]")


train_app = typer.Typer(help="QLoRA fine-tuning.", no_args_is_help=True)
app.add_typer(train_app, name="train")


@train_app.command("build-data")
def train_build_data(
    gold_path: Path = typer.Option(paths.EVAL_DIR / "gold_qa.jsonl", "--gold"),
    out: Path = typer.Option(paths.DATA_DIR / "training" / "sft.jsonl", "--out"),
    n_cite: int = typer.Option(600, help="Examples teaching the citation format."),
    n_abstain: int = typer.Option(150, help="Examples teaching abstention."),
    n_repealed: int = typer.Option(100, help="Examples teaching repealed-law avoidance."),
    seed: int = typer.Option(42),
) -> None:
    """Build instruction data, excluding every article used in evaluation."""
    from kora.eval.dataset import load_gold_set
    from kora.retrieval.index import load_articles
    from kora.training.dataset import build_dataset, save_dataset

    articles = load_articles()
    if not articles:
        console.print("[red]No parsed articles. Run `kora corpus parse` first.[/]")
        raise typer.Exit(1)

    gold = load_gold_set(gold_path)
    examples = build_dataset(
        articles,
        gold,
        n_cite=n_cite,
        n_abstain=n_abstain,
        n_repealed=n_repealed,
        seed=seed,
    )
    path = save_dataset(examples, out)

    table = Table(title="Training data")
    table.add_column("kind", style="bold")
    table.add_column("examples", justify="right")
    for kind in ("cite", "abstain", "repealed"):
        table.add_row(kind, str(sum(1 for e in examples if e.kind == kind)))
    table.add_row("[bold]total[/]", f"[bold]{len(examples)}[/]")
    console.print(table)

    gold_articles = {cid for q in gold for cid in q.gold_chunk_ids}
    console.print(
        f"\nExcluded [bold]{len(gold_articles)}[/] gold articles from "
        f"{len(articles)} available.\nWritten to [dim]{path}[/]"
    )


@train_app.command("run")
def train_run(
    config: Path = typer.Option(Path("configs/training/qlora.yaml"), "--config", "-c", exists=True),
) -> None:
    """Fine-tune and write an adapter."""
    from kora.training.config import load_training_config
    from kora.training.train import train

    cfg = load_training_config(config)

    table = Table(show_header=False, box=None)
    table.add_row("[bold]run[/]", cfg.run_id)
    table.add_row("[bold]base model[/]", cfg.base_model)
    table.add_row("[bold]lora[/]", f"r={cfg.lora.r} alpha={cfg.lora.alpha}")
    table.add_row("[bold]modules[/]", ", ".join(cfg.lora.target_modules))
    table.add_row("[bold]effective batch[/]", str(cfg.effective_batch_size))
    table.add_row("[bold]max length[/]", str(cfg.max_length))
    console.print(table)

    adapter = train(cfg)
    console.print(f"\n[green]Adapter written to[/] {adapter}")
    console.print(
        "\nEvaluate it by pointing an experiment at it:\n"
        f"  [dim]generator.adapter_path: {adapter}[/]"
    )


serve_app = typer.Typer(help="HTTP service and serving benchmarks.", no_args_is_help=True)
app.add_typer(serve_app, name="serve")


@serve_app.command("start")
def serve_start(
    config: Path = typer.Option(
        Path("configs/experiments/07_rerank_fast.yaml"), "--config", "-c", exists=True
    ),
    host: str = typer.Option("127.0.0.1", help="Bind address. Use 0.0.0.0 in a container."),
    port: int = typer.Option(8000),
    preload: bool = typer.Option(True, help="Load the generator at startup, not on first request."),
) -> None:
    """Serve one experiment over HTTP, with a streaming UI at /.

    The config is passed through the environment rather than to the app factory,
    because uvicorn imports the module itself. It is also the reason the served
    config cannot be a request parameter: the service has one identity, and
    /health reports it.
    """
    import os

    import uvicorn

    from kora.config import load_config

    cfg = load_config(config)
    os.environ["KORA_CONFIG"] = str(config.resolve())
    os.environ["KORA_PRELOAD"] = "1" if preload else "0"

    table = Table(show_header=False, box=None)
    table.add_row("[bold]serving[/]", f"{cfg.name}  [dim]{cfg.run_id}[/]")
    table.add_row("[bold]index[/]", cfg.index_fingerprint())
    table.add_row("[bold]generator[/]", cfg.generator.model_name)
    if cfg.generator.adapter_path:
        table.add_row("[bold]adapter[/]", cfg.generator.adapter_path)
    table.add_row("[bold]url[/]", f"http://{host}:{port}")
    console.print(table)
    if preload:
        console.print("\n[dim]Loading models before accepting requests…[/]")

    uvicorn.run("kora.serving.app:app", host=host, port=port, log_level="info")


@serve_app.command("bench")
def serve_bench(
    url: str = typer.Option("http://127.0.0.1:8000", "--url", help="A running kora service."),
    repeats: int = typer.Option(2, help="Times through the question list per wave."),
    concurrency: list[int] = typer.Option([1, 2, 4], "--concurrency", help="Levels to test."),
    out: Path = typer.Option(None, "--out", help="Where to write the report."),
) -> None:
    """Measure time-to-first-token, total latency and behaviour under concurrency."""
    import json

    from kora.serving.bench import run_benchmark

    try:
        report = run_benchmark(
            url, repeats=repeats, concurrencies=tuple(concurrency), timeout=300.0
        )
    except Exception as exc:
        console.print(f"[red]Benchmark failed:[/] {exc}\n[dim]Is `kora serve start` running?[/]")
        raise typer.Exit(1) from exc

    generation = Table(title="Generation (/ask/stream)")
    generation.add_column("concurrency", style="bold")
    generation.add_column("n", justify="right")
    generation.add_column("first token", justify="right")
    generation.add_column("total median", justify="right")
    generation.add_column("total p90", justify="right")
    generation.add_column("answers/min", justify="right")
    for level, row in report["generation"].items():
        generation.add_row(
            level.lstrip("c"),
            str(row.get("n", 0)),
            f"{row.get('first_token_ms_median', 0) / 1000:.1f} s",
            f"{row.get('total_ms_median', 0) / 1000:.1f} s",
            f"{row.get('total_ms_p90', 0) / 1000:.1f} s",
            f"{row.get('throughput_per_min', 0):.1f}",
        )
    console.print(generation)

    retrieval = Table(title="Retrieval only (/search)")
    retrieval.add_column("concurrency", style="bold")
    retrieval.add_column("n", justify="right")
    retrieval.add_column("median", justify="right")
    retrieval.add_column("p90", justify="right")
    retrieval.add_column("req/s", justify="right")
    for level, row in report["retrieval"].items():
        retrieval.add_row(
            level.lstrip("c"),
            str(row.get("n", 0)),
            f"{row.get('total_ms_median', 0):.0f} ms",
            f"{row.get('total_ms_p90', 0):.0f} ms",
            f"{row.get('throughput_per_sec', 0):.1f}",
        )
    console.print(retrieval)

    # Reported per concurrency level rather than only at c=1. The single-client
    # figure is large and flattering, and it collapses under load because
    # time-to-first-token becomes queue wait rather than prefill. Printing only
    # the best row would be the serving equivalent of quoting a latency win from
    # a model that stopped answering the question.
    speedups = [
        (level.lstrip("c"), row["perceived_speedup"])
        for level, row in report["generation"].items()
        if row.get("perceived_speedup")
    ]
    if speedups:
        rendered = ", ".join(f"c={level} [bold]{value}x[/]" for level, value in speedups)
        console.print(f"\nFirst token arrives sooner than the full answer by: {rendered}")
        console.print(
            "[dim]Streaming does not change total time. Under concurrency the first token\n"
            "waits for the generation lock, so the gain shrinks as clients are added.[/]"
        )

    destination = out or paths.RUNS_DIR / f"serving-{report['health'].get('run_id', 'unknown')}"
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "bench.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"\nWritten to [dim]{path}[/]")


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
