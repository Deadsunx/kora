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


if __name__ == "__main__":
    app()
