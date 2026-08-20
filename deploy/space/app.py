"""Gradio demo for Kora, deployable to Hugging Face Spaces.

Wraps `kora.serving.engine.Engine` rather than reimplementing the pipeline, so
the demo runs the same system the ablation table measured. If this file built
its own retriever, the numbers in the report would stop describing what a
visitor actually touches.

Two things differ from `kora serve start`, and both are stated in the UI:

- **Quantisation is configurable by environment.** The measured system is 4-bit
  NF4, which needs bitsandbytes and a CUDA device at load time. Hosts that
  attach a GPU only for the duration of a call cannot do that, so `KORA_QUANT`
  can select fp16 instead. Same weights, different precision.
- **Retrieval-only mode.** On a CPU box the cross-encoder takes 22 seconds and
  the generator takes minutes, so `KORA_RETRIEVAL_ONLY=1` serves the search half
  alone. It is honest about being a reduced system rather than pretending.
"""

from __future__ import annotations

import os
import time

import gradio as gr

# ZeroGPU exposes a decorator that borrows a GPU for the duration of a call.
# Off-platform the import fails, so the decorator degrades to a no-op and the
# app still runs locally.
try:  # pragma: no cover - depends on the host
    import spaces

    def gpu(fn):
        return spaces.GPU(duration=120)(fn)
except ImportError:  # pragma: no cover
    def gpu(fn):
        return fn


RETRIEVAL_ONLY = os.environ.get("KORA_RETRIEVAL_ONLY", "0") == "1"
CONFIG = os.environ.get("KORA_CONFIG", "configs/experiments/07_rerank_fast.yaml")

EXAMPLES = [
    "Quelle est la durée maximale de vie d'une société commerciale ?",
    "Qu'est-ce qu'une hypothèque ?",
    "Quel est le capital social minimum d'une société anonyme ?",
    "Qui peut requérir l'inscription d'une sûreté mobilière au RCCM ?",
    "Quelle est la peine encourue pour abus de biens sociaux ?",
]

_engine = None


def ensure_data() -> None:
    """Pull the index and parsed articles from a private dataset repo.

    The Space is public and the Actes uniformes are not redistributed, so the
    corpus cannot live in the Space repository. It is downloaded at start-up
    instead, using the HF_TOKEN secret, and `KORA_DATA_DIR` points the rest of
    the project at it. Locally the variable is unset and the existing data/
    directory is used unchanged.
    """
    repo = os.environ.get("KORA_DATASET_REPO")
    if not repo:
        return

    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
    )
    os.environ["KORA_DATA_DIR"] = path




def engine():
    """Load once, on first use."""
    global _engine
    if _engine is None:
        ensure_data()

        from kora.config import load_config
        from kora.serving.engine import Engine

        cfg = load_config(CONFIG)
        quant = os.environ.get("KORA_QUANT")
        if quant:
            cfg = cfg.model_copy(
                update={"generator": cfg.generator.model_copy(update={"quantization": quant})}
            )
        # Preloading is wrong here: a retrieval-only demo must never pay for the
        # generator, and on a borrowed GPU the warm-up would run without one.
        _engine = Engine(cfg, preload_generator=False)
    return _engine


def format_passages(passages) -> str:
    blocks = []
    for i, p in enumerate(passages, start=1):
        warn = " · ⚠️ **TEXTE ABROGÉ**" if p.unsafe else ""
        rubric = f"\n*{p.rubric}*" if p.rubric else ""
        body = p.text if len(p.text) < 700 else p.text[:700].rsplit(" ", 1)[0] + " […]"
        blocks.append(f"**[{i}] {p.citation}**  `{p.score:.3f}`{warn}{rubric}\n\n{body}")
    return "\n\n---\n\n".join(blocks) or "_Aucun article retrouvé._"


@gpu
def ask(question: str):
    """Retrieve, then stream an answer. Yields (answer, passages, status)."""
    question = (question or "").strip()
    if not question:
        yield "", "", "Posez une question."
        return

    started = time.perf_counter()
    eng = engine()

    search = eng.search(question)
    passages = format_passages(search.passages)
    status = f"Recherche : {search.retrieval_ms:.0f} ms · {len(search.passages)} articles"

    if RETRIEVAL_ONLY:
        yield (
            "_Mode recherche seule : ce déploiement ne génère pas de réponse._",
            passages,
            status,
        )
        return

    yield "", passages, status + " · génération…"

    parts: list[str] = []
    first_token_ms = None
    for chunk in eng.stream(question):
        if chunk.kind == "token":
            if first_token_ms is None:
                first_token_ms = (time.perf_counter() - started) * 1000
            parts.append(chunk.data["text"])
            yield "".join(parts), passages, status + " · génération…"
        elif chunk.kind == "done":
            d = chunk.data
            marks = []
            if d["abstained"]:
                marks.append("a décliné (INSUFFISANT)")
            else:
                marks.append(f"{len(d['citations'])} citation(s)")
            if d["citations_unretrieved"]:
                marks.append(f"⚠️ {len(d['citations_unretrieved'])} hors contexte")
            if d["citations_repealed"]:
                marks.append(f"⚠️ {len(d['citations_repealed'])} abrogé cité")
            yield (
                d["answer"],
                format_passages(search.passages),
                f"{status.replace(' · génération…', '')} · 1er token "
                f"{first_token_ms or 0:.0f} ms · total {d['generation_ms'] / 1000:.1f} s · "
                + " · ".join(marks),
            )


with gr.Blocks(title="Kora — droit OHADA", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Kora\n"
        "**Questions-réponses sur les Actes uniformes OHADA**, le droit des "
        "affaires partagé par 17 États africains.\n\n"
        "Chaque réponse cite les articles dont elle provient, ou refuse de "
        "répondre. Les extraits marqués **ABROGÉ** sont indexés délibérément : "
        "le système doit les voir et refuser de s'appuyer dessus."
    )

    with gr.Row():
        question = gr.Textbox(
            label="Votre question",
            placeholder=EXAMPLES[0],
            lines=2,
            scale=4,
        )
        submit = gr.Button("Demander", variant="primary", scale=1)

    status = gr.Markdown("")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Réponse")
            answer = gr.Markdown("")
        with gr.Column(scale=1):
            gr.Markdown("### Articles retrouvés")
            passages = gr.Markdown("")

    gr.Examples(examples=EXAMPLES, inputs=question)

    gr.Markdown(
        "---\n"
        "Ce service n'est pas un conseil juridique. "
        "[Rapport technique](https://deadsunx.github.io/kora/) · "
        "[Code](https://github.com/Deadsunx/kora)"
    )

    submit.click(ask, inputs=question, outputs=[answer, passages, status])
    question.submit(ask, inputs=question, outputs=[answer, passages, status])


if __name__ == "__main__":
    demo.queue(max_size=8).launch()
