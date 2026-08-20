"""Gradio demo UI, shared by every host.

Lives in the package rather than beside a deployment script so that Hugging
Face, Modal and a local run all import the same code. Two copies of a demo
drift, and a demo that drifts stops being the system the ablation table
measured.

It wraps `Engine`, so the pipeline is assembled by the same `build_retriever`
the evaluation harness calls.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator

# ZeroGPU hands a GPU to a decorated function for the duration of a call. The
# import only resolves on that platform; everywhere else the decorator has to
# disappear rather than fail.
try:  # pragma: no cover - platform dependent
    import spaces

    def borrow_gpu(fn):
        return spaces.GPU(duration=120)(fn)
except ImportError:  # pragma: no cover

    def borrow_gpu(fn):
        return fn


EXAMPLES = [
    "Quelle est la durée maximale de vie d'une société commerciale ?",
    "Qu'est-ce qu'une hypothèque ?",
    "Quel est le capital social minimum d'une société anonyme ?",
    "Qui peut requérir l'inscription d'une sûreté mobilière au RCCM ?",
    "Quelle est la peine encourue pour abus de biens sociaux ?",
]

_engine = None


def ensure_data() -> None:
    """Pull the index and articles from a private dataset repo, if configured.

    The corpus is not redistributed, so it cannot ride along in a public Space
    or image. `KORA_DATASET_REPO` names a private dataset, and `KORA_DATA_DIR`
    then points the rest of the project at the download. Unset locally, where
    the real `data/` directory is used.
    """
    repo = os.environ.get("KORA_DATASET_REPO")
    if not repo or os.environ.get("KORA_DATA_DIR"):
        return

    from huggingface_hub import snapshot_download

    os.environ["KORA_DATA_DIR"] = snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
    )


def engine():
    """Load the pipeline once, on first use."""
    global _engine
    if _engine is None:
        ensure_data()

        from kora.config import load_config
        from kora.serving.engine import Engine

        cfg = load_config(os.environ.get("KORA_CONFIG", "configs/experiments/07_rerank_fast.yaml"))
        quant = os.environ.get("KORA_QUANT")
        if quant:
            # 4-bit is the measured system. fp16 exists for hosts where
            # bitsandbytes cannot initialise at load time.
            cfg = cfg.model_copy(
                update={"generator": cfg.generator.model_copy(update={"quantization": quant})}
            )
        _engine = Engine(cfg, preload_generator=False)
    return _engine


def format_passages(passages) -> str:
    blocks = []
    for position, p in enumerate(passages, start=1):
        warn = " · ⚠️ **TEXTE ABROGÉ**" if p.unsafe else ""
        rubric = f"\n*{p.rubric}*" if p.rubric else ""
        body = p.text if len(p.text) < 700 else p.text[:700].rsplit(" ", 1)[0] + " […]"
        blocks.append(f"**[{position}] {p.citation}**  `{p.score:.3f}`{warn}{rubric}\n\n{body}")
    return "\n\n---\n\n".join(blocks) or "_Aucun article retrouvé._"


@borrow_gpu
def ask(question: str) -> Iterator[tuple[str, str, str]]:
    """Retrieve, then stream the answer. Yields (answer, passages, status)."""
    question = (question or "").strip()
    if not question:
        yield "", "", "Posez une question."
        return

    started = time.perf_counter()
    eng = engine()

    search = eng.search(question)
    passages = format_passages(search.passages)
    status = f"Recherche : {search.retrieval_ms:.0f} ms · {len(search.passages)} articles"

    if os.environ.get("KORA_RETRIEVAL_ONLY") == "1":
        yield "_Mode recherche seule : ce déploiement ne génère pas de réponse._", passages, status
        return

    yield "", passages, status + " · génération…"

    parts: list[str] = []
    first_token_ms: float | None = None
    for chunk in eng.stream(question):
        if chunk.kind == "token":
            if first_token_ms is None:
                first_token_ms = (time.perf_counter() - started) * 1000
            parts.append(chunk.data["text"])
            yield "".join(parts), passages, status + " · génération…"
        elif chunk.kind == "done":
            data = chunk.data
            marks = []
            if data["abstained"]:
                marks.append("a décliné (INSUFFISANT)")
            else:
                marks.append(f"{len(data['citations'])} citation(s)")
            # Surfaced, not hidden: an article cited but never shown is
            # fabrication, and the demo should say so when it happens.
            if data["citations_unretrieved"]:
                marks.append(f"⚠️ {len(data['citations_unretrieved'])} hors contexte")
            if data["citations_repealed"]:
                marks.append(f"⚠️ {len(data['citations_repealed'])} abrogé cité")
            yield (
                data["answer"],
                passages,
                f"{status} · 1er token {first_token_ms or 0:.0f} ms · "
                f"total {data['generation_ms'] / 1000:.1f} s · " + " · ".join(marks),
            )


def build_demo():
    """Construct the Gradio interface."""
    import gradio as gr

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
                label="Votre question", placeholder=EXAMPLES[0], lines=2, scale=4
            )
            submit = gr.Button("Demander", variant="primary", scale=1)

        status = gr.Markdown("")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Réponse")
                answer = gr.Markdown("")
            with gr.Column():
                gr.Markdown("### Articles retrouvés")
                passages = gr.Markdown("")

        gr.Examples(examples=EXAMPLES, inputs=question)
        gr.Markdown(
            "---\nCe service n'est pas un conseil juridique. "
            "[Rapport technique](https://deadsunx.github.io/kora/) · "
            "[Code](https://github.com/Deadsunx/kora)"
        )

        submit.click(ask, inputs=question, outputs=[answer, passages, status])
        question.submit(ask, inputs=question, outputs=[answer, passages, status])

    return demo
