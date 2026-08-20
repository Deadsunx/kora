"""Serve the Kora demo on Modal.

    modal setup                        # once, opens a browser
    modal secret create kora-hf HF_TOKEN=...        # read access to the corpus dataset
    modal deploy deploy/modal_app.py

Why Modal rather than a Space: Hugging Face now requires PRO to host any Gradio
Space, and this runs the 4-bit configuration the ablation table measured rather
than an fp16 substitute, because a real GPU is attached for the whole container
lifetime instead of per call.

Three decisions worth stating:

**Scale to zero.** `scaledown_window` lets the container die after five idle
minutes, so an unused demo costs nothing. The price is a cold start, which is
model loading, which is why the cache is a volume.

**The model cache is a Volume.** Qwen3-4B, the embedder and the cross-encoder are
about 6 GB. Downloading them on every cold start would make the first question of
each session take minutes and would waste the free credits on bandwidth.

**The corpus is not baked into the image.** It comes from a private dataset repo
at start-up, the same way the Space would have loaded it, because the Actes
uniformes are not redistributed.
"""

from __future__ import annotations

import modal

app = modal.App("kora")

# The corpus and the model weights, kept across container restarts.
cache = modal.Volume.from_name("kora-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .pip_install(
        # Linux wheels for torch are CUDA-enabled by default, so no custom index.
        "torch",
        "sentence-transformers>=3.0",
        "faiss-cpu>=1.8",
        "rank-bm25>=0.2.2",
        "transformers>=4.44",
        "accelerate>=0.33",
        "peft>=0.12",
        "bitsandbytes>=0.43",
        "gradio>=5.0",
        "fastapi",
        "huggingface_hub>=0.25",
    )
    # Cloned rather than pip-installed from the URL, because `configs/` lives at
    # the repository root and never lands in a wheel. An editable install from
    # the clone also makes `paths.PROJECT_ROOT` resolve to /opt/kora, so config
    # and data paths work exactly as they do locally.
    .run_commands(
        "git clone --depth 1 https://github.com/Deadsunx/kora.git /opt/kora",
        "pip install -e /opt/kora",
    )
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "KORA_DATA_DIR": "/cache/data",
            "KORA_DATASET_REPO": "Deadsunx/kora-corpus",
            "KORA_CONFIG": "/opt/kora/configs/experiments/07_rerank_fast.yaml",
            # Load the models when the container starts rather than inside the
            # first request. Modal holds traffic until the ASGI app is returned,
            # so the cost lands on the cold start where it belongs instead of
            # timing out somebody's first question.
            "KORA_PRELOAD": "1",
        }
    )
)


@app.function(
    image=image,
    gpu="a10g",
    volumes={"/cache": cache},
    secrets=[modal.Secret.from_name("kora-hf")],
    # Model loading dominates a cold start, so keep containers alive between
    # questions rather than paying it repeatedly during a demo session.
    scaledown_window=300,
    timeout=900,
    min_containers=0,
)
# One GPU generates one answer at a time, exactly as the Phase 6 benchmark
# measured. Extra concurrency here would not add throughput, it would only let
# more callers queue inside a single container.
@modal.concurrent(max_inputs=2)
@modal.asgi_app()
def ui():
    """Gradio over ASGI."""
    import os
    import shutil
    from pathlib import Path

    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app
    from huggingface_hub import snapshot_download

    from kora.serving.demo import build_demo

    # Corpus first, before anything tries to load an index.
    data_dir = Path(os.environ["KORA_DATA_DIR"])
    if not data_dir.exists():
        downloaded = snapshot_download(
            repo_id=os.environ["KORA_DATASET_REPO"],
            repo_type="dataset",
            token=os.environ["HF_TOKEN"],
        )
        shutil.copytree(downloaded, data_dir)
    cache.commit()

    # Build the pipeline before handing the app to Modal, so the container is
    # only marked ready once it can actually answer.
    from kora.serving.demo import engine

    engine()

    return mount_gradio_app(app=FastAPI(), blocks=build_demo().queue(max_size=8), path="/")
