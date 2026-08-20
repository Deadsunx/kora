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

import subprocess

import modal


def deployed_commit() -> str:
    """The commit this deploy pins the image to.

    Read from the local checkout at deploy time. Two reasons, and the second is
    the one that bites: pinning states exactly which version of the project is
    serving, the same discipline the run fingerprints follow; and it busts
    Modal's layer cache, which otherwise keeps a stale `git clone` forever and
    silently deploys old code after a push.
    """
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


COMMIT = deployed_commit()

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
        "git clone https://github.com/Deadsunx/kora.git /opt/kora",
        f"cd /opt/kora && git checkout {COMMIT}",
        "pip install -e /opt/kora",
    )
    .env(
        {
            # HF_HOME is on the volume, so both the model weights and the
            # corpus snapshot persist across cold starts.
            "HF_HOME": "/cache/huggingface",
            # KORA_DATA_DIR is deliberately NOT set: `ensure_data()` treats it
            # as "the corpus is already here" and skips the download. Leaving it
            # unset lets that function fetch the private dataset and point
            # KORA_DATA_DIR at the snapshot itself.
            "KORA_DATASET_REPO": "Deadsunx/kora-corpus",
            "KORA_CONFIG": "/opt/kora/configs/experiments/07_rerank_fast.yaml",
            # Load the models when the container starts rather than inside the
            # first request. Modal holds traffic until the ASGI app is returned,
            # so the cost lands on the cold start where it belongs instead of
            # timing out somebody's first question.
            "KORA_PRELOAD": "1",
            "KORA_COMMIT": COMMIT,
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
    # Gradio keeps its queue in process memory, so a session that starts on one
    # container cannot be served by another: the browser POSTs to
    # /gradio_api/queue/join, then opens /gradio_api/queue/data, and a second
    # container answers the stream with 404 because it has never seen that
    # session. That was the real cause of the demo hanging, and no amount of
    # preloading or concurrency tuning could touch it.
    #
    # One container costs nothing here. The GPU serialises generation anyway,
    # and Phase 6 measured throughput as flat at 2.8 answers/min regardless of
    # how many callers arrive.
    max_containers=1,
)
# Counts concurrent *connections*, not GPU work, and Gradio holds several
# long-lived ones per browser tab: the queue SSE stream and a heartbeat, before
# any question is asked. Setting this to 2 deadlocked the app -- one tab
# consumed both slots and the request that would have produced an answer had
# nowhere to run.
#
# Serialising the GPU is already handled a layer down, by the generation lock in
# Engine, which is what the Phase 6 benchmark measured. This number only has to
# be large enough for the transport.
@modal.concurrent(max_inputs=20)
# `web_server` rather than `asgi_app`, and the reason is specific.
#
# Gradio's queue runs two long-lived asyncio tasks, start_processing and
# start_progress_updates, spawned when its app starts. Mounting the Blocks into
# a FastAPI app under @modal.asgi_app() does not keep an event loop alive
# between requests, so both tasks were destroyed the moment they were created --
# the logs read "Task was destroyed but it is pending!" on a loop -- and no
# submitted job was ever processed.
#
# Letting Gradio run its own uvicorn keeps its lifespan, and therefore its
# queue, intact. Modal proxies to the port.
@modal.web_server(port=8000, startup_timeout=900)
def ui():
    """Start Gradio's own server; Modal proxies to it."""
    from kora.serving.demo import build_demo, engine

    # Load before the port opens. Modal waits for the port during
    # startup_timeout, so the container is only routed traffic once it can
    # answer, and nobody's first question pays for a four-gigabyte download.
    engine()

    build_demo().queue(max_size=8).launch(
        server_name="0.0.0.0",
        server_port=8000,
        # Return control so Modal sees the port open instead of blocking here.
        prevent_thread_lock=True,
        show_api=False,
    )
