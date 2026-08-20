"""Hugging Face Space entry point. The UI lives in `kora.serving.demo`."""

from kora.serving.demo import build_demo

if __name__ == "__main__":
    build_demo().queue(max_size=8).launch()
