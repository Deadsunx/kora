"""Load a generator and produce grounded answers.

Fitting three models in 8 GiB
-----------------------------
An evaluation run holds the embedder (~1.1 GiB), the cross-encoder (~1.1 GiB at
fp16) and the generator simultaneously. A 4B model in 4-bit NF4 is roughly
2.5 GiB of weights, leaving a few hundred megabytes for activations and the KV
cache once everything else is resident.

That is workable but not comfortable, which is why quantisation is a config
field rather than a constant: on a larger GPU the same experiment should be
runnable at 8-bit or full precision to check that 4-bit is not itself costing
accuracy. Assuming quantisation is free is exactly the sort of thing this
project measures instead.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from threading import Thread

from kora.config import ExperimentConfig
from kora.documents import Article
from kora.generation.prompt import (
    Citation,
    build_messages,
    extract_citations,
    is_abstention,
)
from kora.logging import get_logger
from kora.retrieval.index import _resolve_device

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    """One generated answer with everything needed to score it."""

    text: str
    citations: tuple[Citation, ...]
    abstained: bool
    prompt_tokens: int
    generated_tokens: int

    @property
    def cited_chunk_ids(self) -> tuple[str, ...]:
        return tuple(c.chunk_id for c in self.citations)


class Generator:
    """Wraps a causal LM and turns retrieved articles into cited answers."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return self._model, self._tokenizer

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        generator = self.config.generator
        device = _resolve_device(self.config.embedding.device)

        kwargs: dict = {"dtype": torch.float16 if device == "cuda" else torch.float32}
        if generator.quantization != "none" and device == "cuda":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=generator.quantization == "4bit",
                load_in_8bit=generator.quantization == "8bit",
                # NF4 with double quantisation: the standard QLoRA recipe.
                # bfloat16 compute rather than float16 because the RTX 4070
                # supports it and it is markedly less prone to overflow.
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            kwargs["device_map"] = {"": 0}

        log.info(
            "loading generator",
            model=generator.model_name,
            quantization=generator.quantization,
            device=device,
        )
        tokenizer = AutoTokenizer.from_pretrained(generator.model_name)
        model = AutoModelForCausalLM.from_pretrained(generator.model_name, **kwargs)

        if generator.adapter_path:
            from peft import PeftModel

            log.info("attaching adapter", path=generator.adapter_path)
            model = PeftModel.from_pretrained(model, generator.adapter_path)

        if "device_map" not in kwargs:
            model = model.to(device)
        model.eval()

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        self._model, self._tokenizer = model, tokenizer
        return model, tokenizer

    def chat(self, messages: list[dict[str, str]], *, max_new_tokens: int | None = None) -> str:
        """Run the model on arbitrary messages and return its reply.

        The agent needs the loaded model for questions that are not "answer
        this" — decomposing a query, judging whether passages suffice. Exposing
        `chat` rather than giving the agent its own model instance is not a
        convenience: a second model would not fit. The embedder, cross-encoder
        and a 4-bit 4B generator already occupy most of 8 GiB.

        `max_new_tokens` is separate from the generator's because these replies
        are a line or two, and generation time is linear in output length.
        """
        import torch

        model, tokenizer = self._load()
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.config.agent.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = output[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _prepare(self, question: str, articles: list[Article]):
        """Tokenise one question into model inputs and decoding settings.

        Shared by `answer` and `stream` so that a streamed response and a batch
        response are the same computation. If these drifted apart, the served
        system would stop being the system the ablation table measured.
        """
        model, tokenizer = self._load()
        messages = build_messages(question, articles, self.config)

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            # Qwen3 emits a reasoning block unless this is disabled. Useful in
            # general, but it inflates latency several-fold and its content is
            # not part of what is being measured here.
            enable_thinking=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        generation_kwargs: dict = {
            "max_new_tokens": self.config.generator.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if self.config.generator.temperature > 0:
            generation_kwargs |= {
                "do_sample": True,
                "temperature": self.config.generator.temperature,
            }
        else:
            # Greedy. Every experiment must be reproducible from its config, and
            # sampling would make two runs of the same config disagree.
            generation_kwargs["do_sample"] = False

        return model, tokenizer, inputs, generation_kwargs

    def answer(self, question: str, articles: list[Article]) -> GeneratedAnswer:
        """Generate one grounded answer."""
        import torch

        model, tokenizer, inputs, generation_kwargs = self._prepare(question, articles)

        with torch.inference_mode():
            output = model.generate(**inputs, **generation_kwargs)

        generated = output[0][inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()

        return GeneratedAnswer(
            text=text,
            citations=tuple(extract_citations(text)),
            abstained=is_abstention(text),
            prompt_tokens=int(inputs["input_ids"].shape[1]),
            generated_tokens=int(generated.shape[0]),
        )

    def stream(self, question: str, articles: list[Article]) -> Iterator[str]:
        """Yield the answer in fragments as it is decoded.

        Same prompt, same decoding settings and — at temperature 0 — the same
        tokens as `answer`. Streaming changes *when* the user sees the text, not
        what it is: total latency is unchanged, and Phase 4 established that
        total latency here is essentially linear in output length. What streaming
        buys is time-to-first-token, which `kora serve bench` measures separately
        for exactly that reason.

        `generate` blocks, so it runs on a worker thread and the streamer is
        drained from this one. The thread is non-daemon and joined at the end,
        so a client disconnecting mid-answer cannot leave a generation running
        against the GPU forever.
        """
        import torch
        from transformers import TextIteratorStreamer

        model, tokenizer, inputs, generation_kwargs = self._prepare(question, articles)

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        def run() -> None:
            with torch.inference_mode():
                model.generate(**inputs, **generation_kwargs, streamer=streamer)

        thread = Thread(target=run, name="kora-generate")
        thread.start()
        try:
            yield from streamer
        finally:
            thread.join()
