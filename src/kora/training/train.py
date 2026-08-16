"""QLoRA fine-tuning on a single 8 GiB GPU.

The memory argument
-------------------
A 4B model cannot be fully fine-tuned on 8 GiB: weights alone in bf16 are ~8 GiB
before any gradient, optimizer state or activation exists. QLoRA makes it fit by
changing what is trained rather than what is loaded:

    base weights      4-bit NF4, frozen          ~2.5 GiB
    LoRA adapters     bf16, trainable            ~60 MiB
    gradients         adapters only              ~60 MiB
    optimizer state   8-bit paged AdamW          ~120 MiB
    activations       one sequence, checkpointed ~1-2 GiB

The base model is quantised but never updated; gradients flow *through* the
frozen 4-bit weights into the adapters. That is the whole trick, and it is why
`prepare_model_for_kbit_training` matters: it casts layer norms to fp32 and
enables input gradients, without which the backward pass through a quantised
model silently produces nothing useful.

Loss masking
------------
Loss must cover the assistant turn only. Without that the model is trained to
reproduce the *context* -- five OHADA articles it was given and should never
generate -- and most of the gradient goes into memorising passages instead of
learning the answer contract.

The obvious route, `assistant_only_loss=True` on a `messages` dataset, does not
work with this model and fails silently. It relies on the chat template emitting
`{% generation %}` markers, and Qwen3's template has none: asking for the
assistant mask returns **zero** assistant tokens out of 984, with only a warning.
Training would have proceeded with a healthy-looking loss curve and a broken
objective.

So the dataset is built in TRL's prompt/completion form instead. The prompt is
the system and user turns, the completion is the assistant turn, and TRL masks
the prompt structurally rather than by parsing a rendered string. The property
is then a fact about the data layout rather than about a template's contents,
and it is asserted before training starts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kora.logging import get_logger
from kora.paths import MODELS_DIR
from kora.training.config import TrainingConfig

log = get_logger(__name__)


def _load_examples(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"No training data at {path}. Run `kora train build-data` first.")
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _verify_loss_masking(trainer: Any) -> None:
    """Assert that the prompt is masked and the completion is not.

    Checked against a real collated batch rather than against configuration,
    because the failure this guards is precisely a setting that is accepted,
    reported as enabled, and does nothing. An all-masked batch trains on no
    tokens; an unmasked batch trains the model to emit the retrieved articles.
    Both produce a plausible loss curve.
    """
    batch = next(iter(trainer.get_train_dataloader()))
    labels = batch["labels"][0]

    masked = int((labels == -100).sum())
    supervised = int((labels != -100).sum())
    total = int(labels.numel())

    log.info(
        "loss masking",
        masked=masked,
        supervised=supervised,
        supervised_fraction=round(supervised / max(total, 1), 3),
    )

    if supervised == 0:
        raise RuntimeError(
            "Loss mask covers every token: nothing would be trained on. "
            "This is what assistant_only_loss does with a chat template that "
            "lacks {% generation %} markers."
        )
    if masked == 0:
        raise RuntimeError(
            "Nothing is masked: the model would be trained to reproduce the "
            "retrieved articles as well as the answer."
        )
    if supervised / total > 0.5:
        # Prompts here are ~1000 tokens of context and answers are one or two
        # sentences, so anything above half is a sign the split went wrong.
        raise RuntimeError(
            f"{supervised}/{total} tokens supervised. Expected a small fraction; "
            "the prompt/completion split is probably wrong."
        )


def train(config: TrainingConfig) -> Path:
    """Fine-tune and write the adapter. Returns the adapter directory."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig as PeftLoraConfig
    from peft import prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError(
            "QLoRA training requires CUDA. `kora doctor` reports what torch can see."
        )

    output_dir = MODELS_DIR / config.run_id
    records = _load_examples(Path(config.dataset_path))
    log.info("training data", examples=len(records), path=config.dataset_path)

    # Prompt/completion rather than a single `messages` field, so TRL masks the
    # prompt by construction. See the module docstring: assistant_only_loss
    # silently produces an all-zero mask with this model's chat template.
    rows = []
    for record in records:
        messages = record["messages"]
        assistant = [m for m in messages if m["role"] == "assistant"]
        if len(assistant) != 1:
            raise ValueError(f"expected exactly one assistant turn, got {len(assistant)}")
        rows.append(
            {
                "prompt": [m for m in messages if m["role"] != "assistant"],
                "completion": assistant,
            }
        )

    dataset = Dataset.from_list(rows)
    split = dataset.train_test_split(test_size=config.eval_fraction, seed=config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    log.info("loading base model", model=config.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=quantization,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=config.gradient_checkpointing
    )

    peft_config = PeftLoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=list(config.lora.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )

    # This version of TRL/transformers exposes `warmup_steps` but not
    # `warmup_ratio`, so the ratio is converted here. Doing it explicitly also
    # makes the number visible in the log, which a ratio never is.
    steps_per_epoch = max(
        1, len(split["train"]) // (config.batch_size * config.gradient_accumulation)
    )
    total_steps = max(1, int(steps_per_epoch * config.epochs))
    warmup_steps = int(total_steps * config.warmup_ratio)
    log.info(
        "schedule",
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        effective_batch=config.effective_batch_size,
    )

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        gradient_checkpointing=config.gradient_checkpointing,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler,
        warmup_steps=warmup_steps,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        optim=config.optimizer,
        max_length=config.max_length,
        bf16=True,
        # Loss on the completion only. With a prompt/completion dataset TRL
        # masks the prompt structurally, which is why this replaced
        # assistant_only_loss -- that path depends on chat-template markers this
        # model does not have, and fails by masking nothing at all.
        completion_only_loss=True,
        packing=False,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if config.eval_fraction > 0 else "no",
        report_to=[],
        seed=config.seed,
        # Keeps a 3,000-example run from spending minutes on dataloader workers
        # that a single-GPU laptop cannot use.
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=split["train"],
        eval_dataset=split["test"] if config.eval_fraction > 0 else None,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    _verify_loss_masking(trainer)

    trained_model = trainer.model
    assert trained_model is not None, "SFTTrainer did not build a model"

    trainable = sum(int(p.numel()) for p in trained_model.parameters() if p.requires_grad)
    total = sum(int(p.numel()) for p in trained_model.parameters())
    log.info(
        "trainable parameters",
        trainable=f"{trainable / 1e6:.1f}M",
        total=f"{total / 1e6:.0f}M",
        percent=f"{100 * trainable / total:.3f}%",
    )
    if trainable == 0:
        # Silent failure mode: training runs, loss decreases nowhere, and the
        # adapter is empty. Usually means the target module names do not match
        # this architecture.
        raise RuntimeError(
            "No trainable parameters. Check that lora.target_modules match the "
            f"module names of {config.base_model}."
        )

    trainer.train()

    adapter_dir = output_dir / "adapter"
    # `nn.Module.__getattr__` is annotated as returning Tensor, so mypy resolves
    # any attribute it does not know statically -- including PEFT's
    # save_pretrained -- to a tensor call. The method exists at runtime.
    trained_model.save_pretrained(str(adapter_dir))  # type: ignore[operator]
    tokenizer.save_pretrained(str(adapter_dir))

    # The config that produced the adapter, stored beside it. Same reason the
    # resolved experiment config sits beside its metrics: an artefact whose
    # settings are not recorded cannot be reproduced or trusted.
    (output_dir / "training_config.json").write_text(
        json.dumps(config.model_dump(mode="json"), indent=2), encoding="utf-8"
    )

    log.info("adapter written", path=str(adapter_dir))
    return adapter_dir
