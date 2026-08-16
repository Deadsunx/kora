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
`assistant_only_loss` restricts the loss to assistant turns. Without it the model
is trained to reproduce the *context* -- five OHADA articles it was given, which
it should never generate -- and most of the gradient signal goes into memorising
passages rather than learning the answer contract. This is the single setting
most likely to be wrong in a chat fine-tune and the least likely to announce
itself, since training loss looks healthy either way.
"""

from __future__ import annotations

import json
from pathlib import Path

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

    dataset = Dataset.from_list([{"messages": r["messages"]} for r in records])
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
        # Loss on the assistant turn only. See the module docstring: without
        # this the model is trained to reproduce the retrieved articles.
        assistant_only_loss=True,
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
