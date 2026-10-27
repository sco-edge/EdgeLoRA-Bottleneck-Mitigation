#!/usr/bin/env python3
"""Common FP16 LoRA trainer for all six EdgeLoRA task adapters."""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

TASKS = ("securecode", "math", "ifeval", "bbh", "gpqa", "mmlupro")
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_DATA_ROOT = Path(__file__).parent / "processed" / "six_task_v1_r2"
DEFAULT_OUTPUT_ROOT = Path(__file__).parent / "outputs"
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_CUTOFF_LEN = 512
DEFAULT_SEED = 42


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages")
            if not isinstance(messages, list) or [m.get("role") for m in messages] != ["system", "user", "assistant"]:
                raise ValueError(f"{path}:{line_number}: expected system/user/assistant messages")
            if any(not str(message.get("content", "")).strip() for message in messages):
                raise ValueError(f"{path}:{line_number}: empty message content")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if limit is not None and len(rows) < limit:
        raise ValueError(f"{path}: smoke mode needs {limit} rows, found {len(rows)}")
    return rows


def encode_assistant_only(tokenizer: Any, messages: list[dict[str, str]], cutoff_len: int) -> dict[str, list[int]]:
    """Render a complete chat and mask every token before the assistant answer."""
    if [message.get("role") for message in messages] != ["system", "user", "assistant"]:
        raise ValueError("messages must have system/user/assistant role order")
    # This exact full rendering is the serialized training example required by
    # the pipeline contract.
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prefix_text = tokenizer.apply_chat_template(
        messages[:2], tokenize=False, add_generation_prompt=True
    )
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    if full_ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError("chat template assistant prefix is not a token prefix of the full conversation")
    input_ids = full_ids[:cutoff_len]
    answer_start = min(len(prefix_ids), len(input_ids))
    labels = [-100] * answer_start + input_ids[answer_start:]
    if not any(label != -100 for label in labels):
        raise ValueError("assistant answer was fully truncated by the fixed 512-token cutoff; shorten the prompt")
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def tokenization_details(tokenizer: Any, messages: list[dict[str, str]], cutoff_len: int) -> dict[str, int | bool]:
    """Measure the exact chat serialization used by assistant-only masking."""
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prefix = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    if full_ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError("chat template assistant prefix is not a token prefix of the full conversation")
    assistant_before = len(full_ids) - len(prefix_ids)
    assistant_after = max(0, min(len(full_ids), cutoff_len) - min(len(prefix_ids), cutoff_len))
    truncated = len(full_ids) > cutoff_len
    return {"serialized_tokens":len(full_ids), "assistant_tokens_before":assistant_before,
            "assistant_tokens_after":assistant_after, "truncated":truncated,
            "assistant_fully_lost":assistant_after == 0,
            "assistant_partially_truncated":truncated and 0 < assistant_after < assistant_before}


def tokenization_statistics(tokenizer: Any, rows: Iterable[dict[str, Any]], cutoff_len: int) -> dict[str, Any]:
    """Fail before training if truncation removes any complete assistant answer."""
    input_lengths, assistant_lengths, truncations, full_losses = [], [], 0, 0
    for row in rows:
        details = tokenization_details(tokenizer, row["messages"], cutoff_len)
        input_lengths.append(int(details["serialized_tokens"]))
        assistant_lengths.append(int(details["assistant_tokens_before"]))
        truncations += bool(details["truncated"])
        full_losses += bool(details["assistant_fully_lost"])
    if full_losses:
        raise ValueError(f"{full_losses} rows lose the entire assistant answer at cutoff {cutoff_len}")

    def summary(values: list[int]) -> dict[str, float | int]:
        ordered = sorted(values)
        def percentile(fraction: float) -> float:
            position = (len(ordered) - 1) * fraction
            lower, upper = math.floor(position), math.ceil(position)
            return ordered[lower] if lower == upper else ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
        return {"mean":statistics.fmean(ordered), "p50":percentile(.5), "p95":percentile(.95),
                "p99":percentile(.99), "max":max(ordered)}
    return {"cutoff":cutoff_len, "rows":len(input_lengths), "truncation_rows":truncations,
            "assistant_full_loss_rows":full_losses, "input_tokens":summary(input_lengths),
            "assistant_tokens":summary(assistant_lengths)}


def pad_features(features: list[dict[str, list[int]]], pad_id: int,
                 pad_to_multiple_of: int = 8) -> dict[str, list[list[int]]]:
    longest = max(len(feature["input_ids"]) for feature in features)
    target = int(math.ceil(longest / pad_to_multiple_of) * pad_to_multiple_of)
    ids, masks, labels = [], [], []
    for feature in features:
        padding = target - len(feature["input_ids"])
        ids.append(feature["input_ids"] + [pad_id] * padding)
        masks.append(feature["attention_mask"] + [0] * padding)
        labels.append(feature["labels"] + [-100] * padding)
    return {"input_ids": ids, "attention_mask": masks, "labels": labels}


class AssistantOnlyCollator:
    def __init__(self, tokenizer: Any, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer.pad_token_id must be configured")
        padded = pad_features(features, pad_id, self.pad_to_multiple_of)
        return {
            "input_ids": torch.tensor(padded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(padded["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(padded["labels"], dtype=torch.long),
        }


class EncodedRows:
    def __init__(self, rows: Iterable[dict[str, Any]], tokenizer: Any, cutoff_len: int):
        self.rows = [encode_assistant_only(tokenizer, row["messages"], cutoff_len) for row in rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.rows[index]


def task_paths(data_root: Path, task: str) -> tuple[Path, Path]:
    if task not in TASKS:
        raise ValueError(f"unsupported task: {task}")
    root = data_root / task
    return root / "train.jsonl", root / "validation.jsonl"


def output_path(output_root: Path, task: str, smoke: bool) -> Path:
    return output_root / ("smoke" if smoke else "full") / task


def common_lora_settings() -> dict[str, Any]:
    """Return the immutable LoRA contract shared by every task."""
    return {
        "r": DEFAULT_LORA_R,
        "alpha": DEFAULT_LORA_ALPHA,
        "dropout": DEFAULT_LORA_DROPOUT,
        "target_modules": TARGET_MODULES,
    }


def verify_lora_trainability(model: Any) -> dict[str, int]:
    trainable, frozen = 0, 0
    bad_trainable: list[str] = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            trainable += parameter.numel()
            if "lora_" not in name:
                bad_trainable.append(name)
        else:
            frozen += parameter.numel()
    if trainable == 0:
        raise ValueError("no trainable LoRA parameters")
    if bad_trainable:
        raise ValueError(f"non-LoRA base parameters are trainable: {bad_trainable[:5]}")
    return {"trainable_lora_parameters": trainable, "frozen_parameters": frozen}


def validate_adapter_tensor_keys(keys: Iterable[str]) -> list[str]:
    keys = list(keys)
    if not keys:
        raise ValueError("adapter artifact contains no tensors")
    non_lora = [key for key in keys if "lora_" not in key]
    if non_lora:
        raise ValueError(f"artifact contains non-LoRA tensors (possible base weights): {non_lora[:5]}")
    return keys


def inspect_adapter_artifact(adapter: Path, require_finite: bool = False) -> dict[str, int]:
    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("safetensors validation dependencies are missing") from exc
    weights = adapter / "adapter_model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    tensors, elements = 0, 0
    with safe_open(weights, framework="pt", device="cpu") as handle:
        keys = validate_adapter_tensor_keys(handle.keys())
        for key in keys:
            tensor = handle.get_tensor(key)
            if require_finite and not torch.isfinite(tensor).all().item():
                raise ValueError(f"non-finite adapter tensor: {key}")
            tensors += 1; elements += tensor.numel()
    return {"lora_tensors": tensors, "lora_elements": elements, "non_lora_tensors": 0}


def cuda_memory(torch: Any) -> dict[str, int]:
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def record_cuda_oom(destination: Path, torch: Any, memory_start: dict[str, int], exc: BaseException) -> None:
    memory_oom = cuda_memory(torch)
    logging.exception("CUDA OOM; aborting FP16 LoRA without any automatic fallback memory=%s", memory_oom)
    (destination / "run_result.json").write_text(json.dumps({
        "status": "cuda_oom", "automatic_fallback": False,
        "cuda_memory_start": memory_start, "cuda_memory_final": memory_oom,
        "error": str(exc),
    }, indent=2) + "\n", encoding="utf-8")


def train(args: argparse.Namespace) -> Path:
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed,
        )
    except ImportError as exc:
        raise RuntimeError("training dependencies are missing; install the pinned RTX requirements") from exc

    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FP16 LoRA training")
    torch.cuda.reset_peak_memory_stats()
    memory_start = cuda_memory(torch)
    lora = common_lora_settings()
    train_path, validation_path = task_paths(args.data_root, args.task)
    train_rows = read_jsonl(train_path, 16 if args.smoke else None)
    validation_rows = read_jsonl(validation_path, 8 if args.smoke else None)
    destination = output_path(args.output_root, args.task, args.smoke)
    destination.mkdir(parents=True, exist_ok=True)
    logging_dir = destination / "logs"
    logging_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=destination / "train.log", level=logging.INFO, force=True,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.info("starting task=%s smoke=%s train_rows=%d validation_rows=%d lora=%s cuda_memory=%s",
                 args.task, args.smoke, len(train_rows), len(validation_rows), lora, memory_start)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    token_stats = {
        "train": tokenization_statistics(tokenizer, train_rows, args.cutoff_len),
        "validation": tokenization_statistics(tokenizer, validation_rows, args.cutoff_len),
    }
    logging.info("tokenization_statistics=%s", token_stats)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True,
        )
    except torch.cuda.OutOfMemoryError as exc:
        record_cuda_oom(destination, torch, memory_start, exc)
        raise RuntimeError("CUDA out of memory while loading the FP16 base model; no automatic fallback was attempted") from exc
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        target_modules=list(lora["target_modules"]), bias="none",
    ))
    # Frozen embeddings still need gradient-bearing outputs when activation
    # checkpointing is combined with PEFT; this does not unfreeze their weights.
    model.enable_input_require_grads()
    parameter_counts = verify_lora_trainability(model)
    train_data = EncodedRows(train_rows, tokenizer, args.cutoff_len)
    validation_data = EncodedRows(validation_rows, tokenizer, args.cutoff_len)
    training_args = TrainingArguments(
        output_dir=str(destination), logging_dir=str(logging_dir),
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=16, learning_rate=2e-4, num_train_epochs=2,
        max_steps=10 if args.smoke else -1, fp16=True, bf16=False,
        gradient_checkpointing=True, optim="adamw_torch",
        eval_strategy="steps", eval_steps=5 if args.smoke else args.eval_steps,
        save_strategy="steps", save_steps=10 if args.smoke else args.save_steps,
        logging_steps=1 if args.smoke else args.logging_steps,
        seed=args.seed, data_seed=args.seed, report_to="none",
        remove_unused_columns=False, save_total_limit=2,
    )
    try:
        trainer = Trainer(
            model=model, args=training_args, train_dataset=train_data, eval_dataset=validation_data,
            data_collator=AssistantOnlyCollator(tokenizer),
        )
        trainer.train()
    except torch.cuda.OutOfMemoryError as exc:
        record_cuda_oom(destination, torch, memory_start, exc)
        raise RuntimeError("CUDA out of memory during FP16 LoRA training; no automatic fallback was attempted") from exc
    trainer.save_model(str(destination))
    tokenizer.save_pretrained(str(destination))
    artifact = inspect_adapter_artifact(destination, require_finite=True)
    memory_final = cuda_memory(torch)
    (destination / "run_config.json").write_text(json.dumps({
        "task": args.task, "model": args.model, "smoke": args.smoke, "seed": args.seed,
        "train_rows": len(train_rows), "validation_rows": len(validation_rows),
        "cutoff_len": args.cutoff_len, "tokenization_statistics": token_stats,
        "lora": lora, **parameter_counts,
    }, indent=2) + "\n", encoding="utf-8")
    (destination / "run_result.json").write_text(json.dumps({
        "status": "success", "cuda_memory_start": memory_start, "cuda_memory_final": memory_final,
        "artifact": artifact, **parameter_counts,
    }, indent=2) + "\n", encoding="utf-8")
    logging.info("completed task=%s output=%s cuda_memory=%s artifact=%s", args.task, destination, memory_final, artifact)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--model", default=BASE_MODEL, help="HF model ID or local snapshot")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.set_defaults(cutoff_len=DEFAULT_CUTOFF_LEN, seed=DEFAULT_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
