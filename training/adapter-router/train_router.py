#!/usr/bin/env python3
"""Bounded six-label Router LoRA smoke training and exact-set evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

NUM_LABELS = 6
HIDDEN_SIZE = 3072
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTERS = {0: "securecode", 1: "math", 2: "ifeval", 3: "bbh", 4: "gpqa", 5: "mmlupro"}
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ROUTER_INSTRUCTION = "Please analyze this text and predict which adapter is suitable for it.\nText: "
_OWNED_OUTPUT: Path | None = None
_CLASSIFIER_DIAGNOSTICS: dict[str, Any] = {}


def multilabel_bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.shape != labels.shape or logits.ndim != 2 or logits.shape[1] != NUM_LABELS:
        raise ValueError("logits and labels must both have shape [batch, 6]")
    if torch.any(labels.sum(dim=1) == 0):
        raise ValueError("empty exact labels are forbidden")
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.to(logits.dtype))


def validate_classifier(model: torch.nn.Module, hidden_size: int) -> torch.Tensor:
    candidates = [(name, parameter) for name, parameter in model.named_parameters()
                  if name.endswith("score.weight") or ".score." in name and name.endswith("weight")]
    matching = [(name, value) for name, value in candidates if tuple(value.shape) == (NUM_LABELS, hidden_size)]
    if not matching:
        raise RuntimeError(f"classifier shape [6, {hidden_size}] not found: {[(n, tuple(v.shape)) for n, v in candidates]}")
    preferred = [(name, value) for name, value in matching if "modules_to_save.default" in name]
    tensor = (preferred or matching)[0][1].detach()
    if not tensor.dtype.is_floating_point or not torch.isfinite(tensor).all():
        raise RuntimeError("classifier must have floating dtype and finite values")
    return tensor


def active_peft_classifier(model: torch.nn.Module, hidden_size: int) -> tuple[str, str, torch.Tensor]:
    """Return the active ModulesToSaveWrapper score, never original_module."""
    candidates = []
    for module_path, module in model.named_modules():
        modules_to_save = getattr(module, "modules_to_save", None)
        if modules_to_save is None or not hasattr(module, "original_module"):
            continue
        active = getattr(module, "active_adapter", None)
        if active is None: active = getattr(module, "active_adapters", None)
        if isinstance(active, (list, tuple, set)):
            if len(active) != 1:
                raise RuntimeError(f"expected one active adapter at {module_path}, got {list(active)}")
            active = next(iter(active))
        if not isinstance(active, str) or active not in modules_to_save:
            raise RuntimeError(f"active adapter {active!r} is unavailable at {module_path}; "
                               f"available={list(modules_to_save.keys())}")
        active_module = modules_to_save[active]
        weight = getattr(active_module, "weight", None)
        if weight is not None and tuple(weight.shape) == (NUM_LABELS, hidden_size):
            candidates.append((module_path, active, weight))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one active PEFT classifier with shape [6, {hidden_size}], "
                           f"found {[(path, adapter) for path, adapter, _ in candidates]}")
    module_path, active, weight = candidates[0]
    tensor = weight.detach()
    if not tensor.dtype.is_floating_point or not torch.isfinite(tensor).all():
        raise RuntimeError("active PEFT classifier must have floating dtype and finite values")
    return f"{module_path}.modules_to_save[{active!r}].weight", active, tensor


def ensure_active_peft_classifier_float32(model: torch.nn.Module, hidden_size: int) -> torch.Tensor:
    """Cast the active modules_to_save classifier module before optimizer creation."""
    path, adapter, _ = active_peft_classifier(model, hidden_size)
    matches = []
    for module in model.modules():
        modules_to_save = getattr(module, "modules_to_save", None)
        if modules_to_save is None or adapter not in modules_to_save:
            continue
        active_module = modules_to_save[adapter]
        weight = getattr(active_module, "weight", None)
        if weight is not None and tuple(weight.shape) == (NUM_LABELS, hidden_size):
            matches.append(active_module)
    if len(matches) != 1:
        raise RuntimeError(f"could not uniquely resolve active classifier module for {path}")
    matches[0].to(dtype=torch.float32)
    _, checked_adapter, tensor = active_peft_classifier(model, hidden_size)
    if checked_adapter != adapter or tensor.dtype != torch.float32:
        raise RuntimeError("active PEFT classifier must be torch.float32 before optimizer creation")
    return tensor


def file_record(path: Path, include_rows: bool = False) -> dict[str, Any]:
    digest = hashlib.sha256(); rows = 0
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            if line.strip(): rows += 1
    result: dict[str, Any] = {"path": str(path.resolve()), "sha256": digest.hexdigest()}
    if include_rows: result["rows"] = rows
    return result


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def ensure_new_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def tensor_record(path: str, tensor: torch.Tensor) -> dict[str, Any]:
    return {"path": path, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "finite": bool(torch.isfinite(tensor).all()), "sha256": tensor_sha256(tensor)}


def record_classifier_tensor(name: str, path: str, tensor: torch.Tensor) -> None:
    record = tensor_record(path, tensor)
    record["_tensor"] = tensor.detach().cpu().clone()
    _CLASSIFIER_DIAGNOSTICS[name] = record
    pairs = (("before_save", "safetensors"), ("safetensors", "reloaded"),
             ("before_save", "reloaded"))
    _CLASSIFIER_DIAGNOSTICS["torch_equal"] = {
        f"{left}_equals_{right}": torch.equal(_CLASSIFIER_DIAGNOSTICS[left]["_tensor"],
                                                _CLASSIFIER_DIAGNOSTICS[right]["_tensor"])
        for left, right in pairs
        if left in _CLASSIFIER_DIAGNOSTICS and right in _CLASSIFIER_DIAGNOSTICS
    }


def classifier_diagnostics() -> dict[str, Any]:
    return {key: ({field: value for field, value in record.items() if field != "_tensor"}
                  if isinstance(record, dict) else record)
            for key, record in _CLASSIFIER_DIAGNOSTICS.items()}


def prepare_score_for_peft_reload(model: torch.nn.Module, hidden_size: int) -> torch.Tensor:
    """Keep the FP16 base intact while making only its classifier load FP32 exactly."""
    score = getattr(model, "score", None)
    weight = getattr(score, "weight", None)
    if weight is None or tuple(weight.shape) != (NUM_LABELS, hidden_size):
        raise RuntimeError(f"base score classifier shape [6, {hidden_size}] not found")
    score.to(dtype=torch.float32)
    if score.weight.dtype != torch.float32:
        raise RuntimeError("base score classifier could not be converted to torch.float32")
    return score.weight.detach()


def saved_classifier(path: Path, expected_shape: tuple[int, int] = (NUM_LABELS, HIDDEN_SIZE)) -> tuple[str, torch.Tensor]:
    from safetensors import safe_open
    candidates = []
    with safe_open(path, framework="pt", device="cpu") as weights:
        for key in weights.keys():
            if "score" in key and key.endswith("weight"):
                tensor = weights.get_tensor(key)
                if tuple(tensor.shape) == expected_shape: candidates.append((key, tensor))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one saved score.weight with shape {list(expected_shape)}, found "
                           f"{[(key, list(value.shape)) for key, value in candidates]}")
    key, tensor = candidates[0]
    if not tensor.dtype.is_floating_point or not torch.isfinite(tensor).all():
        raise RuntimeError("saved classifier must have floating dtype and finite values")
    return key, tensor


def package_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in ("transformers", "torch", "peft")}


class ExactJsonlDataset(Dataset):
    def __init__(self, path: Path, tokenizer, cutoff_len: int, max_samples: int | None = None):
        self.rows = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip(): continue
                row = json.loads(line)
                where = f"{path}:{line_number}"
                if not isinstance(row, dict):
                    raise ValueError(f"training row must be a JSON object: {where}")
                prompt = row.get("prompt")
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError(f"prompt must be a nonempty string: {where}")
                if "user_prompt" in row and row["user_prompt"] != prompt:
                    raise ValueError(f"prompt and user_prompt conflict: {where}")
                for field in ("source_task", "source_group", "prompt_hash"):
                    value = row.get(field)
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError(f"{field} must be a nonempty string: {where}")
                labels = row.get("labels")
                if row.get("label_type") != "exact" or not isinstance(labels, list) or len(labels) != NUM_LABELS:
                    raise ValueError(f"training data must contain six-dimensional exact labels: {where}")
                if any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or value not in (0, 1) for value in labels):
                    raise ValueError(f"exact labels must be binary values: {where}")
                tensor = torch.tensor(labels, dtype=torch.float32)
                if not torch.all((tensor == 0) | (tensor == 1)) or tensor.sum() < 1:
                    raise ValueError(f"exact labels must be non-empty binary multi-hot vectors: {where}")
                label_width = row.get("label_width")
                if isinstance(label_width, bool) or not isinstance(label_width, int) or label_width != int(tensor.sum()):
                    raise ValueError(f"label_width must equal sum(labels): {where}")
                if max_samples is None or len(self.rows) < max_samples:
                    row["label_tensor"] = tensor
                    self.rows.append(row)
        self.tokenizer, self.cutoff_len = tokenizer, cutoff_len

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        encoded = self.tokenizer(ROUTER_INSTRUCTION + row["prompt"], truncation=True,
                                 max_length=self.cutoff_len)
        encoded["labels"] = row["label_tensor"]
        return encoded


class Collator:
    def __init__(self, tokenizer):
        from transformers import DataCollatorWithPadding
        self.padding = DataCollatorWithPadding(tokenizer)

    def __call__(self, features):
        values = [dict(value) for value in features]
        labels = torch.stack([value.pop("labels") for value in values])
        result = self.padding(values); result["labels"] = labels
        return result


def _mean(values): return sum(values) / len(values) if values else None


def evaluation_metrics(logits: torch.Tensor, labels: torch.Tensor,
                       rows: list[dict[str, Any]], threshold: float = 0.5) -> dict[str, Any]:
    if logits.shape != labels.shape or logits.shape[1] != NUM_LABELS or len(rows) != len(labels):
        raise ValueError("evaluation inputs disagree")
    truth = labels.bool(); probabilities = torch.sigmoid(logits)
    rankings = torch.argsort(probabilities, dim=1, descending=True)
    predicted = probabilities >= threshold
    tp = (predicted & truth).sum(dim=0).float(); fp = (predicted & ~truth).sum(dim=0).float()
    fn = (~predicted & truth).sum(dim=0).float()
    def prf(a, b, c):
        precision = float(a / (a + b)) if a + b else 0.0
        recall = float(a / (a + c)) if a + c else 0.0
        return precision, recall, 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    micro = prf(float(tp.sum()), float(fp.sum()), float(fn.sum()))
    per_label = [prf(float(tp[i]), float(fp[i]), float(fn[i])) for i in range(NUM_LABELS)]
    def subset(indices):
        hits = {k: [] for k in (1, 2, 3)}; rr = []; jaccard = []; exact = []
        for i in indices:
            positive = set(torch.where(truth[i])[0].tolist())
            ranked = rankings[i].tolist()
            for k in hits: hits[k].append(bool(positive & set(ranked[:k])))
            rr.append(1.0 / next(rank for rank, adapter in enumerate(ranked, 1) if adapter in positive))
            selected = set(torch.where(predicted[i])[0].tolist())
            jaccard.append(len(selected & positive) / len(selected | positive))
            exact.append(selected == positive)
        return {"rows": len(indices), "mean_label_width": _mean([int(truth[i].sum()) for i in indices]),
                "top_1_hit": _mean(hits[1]), "hit_at_2": _mean(hits[2]), "hit_at_3": _mean(hits[3]),
                "mrr": _mean(rr), "jaccard": _mean(jaccard), "exact_set_match": _mean(exact)}
    overall = subset(list(range(len(rows))))
    overall.update({"bce_loss": float(multilabel_bce(logits, labels)),
                    "micro_precision": micro[0], "micro_recall": micro[1], "micro_f1": micro[2],
                    "macro_precision": _mean([x[0] for x in per_label]),
                    "macro_recall": _mean([x[1] for x in per_label]), "macro_f1": _mean([x[2] for x in per_label])})
    buckets: dict[str, dict[Any, list[int]]] = {"label_width": defaultdict(list), "source_task": defaultdict(list)}
    for index, row in enumerate(rows):
        buckets["label_width"][str(int(truth[index].sum()))].append(index)
        buckets["source_task"][row["source_task"]].append(index)
    overall["by_label_width"] = {str(key): subset(indices) for key, indices in sorted(buckets["label_width"].items())}
    overall["by_source_task"] = {str(key): subset(indices) for key, indices in sorted(buckets["source_task"].items())}
    return overall


def baseline_metrics(labels: torch.Tensor, rows: list[dict[str, Any]], seed: int = 42) -> dict[str, Any]:
    rng = random.Random(seed)
    random_logits = torch.empty_like(labels)
    for row in random_logits: row.copy_(torch.tensor([rng.random() for _ in range(NUM_LABELS)]))
    frequency = labels.sum(dim=0)
    # Stable adapter-ID tie break and repeated logits provide a frequency-ranking baseline.
    frequency_logits = frequency.unsqueeze(0).repeat(len(labels), 1)
    return {"random_seed_42": evaluation_metrics(random_logits, labels, rows),
            "validation_label_frequency": evaluation_metrics(frequency_logits, labels, rows)}


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--data-dir", type=Path, default=root / "processed/six_label_router_smoke_v1")
    parser.add_argument("--output-dir", type=Path, default=root / "trained_models/six_label_router_smoke_v1")
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--train-samples", type=int, default=32)
    parser.add_argument("--validation-samples", type=int, default=32)
    parser.add_argument("--cutoff-len", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    return parser.parse_args()


def main():
    global _OWNED_OUTPUT
    _CLASSIFIER_DIAGNOSTICS.clear()
    args = parse_args()
    if args.max_steps < 1: raise ValueError("max-steps must be positive")
    ensure_new_output(args.output_dir)
    args.output_dir.mkdir(parents=True)
    _OWNED_OUTPUT = args.output_dir
    data_files = {name: args.data_dir / name for name in
                  ("train.jsonl", "validation.jsonl", "stats.json", "manifest.json")}
    missing = [str(path) for path in data_files.values() if not path.is_file()]
    if missing: raise FileNotFoundError("missing dataset files: " + ", ".join(missing))
    input_records = {
        "train.jsonl": file_record(data_files["train.jsonl"], include_rows=True),
        "validation.jsonl": file_record(data_files["validation.jsonl"], include_rows=True),
        "stats.json": file_record(data_files["stats.json"]),
        "manifest.json": file_record(data_files["manifest.json"]),
    }
    train = ExactJsonlDataset(
        args.data_dir / "train.jsonl", None, args.cutoff_len, args.train_samples)
    validation = ExactJsonlDataset(
        args.data_dir / "validation.jsonl", None, args.cutoff_len, args.validation_samples)
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel
    from transformers import (AutoTokenizer, LlamaForSequenceClassification, Trainer,
                              TrainingArguments, set_seed)
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token_id = tokenizer.eos_token_id; tokenizer.padding_side = "left"
    train.tokenizer = tokenizer
    validation.tokenizer = tokenizer
    model = LlamaForSequenceClassification.from_pretrained(
        args.base_model, num_labels=NUM_LABELS, torch_dtype=torch.float16)
    if model.config.hidden_size != HIDDEN_SIZE:
        raise RuntimeError(f"expected hidden_size={HIDDEN_SIZE}, got {model.config.hidden_size}")
    model.config.pad_token_id = tokenizer.pad_token_id; model.config.problem_type = "multi_label_classification"
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=args.lora_dropout, bias="none", task_type=TaskType.SEQ_CLS, modules_to_save=["score"]))
    ensure_active_peft_classifier_float32(model, model.config.hidden_size)
    for parameter in model.parameters():
        if parameter.requires_grad: parameter.data = parameter.data.float()
    _, _, training_classifier = active_peft_classifier(model, model.config.hidden_size)
    if training_classifier.dtype != torch.float32:
        raise RuntimeError("active PEFT classifier lost FP32 dtype before Trainer creation")
    class BCETrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels"); outputs = model(**inputs); loss = multilabel_bce(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss
    trainer = BCETrainer(model=model, args=TrainingArguments(output_dir=str(args.output_dir), max_steps=args.max_steps,
        per_device_train_batch_size=args.micro_batch_size, per_device_eval_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps, learning_rate=args.learning_rate,
        seed=42, data_seed=42, fp16=torch.cuda.is_available(), logging_steps=1, save_strategy="no",
        evaluation_strategy="no", report_to=[], remove_unused_columns=False), train_dataset=train,
        eval_dataset=validation, data_collator=Collator(tokenizer))
    trainer.train()
    before_path, before_adapter, before_tensor = active_peft_classifier(model, HIDDEN_SIZE)
    classifier_before_save = before_tensor.detach().cpu().clone()
    record_classifier_tensor("before_save", before_path, classifier_before_save)
    if classifier_before_save.dtype != torch.float32:
        raise RuntimeError("classifier immediately before save must be torch.float32")
    trainer.save_model(str(args.output_dir)); tokenizer.save_pretrained(args.output_dir)
    saved_key, classifier_saved = saved_classifier(args.output_dir / "adapter_model.safetensors")
    saved_path = f"{(args.output_dir / 'adapter_model.safetensors').resolve()}::{saved_key}"
    record_classifier_tensor("safetensors", saved_path, classifier_saved)
    if classifier_saved.dtype != torch.float32:
        raise RuntimeError("saved classifier must be torch.float32")
    if not torch.equal(classifier_before_save, classifier_saved):
        raise RuntimeError("saved classifier differs from classifier immediately before save")
    prediction = trainer.predict(validation)
    labels = torch.as_tensor(prediction.label_ids); logits = torch.as_tensor(prediction.predictions)
    metrics = {"model": evaluation_metrics(logits, labels, validation.rows),
               "baselines": baseline_metrics(labels, validation.rows),
               "warning": "Smoke validation only; not final test performance."}
    atomic_write_json(args.output_dir / "validation_metrics.json", metrics)
    del trainer, model
    reload_base = LlamaForSequenceClassification.from_pretrained(
        args.base_model, num_labels=NUM_LABELS, torch_dtype=torch.float16)
    prepare_score_for_peft_reload(reload_base, HIDDEN_SIZE)
    reloaded = PeftModel.from_pretrained(reload_base, args.output_dir)
    reload_path, reload_adapter, reload_tensor = active_peft_classifier(reloaded, HIDDEN_SIZE)
    tensor = reload_tensor.detach().cpu()
    record_classifier_tensor("reloaded", reload_path, tensor)
    if tensor.dtype != torch.float32:
        raise RuntimeError("reloaded active classifier must be torch.float32")
    if before_adapter != reload_adapter:
        raise RuntimeError(f"active adapter changed across reload: {before_adapter!r} != {reload_adapter!r}")
    if not torch.equal(classifier_saved, tensor):
        raise RuntimeError("reloaded classifier differs from adapter_model.safetensors")
    if not torch.equal(classifier_before_save, tensor):
        raise RuntimeError("reloaded classifier differs from classifier immediately before save")
    run_config = {
        "base_model": args.base_model, "num_labels": NUM_LABELS,
        "adapter_id_name_mapping": {str(key): value for key, value in ADAPTERS.items()},
        "seed": 42, "data_seed": 42, "data_directory": str(args.data_dir.resolve()),
        "used_rows": {"train": len(train), "validation": len(validation)},
        "max_steps": args.max_steps,
        "batch_size": args.micro_batch_size * args.gradient_accumulation_steps,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout,
                 "target_modules": LORA_TARGET_MODULES},
        "cutoff_length": args.cutoff_len, "dtype": "torch.float16",
        "problem_type": "multi_label_classification", "input_files": input_records,
        "versions": package_versions(),
        "classifier_verification": {
            "active_adapter": before_adapter,
            **classifier_diagnostics(),
        },
    }
    atomic_write_json(args.output_dir / "run_config.json", run_config)
    print(json.dumps({"saved_and_reloaded": True, "classifier_shape": list(tensor.shape),
                      "classifier_dtype": str(tensor.dtype), "classifier_finite": bool(torch.isfinite(tensor).all()),
                      "metrics_path": str(args.output_dir / "validation_metrics.json"),
                      "run_config_path": str(args.output_dir / "run_config.json")}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if _OWNED_OUTPUT is not None:
            atomic_write_json(_OWNED_OUTPUT / "failure_result.json", {
                "complete": False, "error_type": type(error).__name__, "error": str(error),
                "classifier_verification": classifier_diagnostics()})
        raise
