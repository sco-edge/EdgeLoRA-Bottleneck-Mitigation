#!/usr/bin/env python3
"""Pilot cross-profiling for six task adapters on held-out test data only."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Any

import evaluate_task_adapter as task_eval

TASKS = task_eval.TASKS
BASE_MODEL = task_eval.BASE_MODEL
SEED = task_eval.SEED
ADAPTER_IDS = {task: index for index, task in enumerate(TASKS)}
DEFAULT_DATA_ROOTS = {
    "securecode": Path("adapter-training/processed/six_task_v2_r1/securecode"),
    "math": Path("adapter-training/processed/six_task_v2_r3/math"),
    **{task: Path(f"adapter-training/processed/six_task_v1_r3/{task}")
       for task in ("ifeval", "bbh", "gpqa", "mmlupro")},
}
OUTPUT_FILES = ("cross_predictions.jsonl", "routing_labels.jsonl", "summary.json", "run_config.json",
                "evaluation.log")


def log_cuda_memory(torch: Any, stage: str, **context: Any) -> None:
    """Log allocator state without retaining CUDA tensors."""
    if not torch.cuda.is_available():
        return
    mib = 1024 ** 2
    fields = " ".join(f"{key}={value}" for key, value in context.items())
    logging.info("cuda_memory stage=%s %s allocated_mib=%.2f reserved_mib=%.2f peak_mib=%.2f",
                 stage, fields, torch.cuda.memory_allocated() / mib,
                 torch.cuda.memory_reserved() / mib, torch.cuda.max_memory_allocated() / mib)


def math_score(row: dict[str, Any], prediction: str) -> dict[str, Any]:
    return task_eval.score_prediction("math", row, prediction)


def normalized_containment(reference: str | None, prediction: str | None) -> bool:
    if not reference or not prediction: return False
    ref, pred = task_eval.normalize_answer(reference), task_eval.normalize_answer(prediction)
    if not ref or not pred: return False
    return bool(re.search(rf"(?<!\w){re.escape(ref)}(?!\w)", pred))


def score_generation(task: str, row: dict[str, Any], prediction: str,
                     generation_error: str | None = None, generation_truncated: bool = False) -> dict[str, Any]:
    if generation_error:
        return {"parsed_reference": None, "parsed_prediction": None, "parse_success": False,
                "generation_score": 0.0, "details": {}, "scoring_error": generation_error}
    scored = math_score(row, prediction) if task == "math" else task_eval.score_prediction(task, row, prediction)
    if generation_truncated:
        return {"parsed_reference": scored.get("parsed_reference"),
                "parsed_prediction": scored.get("parsed_prediction"),
                "parse_success": bool(scored.get("parse_success")), "generation_score": None,
                "details": dict(scored.get("details", {})), "scoring_error": "generation_truncated"}
    if task == "math" and not scored.get("parse_success"):
        return {"parsed_reference": scored.get("parsed_reference"),
                "parsed_prediction": scored.get("parsed_prediction"), "parse_success": False,
                "generation_score": None, "details": dict(scored.get("details", {})),
                "scoring_error": "unscorable_math_answer"}
    score_error = None
    score = float(scored.get("per_sample_score") or 0.0)
    if not scored.get("parse_success"):
        score, score_error = 0.0, "generation_parse_failure"
    details = dict(scored.get("details", {}))
    if task == "bbh":
        details["normalized_containment"] = normalized_containment(
            scored.get("parsed_reference"), scored.get("parsed_prediction"))
        details["containment_is_auxiliary_only"] = True
    if task == "securecode":
        details["proxy_limit"] = "Syntax/security heuristics and text similarity do not prove code correctness or security."
    return {"parsed_reference": scored.get("parsed_reference"),
            "parsed_prediction": scored.get("parsed_prediction"),
            "parse_success": bool(scored.get("parse_success")), "generation_score": score,
            "details": details, "scoring_error": score_error}


def encode_reference_only(tokenizer: Any, row: dict[str, Any]) -> dict[str, list[int] | int]:
    """Serialize the training contract and mask system/user prompt tokens from loss."""
    messages = row["messages"]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prefix_text = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    if full_ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError("chat template assistant prefix is not a token prefix of the full conversation")
    labels = [-100] * len(prefix_ids) + list(full_ids[len(prefix_ids):])
    answer_tokens = len(full_ids) - len(prefix_ids)
    if answer_tokens <= 0: raise ValueError("assistant reference has no tokens")
    return {"input_ids": list(full_ids), "attention_mask": [1] * len(full_ids),
            "labels": labels, "reference_token_count": answer_tokens}


def reference_mean_nll(model: Any, tokenizer: Any, row: dict[str, Any], torch: Any) -> tuple[float, int]:
    """Compute assistant-only NLL while retaining at most one-token vocab logits.

    Call the LoRA-patched Llama decoder directly so the prefix never creates
    sequence_length x vocabulary logits. Only its final hidden row is projected
    through the model's lm_head, then reference tokens are streamed with KV cache.
    """
    encoded = encode_reference_only(tokenizer, row)
    answer_count = int(encoded["reference_token_count"])
    prefix_count = len(encoded["input_ids"]) - answer_count
    input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=model.device)
    prefix_ids = input_ids[:, :prefix_count]
    answer_ids = input_ids[:, prefix_count:]
    total_nll = 0.0
    past_key_values = None
    causal_model = model.get_base_model()
    decoder = causal_model.model
    lm_head = causal_model.lm_head
    try:
        with torch.inference_mode():
            outputs = decoder(input_ids=prefix_ids, use_cache=True, return_dict=True)
            past_key_values = outputs.past_key_values
            last_hidden = outputs.last_hidden_state[:, -1:, :]
            logits = lm_head(last_hidden).float()
            total_nll += float(torch.nn.functional.cross_entropy(
                logits[:, 0, :], answer_ids[:, 0]).cpu().item())
            del outputs, last_hidden, logits
            for index in range(answer_count - 1):
                outputs = decoder(input_ids=answer_ids[:, index:index + 1],
                                  past_key_values=past_key_values, use_cache=True, return_dict=True)
                past_key_values = outputs.past_key_values
                last_hidden = outputs.last_hidden_state[:, -1:, :]
                logits = lm_head(last_hidden).float()
                total_nll += float(torch.nn.functional.cross_entropy(
                    logits[:, 0, :], answer_ids[:, index + 1]).cpu().item())
                del outputs, last_hidden, logits
        return total_nll / answer_count, answer_count
    finally:
        del input_ids, prefix_ids, answer_ids, past_key_values, causal_model, decoder, lm_head


def candidate(row: dict[str, Any], source_task: str, adapter_task: str, prediction: str,
              mean_nll: float, reference_tokens: int, generation_error: str | None = None,
              nll_error: str | None = None, *, generated_tokens: int = 0,
              reached_max_new_tokens: bool = False, eos_terminated: bool = False) -> dict[str, Any]:
    truncated = reached_max_new_tokens and not eos_terminated
    scored = score_generation(source_task, row, prediction, generation_error, truncated)
    scoring_error = scored.get("scoring_error")
    errors = [error for error in (generation_error, scoring_error, nll_error) if error]
    finite = math.isfinite(mean_nll)
    if not finite: errors.append("non_finite_mean_nll")
    return {"prompt_hash": row.get("prompt_hash"), "source_group": row.get("source_group"),
            "source_task": source_task, "adapter_id": ADAPTER_IDS[adapter_task],
            "adapter_task": adapter_task, "reference": str(row["messages"][2]["content"]),
            "user_prompt": str(row["messages"][1]["content"]),
            "prediction": prediction, **scored,
            "task_aware_generation_score": scored["generation_score"],
            "generated_tokens": generated_tokens, "actual_generated_tokens": generated_tokens,
            "reached_max_new_tokens": reached_max_new_tokens, "eos_terminated": eos_terminated,
            "generation_truncated": truncated,
            "scoring_error": scoring_error,
            "mean_nll": mean_nll if finite else None,
            "reference_token_count": reference_tokens, "candidate_valid": finite and nll_error is None,
            "error": "; ".join(errors) if errors else None}


def select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [item for item in candidates if item.get("candidate_valid") and
             item.get("generation_score") is not None]
    if not valid:
        return {"resolved": False, "selected": None, "selection_reason": "all_candidates_invalid"}
    ordered = sorted(valid, key=lambda item: (-float(item["generation_score"]),
                                               float(item["mean_nll"]), int(item["adapter_id"])))
    selected = ordered[0]
    score_ties = [item for item in valid if float(item["generation_score"]) == float(selected["generation_score"])]
    if len(score_ties) == 1: reason = "highest_generation_score"
    else:
        best_nll = min(float(item["mean_nll"]) for item in score_ties)
        nll_ties = [item for item in score_ties if float(item["mean_nll"]) == best_nll]
        reason = "lower_mean_nll" if len(nll_ties) == 1 else "lower_adapter_id"
    return {"resolved": True, "selected": selected, "selection_reason": reason}


def routing_label(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    first = candidates[0]
    correct = sorted(int(item["adapter_id"]) for item in candidates
                     if item.get("generation_score") == 1.0 and not item.get("generation_truncated"))
    weak_candidates = [item for item in candidates if item.get("candidate_valid")]
    weak = min(weak_candidates, key=lambda item: (float(item["mean_nll"]), int(item["adapter_id"]))) \
        if weak_candidates else None
    base = {"prompt_hash": first.get("prompt_hash"), "source_group": first.get("source_group"),
            "source_task": first["source_task"], "resolved": bool(correct),
            "ground_truth_adapter_ids": correct,
            "selected_adapter_id": correct[0] if correct else None,
            "generation_score": 1.0 if correct else None,
            "mean_nll": None,
            "selection_reason": "exact_generation_multi_label" if correct else "no_exact_generation",
            "weak_label_adapter_id": weak["adapter_id"] if weak is not None else None,
            "weak_label_mean_nll": weak["mean_nll"] if weak is not None else None,
            "weak_label_reason": "lowest_finite_reference_mean_nll" if weak is not None else "all_candidates_invalid"}
    return base


def parse_data_roots(values: list[str]) -> dict[str, Path]:
    result = dict(DEFAULT_DATA_ROOTS)
    for value in values:
        if "=" not in value: raise ValueError("--data-root must use TASK=PATH")
        task, path = value.split("=", 1)
        if task not in TASKS or not path: raise ValueError(f"invalid --data-root: {value}")
        result[task] = Path(path)
    return result


def load_heldout(data_roots: dict[str, Path], max_samples: int) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    rows, fingerprints = {}, {}
    for task in TASKS:
        path = data_roots[task] / "test.jsonl"
        fingerprints[task] = task_eval.file_fingerprint(path)
        rows[task] = task_eval.load_rows(path, task, max_samples)
        for index, row in enumerate(rows[task], 1):
            if not row.get("prompt_hash") or not row.get("source_group"):
                raise ValueError(f"{path}: selected row {index} lacks prompt_hash/source_group")
    return rows, fingerprints


def profile_adapter(base: Any, tokenizer: Any, adapter_task: str, adapter_path: Path,
                    rows_by_task: dict[str, list[dict[str, Any]]], batch_size: int,
                    max_new_tokens: int, torch: Any) -> tuple[list[dict[str, Any]], Any]:
    from peft import PeftModel
    from train_task_adapter import inspect_adapter_artifact
    inspect_adapter_artifact(adapter_path, require_finite=True)
    torch.cuda.reset_peak_memory_stats()
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False).to("cuda").eval()
    log_cuda_memory(torch, "adapter_loaded", adapter=adapter_task)
    records = []
    try:
        with torch.inference_mode():
            for source_task, rows in rows_by_task.items():
                log_cuda_memory(torch, "task_start", adapter=adapter_task, task=source_task)
                for start in range(0, len(rows), batch_size):
                    batch = rows[start:start + batch_size]
                    prompts = [tokenizer.apply_chat_template(task_eval.prompt_messages(row), tokenize=False,
                                                              add_generation_prompt=True) for row in batch]
                    encoded = generated = None
                    try:
                        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
                        width = encoded["input_ids"].shape[1]
                        log_cuda_memory(torch, "before_generation", adapter=adapter_task,
                                        task=source_task, batch_start=start)
                        generated = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False,
                                                   use_cache=True, pad_token_id=tokenizer.pad_token_id)
                        decoded = [task_eval.decode_generated_only(tokenizer, generated[i], width)
                                   for i in range(len(batch))]
                        terminations = [task_eval.generation_termination(
                            generated[i], width, max_new_tokens, tokenizer.eos_token_id)
                            for i in range(len(batch))]
                    except Exception as exc:
                        logging.exception("generation failed adapter=%s task=%s", adapter_task, source_task)
                        decoded = [("", 0)] * len(batch)
                        terminations = [{"reached_max_new_tokens": False, "eos_terminated": False,
                                         "generation_truncated": False}] * len(batch)
                        generation_error = f"generation_failure: {type(exc).__name__}: {exc}"
                    else:
                        generation_error = None
                    finally:
                        del encoded, generated, prompts
                        torch.cuda.empty_cache()
                    log_cuda_memory(torch, "after_generation_cleanup", adapter=adapter_task,
                                    task=source_task, batch_start=start)
                    for row, (prediction, generated_count), termination in zip(batch, decoded, terminations):
                        gen_error = generation_error or ("empty_response" if not prediction else None)
                        try:
                            nll, token_count = reference_mean_nll(model, tokenizer, row, torch)
                        except Exception as exc:
                            logging.exception("reference NLL failed adapter=%s task=%s", adapter_task, source_task)
                            nll, token_count = float("nan"), 0
                            nll_error = f"nll_failure: {type(exc).__name__}: {exc}"
                        else:
                            nll_error = None
                        records.append(candidate(row, source_task, adapter_task, prediction, nll, token_count,
                                                 gen_error, nll_error, generated_tokens=generated_count,
                                                 reached_max_new_tokens=termination["reached_max_new_tokens"],
                                                 eos_terminated=termination["eos_terminated"]))
                    del decoded, terminations, batch
                gc.collect()
                torch.cuda.empty_cache()
                log_cuda_memory(torch, "task_end_cleanup", adapter=adapter_task, task=source_task)
    finally:
        base = model.unload()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        log_cuda_memory(torch, "adapter_end_cleanup", adapter=adapter_task)
    return records, base


def load_base_runtime(model: str = BASE_MODEL) -> tuple[Any, Any, Any]:
    """Load the single shared FP16 base model and return runtime dependencies."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FP16 cross-profiling")
    set_seed(SEED)
    base = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda").eval()
    return torch, base, AutoTokenizer


def run(args: argparse.Namespace) -> Path:
    if args.split != "test": raise ValueError("cross-profiling accepts held-out --split test only")
    if args.max_samples <= 0 or args.batch_size <= 0 or args.max_new_tokens <= 0:
        raise ValueError("sample, batch, and generation limits must be positive")
    if args.output.exists(): raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    data_roots = parse_data_roots(args.data_root)
    rows_by_task, before = load_heldout(data_roots, args.max_samples)
    args.output.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, force=True, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(args.output / "evaluation.log", encoding="utf-8")])
    model_name = getattr(args, "model", BASE_MODEL)
    config = {"base_model": model_name, "dtype": "float16", "seed": SEED, "do_sample": False,
              "automatic_quantization_fallback": False, "split": "test", "max_samples_per_task": args.max_samples,
              "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens,
              "adapter_ids": ADAPTER_IDS, "adapters_root": str(args.adapters_root),
              "data_roots": {k:str(v) for k,v in data_roots.items()}, "input_fingerprints_before": before}
    (args.output/"run_config.json").write_text(json.dumps(config, indent=2)+"\n", encoding="utf-8")
    torch, base, AutoTokenizer = load_base_runtime(model_name)
    log_cuda_memory(torch, "base_loaded")
    started = time.perf_counter(); records = []
    try:
        for adapter_task in TASKS:
            path = args.adapters_root / adapter_task
            tokenizer = None
            try:
                tokenizer = AutoTokenizer.from_pretrained(path)
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = "left"
                adapter_records, base = profile_adapter(
                    base, tokenizer, adapter_task, path, rows_by_task,
                    args.batch_size, args.max_new_tokens, torch)
                records.extend(adapter_records)
                del adapter_records
            except Exception as exc:
                logging.exception("adapter load/evaluation failed: %s", adapter_task)
                for source_task, rows in rows_by_task.items():
                    records.extend(candidate(row, source_task, adapter_task, "", float("nan"), 0,
                                             f"adapter_loading_failure: {type(exc).__name__}: {exc}") for row in rows)
            finally:
                del tokenizer
                gc.collect()
                torch.cuda.empty_cache()
                log_cuda_memory(torch, "adapter_loop_end", adapter=adapter_task)
    finally:
        del base
        gc.collect()
        torch.cuda.empty_cache()
        log_cuda_memory(torch, "base_unloaded")
    after = {task:task_eval.file_fingerprint(data_roots[task]/"test.jsonl") for task in TASKS}
    if before != after: raise RuntimeError(f"held-out input changed during profiling: before={before}, after={after}")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in records: grouped.setdefault((item["source_task"], item["prompt_hash"]), []).append(item)
    labels = [routing_label(grouped[key]) for key in sorted(grouped)]
    with (args.output/"cross_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for item in records: handle.write(json.dumps(item, ensure_ascii=False)+"\n")
    with (args.output/"routing_labels.jsonl").open("w", encoding="utf-8") as handle:
        for item in labels: handle.write(json.dumps(item, ensure_ascii=False)+"\n")
    summary = {"source_samples": sum(map(len, rows_by_task.values())), "candidate_rows": len(records),
               "resolved_samples": sum(item["resolved"] for item in labels),
               "unresolved_samples": sum(not item["resolved"] for item in labels),
               "selected_adapter_counts": {
                   task:sum(item["selected_adapter_id"] == adapter_id for item in labels)
                   for task, adapter_id in ADAPTER_IDS.items()},
               "exact_multilabel_positive_counts": {
                   task:sum(ADAPTER_IDS[task] in item.get("ground_truth_adapter_ids", []) for item in labels)
                   for task in TASKS},
               "runtime_seconds": time.perf_counter()-started, "input_fingerprints_unchanged": True,
               "metric_manifest": {
                   "router_ground_truth": "All adapters with an exact task-aware generation score of 1 form the multi-label ground truth. Truncated generations are excluded.",
                   "weak_label": "Lowest finite assistant-reference mean NLL is reported separately as weak_label and is never mixed into exact ground truth.",
                   "bbh": "Normalized containment is auxiliary only; exact match remains the generation score.",
                   "securecode": "Syntax/security heuristics, token F1, and text similarity are proxy signals only; they do not prove correctness or security.",
                   "nll": "Mean teacher-forced NLL is calculated only on assistant reference tokens; system/user labels are masked."}}
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    return args.output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=BASE_MODEL, help="HF model ID or local snapshot")
    parser.add_argument("--adapters-root", type=Path, required=True)
    parser.add_argument("--data-root", action="append", default=[], metavar="TASK=PATH")
    parser.add_argument("--split", default="test", choices=("test",))
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__": run(parse_args())
