#!/usr/bin/env python3
"""Held-out response-quality evaluation for the six EdgeLoRA task adapters."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import math
import os
import random
import re
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

TASKS = ("securecode", "math", "ifeval", "bbh", "gpqa", "mmlupro")
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
SEED = 42


def file_fingerprint(path: Path) -> dict[str, Any]:
    digest, rows = hashlib.sha256(), 0
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            if line.strip(): rows += 1
    return {"path": str(path), "rows": rows, "sha256": digest.hexdigest()}


def load_rows(path: Path, task: str, limit: int | None) -> list[dict[str, Any]]:
    result = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip(): continue
            row = json.loads(line)
            messages = row.get("messages")
            if not isinstance(messages, list) or [m.get("role") for m in messages] != ["system", "user", "assistant"]:
                raise ValueError(f"{path}:{number}: expected system/user/assistant messages")
            if row.get("task", task) != task or any(not str(m.get("content", "")).strip() for m in messages):
                raise ValueError(f"{path}:{number}: task mismatch or empty message content")
            result.append(row)
            if limit is not None and len(result) >= limit: break
    if not result: raise ValueError(f"{path}: no evaluation rows")
    return result


def prompt_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """Return a fresh system/user-only prompt; the reference can never leak."""
    messages = row["messages"]
    if [m.get("role") for m in messages] != ["system", "user", "assistant"]:
        raise ValueError("expected system/user/assistant messages")
    return [{"role": str(m["role"]), "content": str(m["content"])} for m in messages[:2]]


def normalize_text(value: str) -> str:
    return " ".join(str(value).strip().split()).casefold()


def normalize_answer(value: str) -> str:
    value = normalize_text(value)
    value = re.sub(r"^[\s\[\(]*(?:answe(?:r)?|result|final answer)\s*(?:is|[:=])\s*", "", value)
    value = re.sub(r"^\\(?:boxed|text|mathrm)\{(.*)\}$", r"\1", value)
    value = re.sub(r"^\\frac\{\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\}\{\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\}$",
                   r"\1/\2", value)
    value = value.replace(r"\,", "").replace(r"\!", "")
    value = value.strip(" \t\n.。,:;!?[](){}$`\"")
    value = re.sub(r"\s*([=+\-*/^])\s*", r"\1", value)
    value = re.sub(r"\s+", " ", value)
    try:
        number = float(value.replace(",", ""))
        if math.isfinite(number): return format(number, ".15g")
    except ValueError: pass
    return value


def extract_math_label(value: str) -> str | None:
    """Extract an explicitly presented MCQ label, including legacy `answe` rows."""
    text = str(value)
    patterns = (
        r"(?:the\s+)?answe(?:r)?\s*(?:is|[:=])\s*[\[(]?([A-E])\b",
        r"\[RESULT\]\s*[\[(]?([A-E])\b",
        r"\\boxed\{\s*([A-E])\s*\}",
        r"^\s*[\[(]?([A-E])(?:[\])\s.:\-]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return match.group(1).upper()
    return None


_NUMBER = r"[-+]?\d[\d,]*(?:\.\d+)?"
_SCALAR = rf"(?:{_NUMBER}(?:\s*/\s*{_NUMBER})?|\\frac\{{\s*{_NUMBER}\s*\}}\{{\s*{_NUMBER}\s*\}})"


def extract_math_answer_with_method(value: str) -> tuple[str | None, str]:
    """Extract only answers behind a trusted boundary; never scan for a stray number."""
    text = str(value)
    boxed = re.findall(rf"\\boxed\{{\s*({_SCALAR}|[^{{}}]+)\s*\}}", text)
    if boxed:
        answer = normalize_answer(boxed[-1])
        return (answer, "boxed") if answer else (None, "unscorable")
    result = re.search(r"\[RESULT\][ \t]*(?:\r?\n)?((?:[ \t]*\r?\n)*[^\r\n]+)", text, re.I)
    if result:
        answer = normalize_answer(result.group(1).strip())
        return (answer, "result_block") if answer else (None, "unscorable")
    phrase = re.findall(
        rf"(?:therefore\s+the\s+answer\s+is|the\s+answer\s+is|final\s+answer\s*(?:is|[:=])?|"
        rf"answe(?:r)?\s*(?:is|[:=])|the\s+result\s+is|the\s+value\s+is|is\s+equal\s+to)"
        rf"\s*({_SCALAR})(?![\w.])",
        text, re.I)
    if phrase:
        return normalize_answer(phrase[-1]), "answer_phrase"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # Inspect only the final line for a terminal measured result. A bare
    # assignment such as z=2 or a code/function argument remains unscorable.
    if lines:
        quantity = re.fullmatch(
            rf"\s*[^=`()]+?\s+(?:is|=)\s*({_SCALAR})\s+[A-Za-z][A-Za-z -]*[.!]?\s*", lines[-1])
        if quantity:
            return normalize_answer(quantity.group(1)), "standalone_final_line"
    if lines and re.fullmatch(rf"\s*({_SCALAR})\s*[.]?\s*", lines[-1]):
        return normalize_answer(lines[-1]), "standalone_final_line"
    return None, "unscorable"


def extract_math_answer(value: str) -> str | None:
    return extract_math_answer_with_method(value)[0]


def extract_bbh_answer(value: str) -> str | None:
    match = re.findall(r"\[RESULT\]\s*(.+)", value, re.I | re.S)
    candidate = match[-1].strip().splitlines()[0] if match else (value.strip().splitlines()[-1] if value.strip() else "")
    normalized = " ".join(candidate.strip().split()).casefold().rstrip(".。,:;!?").rstrip()
    return normalized or None


def choice_labels(prompt: str) -> list[str]:
    return re.findall(r"(?m)^\s*([A-Z])\s*[.):=]\s*", prompt)


def _normalize_math_choice_value(raw: str) -> str:
    raw = raw.strip()
    scalar = re.match(rf"^({_SCALAR})(?=\s*(?:[A-Za-z]|$))", raw)
    return normalize_answer(scalar.group(1) if scalar else raw)


def math_choice_values(prompt: str) -> dict[str, str]:
    """Recover an unambiguous line-based or explicitly delimited option mapping."""
    line_result: dict[str, str] = {}
    for match in re.finditer(r"(?m)^\s*([A-J])\s*[.):=]\s*(\S[^\r\n]*)$", str(prompt), re.I):
        label, raw = match.group(1).upper(), match.group(2).strip()
        value = _normalize_math_choice_value(raw)
        if not value or label in line_result:
            return {}
        line_result[label] = value
    if len(line_result) >= 2:
        return line_result

    section = re.search(r"(?:answer\s+choices|choices|options)\s*:\s*(.*)", str(prompt), re.I | re.S)
    if not section:
        return {}
    body = section.group(1)
    markers = list(re.finditer(r"\(([A-J])\)", body, re.I))
    if len(markers) < 2:
        return {}
    result: dict[str, str] = {}
    for index, marker in enumerate(markers):
        label = marker.group(1).upper()
        end = markers[index + 1].start() if index + 1 < len(markers) else len(body)
        raw = body[marker.end():end].strip()
        if label in result or not raw:
            return {}
        value = _normalize_math_choice_value(raw)
        if not value:
            return {}
        result[label] = value
    return result if len(result) >= 2 else {}


def extract_math_mapped_final_sentence(value: str, options: dict[str, str]) -> tuple[str | None, str]:
    """Extract a terminal equation only when its RHS uniquely matches an MCQ option."""
    if len(options) < 2:
        return None, "unscorable"
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", str(value).strip()) if part.strip()]
    if not sentences:
        return None, "unscorable"
    sentence = sentences[-1]
    if sentence.count("=") != 1 or "," in sentence:
        return None, "unscorable"
    lhs, rhs = (part.strip() for part in sentence.split("=", 1))
    target_words = {"length", "area", "total", "value", "answer", "result"}
    if not target_words.intersection(re.findall(r"[A-Za-z]+", lhs.casefold())):
        return None, "unscorable"
    match = re.fullmatch(rf"({_SCALAR})\s*(?:[A-Za-z][A-Za-z0-9^/ -]*)?[.!]?\s*", rhs)
    if not match:
        return None, "unscorable"
    answer = normalize_answer(match.group(1))
    option_mentions = sum(
        len(re.findall(rf"(?<!\d){re.escape(option)}(?!\d)", rhs.replace(",", "")))
        for option in set(options.values()) if re.fullmatch(_NUMBER, option)
    )
    if option_mentions != 1 or sum(answer == option for option in options.values()) != 1:
        return None, "unscorable"
    return answer, "final_sentence_terminal_equation"


def parse_mcq_label(value: str, labels: Iterable[str] = tuple("ABCDEFGHIJ")) -> str | None:
    allowed = {str(label).upper() for label in labels}
    patterns = (r"(?:answer|choice|option|label)\s*(?:is|:|=)?\s*[\[\(]?([A-J])\b",
                r"\[RESULT\]\s*[\[\(]?([A-J])(?:\b|\s*$)",
                r"^\s*[\[\(]?([A-J])[\]\)]?\s*[\.:\-]", r"\\boxed\{\s*([A-J])\s*\}")
    for pattern in patterns:
        match = re.search(pattern, value, re.I)
        if match and match.group(1).upper() in allowed: return match.group(1).upper()
    lone = re.fullmatch(r"\s*[\[\(]?([A-J])[\]\)]?\s*", value, re.I)
    return lone.group(1).upper() if lone and lone.group(1).upper() in allowed else None


def infer_constraint(user: str) -> tuple[str, str | None]:
    if 'Begin the response with exactly "Answer: ".' in user: return "exact_prefix", None
    if 'End the response with exactly " [END]".' in user: return "exact_suffix", None
    if "Wrap the entire response in <response> and </response> tags." in user: return "xml", None
    if 'Return a JSON object with exactly one field named "answer".' in user: return "json", None
    match = re.search(r'Include the exact keyword "([^"]+)"\.', user)
    if match: return "keyword", match.group(1)
    return "general", None


def check_constraint(prediction: str, kind: str, keyword: str | None = None) -> bool:
    if kind == "general": return bool(prediction.strip())
    if kind == "exact_prefix": return prediction.startswith("Answer: ")
    if kind == "exact_suffix": return prediction.endswith(" [END]")
    if kind == "xml": return prediction.startswith("<response>") and prediction.endswith("</response>")
    if kind == "keyword": return bool(keyword) and keyword in prediction
    if kind == "json":
        try: value = json.loads(prediction)
        except (json.JSONDecodeError, TypeError): return False
        return isinstance(value, dict) and list(value) == ["answer"] and isinstance(value["answer"], str)
    raise ValueError(f"unknown constraint: {kind}")


def token_f1(reference: str, prediction: str) -> float:
    left, right = normalize_text(reference).split(), normalize_text(prediction).split()
    if not left or not right: return float(left == right)
    overlap = sum((Counter(left) & Counter(right)).values())
    return 2 * overlap / (len(left) + len(right))


def code_blocks(value: str) -> list[tuple[str, str]]:
    return [(lang.casefold(), code.strip()) for lang, code in re.findall(r"```([\w+#.-]*)\s*\n(.*?)```", value, re.S)]


def securecode_checks(prediction: str) -> dict[str, Any]:
    blocks = code_blocks(prediction)
    syntax_results = []
    for language, code in blocks:
        if language in {"python", "py"}:
            try: ast.parse(code); syntax_results.append(True)
            except SyntaxError: syntax_results.append(False)
        elif language == "json":
            try: json.loads(code); syntax_results.append(True)
            except json.JSONDecodeError: syntax_results.append(False)
    risky = re.compile(r"\b(eval|exec)\s*\(|shell\s*=\s*True|strcpy\s*\(|gets\s*\(|verify\s*=\s*False", re.I)
    return {"code_block_count": len(blocks), "syntax_check_applicable": bool(syntax_results),
            "syntax_valid": all(syntax_results) if syntax_results else None,
            "security_proxy_pass": not bool(risky.search(prediction))}


def score_prediction(task: str, row: dict[str, Any], prediction: str) -> dict[str, Any]:
    reference, user = str(row["messages"][2]["content"]), str(row["messages"][1]["content"])
    if task == "math":
        ref_label, pred_label = extract_math_label(reference), extract_math_label(prediction)
        if ref_label is not None and pred_label is not None:
            return {"parsed_reference": ref_label, "parsed_prediction": pred_label, "parse_success": True,
                    "per_sample_score": float(ref_label == pred_label),
                    "details": {"scoring_mode": "choice_label", "extraction_method": "choice_label",
                                "reference_extraction_method": "choice_label",
                                "prediction_extraction_method": "choice_label"}}
        options = math_choice_values(user)
        if ref_label is not None or pred_label is not None:
            labelled, other_is_prediction = ((ref_label, True) if ref_label is not None else
                                               (pred_label, False))
            expected = options.get(str(labelled)) if options else None
            other_text = prediction if other_is_prediction else reference
            other, other_method = extract_math_answer_with_method(other_text)
            if other is None and options:
                other, other_method = extract_math_mapped_final_sentence(other_text, options)
            success = expected is not None and other is not None
            return {"parsed_reference": expected if other_is_prediction else other,
                    "parsed_prediction": other if other_is_prediction else expected,
                    "parse_success": success,
                    "per_sample_score": float(expected == other) if success else None,
                    "details": {"scoring_mode": "choice_value" if success else "unscorable",
                                "reference_label": ref_label, "prediction_label": pred_label,
                                "choice_mapping_recovered": bool(options),
                                "reference_extraction_method": "choice_label" if ref_label else other_method,
                                "prediction_extraction_method": other_method if ref_label else "choice_label",
                                "extraction_method": other_method if ref_label else "choice_label"}}
        ref, ref_method = extract_math_answer_with_method(reference)
        pred, pred_method = extract_math_answer_with_method(prediction)
        success = ref is not None and pred is not None
        return {"parsed_reference": ref, "parsed_prediction": pred, "parse_success": success,
                "per_sample_score": float(pred == ref) if success else None,
                "details": {"scoring_mode": "normalized_exact_match" if success else "unscorable",
                            "reference_extraction_method": ref_method,
                            "prediction_extraction_method": pred_method,
                            "extraction_method": pred_method}}
    if task == "bbh":
        ref, pred = extract_bbh_answer(reference), extract_bbh_answer(prediction)
        return {"parsed_reference": ref, "parsed_prediction": pred, "parse_success": pred is not None,
                "per_sample_score": float(ref is not None and pred == ref), "details": {}}
    if task in {"gpqa", "mmlupro"}:
        labels = choice_labels(user)
        ref, pred = parse_mcq_label(reference, labels), parse_mcq_label(prediction, labels)
        return {"parsed_reference": ref, "parsed_prediction": pred, "parse_success": pred is not None,
                "per_sample_score": float(ref is not None and pred == ref), "details": {}}
    if task == "ifeval":
        kind, keyword = infer_constraint(user); passed = check_constraint(prediction, kind, keyword)
        return {"parsed_reference": {"constraint": kind, "keyword": keyword},
                "parsed_prediction": {"constraint_satisfied": passed}, "parse_success": True,
                "per_sample_score": float(passed), "details": {"constraint": kind}}
    checks = securecode_checks(prediction)
    applicable = checks["syntax_check_applicable"]
    score = float(checks["syntax_valid"]) if applicable else token_f1(reference, prediction)
    return {"parsed_reference": {"code_blocks": len(code_blocks(reference))}, "parsed_prediction": checks,
            "parse_success": bool(prediction.strip()), "per_sample_score": score,
            "details": {**checks, "token_f1": token_f1(reference, prediction),
                        "text_similarity": SequenceMatcher(None, normalize_text(reference), normalize_text(prediction)).ratio(),
                        "exact_match": normalize_text(reference) == normalize_text(prediction)}}


def aggregate(task: str, predictions: list[dict[str, Any]], runtime: float, peak: int) -> dict[str, Any]:
    success = [p for p in predictions if not p["error"]]
    scores = [float(p["per_sample_score"]) for p in success if p["per_sample_score"] is not None]
    result: dict[str, Any] = {"task": task, "evaluated_rows": len(predictions),
        "successful_generations": len(success), "failed_generations": len(predictions) - len(success),
        "parse_success_count": sum(bool(p["parse_success"]) for p in success),
        "mean_prompt_tokens": sum(p.get("prompt_tokens", 0) for p in success) / len(success) if success else 0.0,
        "mean_generated_tokens": sum(p.get("generated_tokens", 0) for p in success) / len(success) if success else 0.0,
        "runtime": {"seconds": runtime}, "runtime_seconds": runtime,
        "samples_per_second": len(success) / runtime if runtime > 0 else 0.0,
        "peak_cuda_memory_bytes": peak}
    result["auxiliary_metrics"] = {
        "parse_success_rate": result["parse_success_count"] / len(success) if success else 0.0,
        "nonempty_response_rate": sum(bool(p["prediction"].strip()) for p in success) / len(success) if success else 0.0,
    }
    if task in {"math", "bbh"}: result["exact_match"] = sum(scores) / len(scores) if scores else 0.0
    elif task in {"gpqa", "mmlupro"}: result["label_accuracy"] = sum(scores) / len(scores) if scores else 0.0
    elif task == "ifeval":
        result["constraint_compliance_rate"] = sum(scores) / len(scores) if scores else 0.0
        groups: dict[str, list[float]] = {}
        for p in success: groups.setdefault(p["details"]["constraint"], []).append(float(p["per_sample_score"]))
        result["compliance_by_constraint"] = {k: sum(v)/len(v) for k, v in sorted(groups.items())}
    else:
        applicable = [p for p in success if p["details"]["syntax_check_applicable"]]
        result.update({"syntax_check_applicable_rows": len(applicable),
            "syntax_valid_rate": sum(bool(p["details"]["syntax_valid"]) for p in applicable)/len(applicable) if applicable else None,
            "security_heuristic_proxy_pass_rate": sum(bool(p["details"]["security_proxy_pass"]) for p in success)/len(success) if success else 0.0,
            "auxiliary_token_f1": sum(p["details"]["token_f1"] for p in success)/len(success) if success else 0.0,
            "auxiliary_exact_match": sum(p["details"]["exact_match"] for p in success)/len(success) if success else 0.0})
    result["metric_meaning_and_limitations"] = metric_notes(task)
    return result


def metric_notes(task: str) -> str:
    notes = {
        "math": "Exact match after heuristic final-answer extraction; equivalent forms and proof quality are not judged.",
        "bbh": "Normalized exact match on the final/[RESULT] answer; task-specific semantic equivalence is not judged.",
        "gpqa": "Accuracy of the parsed choice label only; reasoning quality is not judged.",
        "mmlupro": "Accuracy of the parsed choice label only; reasoning quality is not judged.",
        "ifeval": "Rule compliance for the synthetic constraint encoded in the user prompt; general rows only require a non-empty response.",
        "securecode": "Syntax validity is checked only for fenced Python/JSON. Security is a narrow risky-pattern heuristic and token/text metrics are proxies, not proof of correctness or security.",
    }
    return notes[task]


def _prediction(row: dict[str, Any], task: str, prediction: str, prompt_tokens: int,
                generated_tokens: int, error: str | None = None, *, reached_max_new_tokens: bool = False,
                eos_terminated: bool = False) -> dict[str, Any]:
    reference = str(row["messages"][2]["content"])
    scored = score_prediction(task, row, prediction) if not error else {
        "parsed_reference": None, "parsed_prediction": None, "per_sample_score": None,
        "parse_success": False, "details": {}}
    truncated = reached_max_new_tokens and not eos_terminated
    if truncated:
        scored["per_sample_score"] = None
    return {"task": task, "sample_id": row.get("sample_id") or row.get("prompt_hash"),
        "prompt_hash": row.get("prompt_hash"), "source_group": row.get("source_group", row.get("source_dataset", "unknown")),
        "reference": reference, "prediction": prediction, **scored, "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens, "actual_generated_tokens": generated_tokens,
        "reached_max_new_tokens": reached_max_new_tokens, "eos_terminated": eos_terminated,
        "generation_truncated": truncated,
        "scoring_error": "generation_truncated" if truncated else None,
        "error": error}


def decode_generated_only(tokenizer: Any, generated_ids: Any, padded_prompt_width: int) -> tuple[str, int]:
    """Decode only tokens appended by generate, including for left-padded batches."""
    new_ids = generated_ids[padded_prompt_width:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip(), int(new_ids.numel())


def generation_termination(generated_ids: Any, padded_prompt_width: int, max_new_tokens: int,
                           eos_token_id: int | list[int] | tuple[int, ...] | None) -> dict[str, bool]:
    new_ids = generated_ids[padded_prompt_width:]
    count = int(new_ids.numel())
    eos_ids = set(eos_token_id if isinstance(eos_token_id, (list, tuple)) else
                  ([] if eos_token_id is None else [eos_token_id]))
    values = new_ids.detach().cpu().tolist() if hasattr(new_ids, "detach") else list(new_ids.values)
    eos_terminated = bool(values and eos_ids and values[-1] in eos_ids)
    reached = count >= max_new_tokens
    return {"reached_max_new_tokens": reached, "eos_terminated": eos_terminated,
            "generation_truncated": reached and not eos_terminated}


def generate_batches(rows: list[dict[str, Any]], task: str, adapter: Path, batch_size: int,
                     max_new_tokens: int, base_model: str = BASE_MODEL) -> tuple[list[dict[str, Any]], int]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from train_task_adapter import inspect_adapter_artifact
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required for FP16 adapter evaluation")
    set_seed(SEED); torch.cuda.reset_peak_memory_stats()
    inspect_adapter_artifact(adapter, require_finite=True)
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False).to("cuda").eval()
    predictions = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        rendered = [tokenizer.apply_chat_template(prompt_messages(row), tokenize=False, add_generation_prompt=True) for row in batch]
        encoded = tokenizer(rendered, return_tensors="pt", padding=True).to(model.device)
        input_width = encoded["input_ids"].shape[1]
        try:
            with torch.inference_mode():
                output = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                                        pad_token_id=tokenizer.pad_token_id)
            for index, row in enumerate(batch):
                text, generated_count = decode_generated_only(tokenizer, output[index], input_width)
                termination = generation_termination(output[index], input_width, max_new_tokens,
                                                     tokenizer.eos_token_id)
                error = "empty_response" if not text else None
                predictions.append(_prediction(row, task, text, int(encoded["attention_mask"][index].sum().item()),
                                               generated_count, error, **{k: termination[k] for k in
                                               ("reached_max_new_tokens", "eos_terminated")}))
        except torch.cuda.OutOfMemoryError as exc:
            logging.exception("CUDA OOM; no fallback attempted")
            predictions.extend(_prediction(row, task, "", 0, 0, f"cuda_oom: {exc}") for row in batch)
            for row in rows[start + len(batch):]: predictions.append(_prediction(row, task, "", 0, 0, "not_run_after_cuda_oom"))
            break
    return predictions, int(torch.cuda.max_memory_allocated())


def evaluate(args: argparse.Namespace) -> Path:
    if args.max_samples is not None and args.max_samples <= 0: raise ValueError("--max-samples must be positive")
    if args.batch_size <= 0 or args.max_new_tokens <= 0: raise ValueError("batch size and max new tokens must be positive")
    output = args.output
    if output.exists() and not args.overwrite: raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for name in ("predictions.jsonl", "metrics.json", "run_config.json", "evaluation.log"):
            target = output / name
            if target.is_file(): target.unlink()
    logging.basicConfig(filename=output/"evaluation.log", level=logging.INFO, force=True,
                        format="%(asctime)s %(levelname)s %(message)s")
    data_path = args.data_root / f"{args.split}.jsonl"
    before = file_fingerprint(data_path); rows = load_rows(data_path, args.task, args.max_samples)
    model_name = getattr(args, "model", BASE_MODEL)
    config = {"task": args.task, "adapter": str(args.adapter), "data_root": str(args.data_root),
        "split": args.split, "max_samples": args.max_samples, "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens, "output": str(output), "base_model": model_name,
        "dtype": "float16", "seed": SEED, "do_sample": False, "automatic_quantization_fallback": False,
        "input_fingerprint_before": before}
    (output/"run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    started = time.perf_counter(); peak = 0
    try: predictions, peak = generate_batches(rows, args.task, args.adapter, args.batch_size, args.max_new_tokens, model_name)
    except Exception as exc:
        logging.exception("model/adapter loading or evaluation failed")
        predictions = [_prediction(row, args.task, "", 0, 0, f"loading_or_runtime_failure: {type(exc).__name__}: {exc}") for row in rows]
    runtime = time.perf_counter() - started
    after = file_fingerprint(data_path)
    if before != after: raise RuntimeError(f"input test data changed during evaluation: before={before}, after={after}")
    with (output/"predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions: handle.write(json.dumps(row, ensure_ascii=False)+"\n")
    metrics = aggregate(args.task, predictions, runtime, peak)
    metrics["input_fingerprint_unchanged"] = True
    (output/"metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    logging.info("completed metrics=%s", metrics)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=BASE_MODEL, help="HF model ID or local snapshot")
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--max-samples", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__": evaluate(parse_args())
