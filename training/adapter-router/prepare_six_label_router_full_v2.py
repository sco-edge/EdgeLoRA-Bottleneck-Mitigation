#!/usr/bin/env python3
"""Freeze completed cross-profiling results into the six-label Router dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

TASKS = ("securecode", "math", "ifeval", "bbh", "gpqa", "mmlupro")
ADAPTER_MAPPING = {task: index for index, task in enumerate(TASKS)}
ADAPTER_IDS = tuple(range(6))
TRAINING_GROUPS = ("shard000", "shard001", "shard003", "shard004", "shard005",
                   "shard006", "shard007", "shard008", "shard009")
VALIDATION_GROUPS = ("shard010", "shard011")
REQUIRED_GROUPS = TRAINING_GROUPS + VALIDATION_GROUPS
EXCLUDED_GROUPS = {
    "shard002": "frozen e6 independent evaluation only",
    "shard012": "reserved and untouched for final independent testing",
    "shard013": "reserved and untouched for final independent testing",
}
REQUIRED_FILES = ("cross_predictions.jsonl", "routing_labels.jsonl", "summary.json",
                  "run_config.json")
OUTPUT_DATA_FILES = ("train.jsonl", "validation.jsonl", "train_weak_only.jsonl",
                     "validation_weak_only.jsonl")
GROUP_RE = re.compile(r"(?<![A-Za-z0-9])(shard\d{3})(?!\d)")


def expected_date(group: str) -> str:
    return "20260828" if group in ("shard000", "shard001") else "20260901"


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _validate_finite(value: Any, where: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"non-finite numeric value at {where}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite(item, f"{where}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite(item, f"{where}.{key}")
        return
    raise ValueError(f"unsupported JSON value at {where}: {type(value).__name__}")


def _loads(text: str, where: str) -> Any:
    try:
        value = json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_pairs)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid JSON at {where}: {error}") from error
    _validate_finite(value, where)
    return value


def read_json(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"invalid UTF-8: {path}") from error
    value = _loads(text, str(path))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        handle = path.open("r", encoding="utf-8")
        with handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = _loads(line, f"{path}:{number}")
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL row must be an object: {path}:{number}")
                rows.append(value)
    except UnicodeDecodeError as error:
        raise ValueError(f"invalid UTF-8: {path}") from error
    return rows


def fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    rows = len(read_jsonl(path)) if path.suffix == ".jsonl" else 1
    return {"path": str(path.resolve()), "rows": rows, "size": path.stat().st_size,
            "sha256": digest.hexdigest()}


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _directory_group(path: Path) -> str | None:
    matches = set(GROUP_RE.findall(path.name))
    return next(iter(matches)) if len(matches) == 1 else None


def validate_profiling_directory(path: Path, group: str) -> Path:
    if group in EXCLUDED_GROUPS:
        raise ValueError(f"forbidden profiling group {group}: {EXCLUDED_GROUPS[group]}")
    if group not in REQUIRED_GROUPS:
        raise ValueError(f"unsupported profiling group: {group}")
    if not path.is_dir():
        raise FileNotFoundError(f"missing profiling directory for {group}: {path}")
    actual_group = _directory_group(path)
    if actual_group != group:
        raise ValueError(f"profiling directory/group mismatch for {group}: {path}")
    date = expected_date(group)
    if date not in path.name:
        if "20260831" in path.name:
            raise ValueError(f"20260831 profiling output is forbidden for {group}: {path}")
        raise ValueError(f"{group} must use the {date} profiling output: {path}")
    if "20260831" in str(path):
        raise ValueError(f"20260831 profiling output is forbidden: {path}")
    forbidden_ancestors = sorted({match for part in path.parts for match in GROUP_RE.findall(part)
                                  if match in EXCLUDED_GROUPS})
    if forbidden_ancestors:
        raise ValueError(f"forbidden profiling group in input path: {forbidden_ancestors}: {path}")
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete profiling directory for {group}; missing {missing}: {path}")
    return path.resolve()


def discover_profiling_directories(evaluations_root: Path) -> dict[str, Path]:
    if not evaluations_root.is_dir():
        raise FileNotFoundError(f"profiling evaluations root does not exist: {evaluations_root}")
    resolved: dict[str, Path] = {}
    children = sorted((path for path in evaluations_root.rglob("*") if path.is_dir()), key=str)
    for group in REQUIRED_GROUPS:
        date = expected_date(group)
        candidates = [path for path in children if _directory_group(path) == group and date in path.name
                      and "20260831" not in str(path)
                      and all((path / name).is_file() for name in REQUIRED_FILES)]
        if not candidates:
            seen = [path for path in children if _directory_group(path) == group]
            detail = ", ".join(map(str, seen)) if seen else "none"
            raise FileNotFoundError(
                f"no valid completed {date} profiling directory for {group}; discovered: {detail}")
        if len(candidates) != 1:
            raise ValueError(f"ambiguous profiling directories for {group}: " +
                             ", ".join(map(str, candidates)))
        resolved[group] = validate_profiling_directory(candidates[0], group)
    return resolved


def validate_directory_map(profiling_dirs: dict[str, Path]) -> dict[str, Path]:
    forbidden = sorted(set(profiling_dirs) & set(EXCLUDED_GROUPS))
    if forbidden:
        raise ValueError("forbidden profiling groups supplied: " + ", ".join(forbidden))
    missing = sorted(set(REQUIRED_GROUPS) - set(profiling_dirs))
    extra = sorted(set(profiling_dirs) - set(REQUIRED_GROUPS))
    if missing or extra:
        raise ValueError(f"profiling groups must be exactly {list(REQUIRED_GROUPS)}; "
                         f"missing={missing}, extra={extra}")
    return {group: validate_profiling_directory(Path(profiling_dirs[group]), group)
            for group in REQUIRED_GROUPS}


def _string(row: dict[str, Any], field: str, where: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing or empty {field}: {where}")
    return value


def _adapter_ids(value: Any, where: str) -> list[int]:
    if not isinstance(value, list) or any(isinstance(item, bool) or not isinstance(item, int)
                                         or item not in ADAPTER_IDS for item in value):
        raise ValueError(f"invalid ground_truth_adapter_ids: {where}")
    if len(value) != len(set(value)):
        raise ValueError(f"duplicate ground_truth_adapter_ids: {where}")
    return sorted(value)


def _exact_positive(candidate: dict[str, Any], where: str) -> tuple[bool, str]:
    truncated = candidate.get("generation_truncated")
    parse_success = candidate.get("parse_success")
    if not isinstance(truncated, bool) or not isinstance(parse_success, bool):
        raise ValueError(f"candidate must contain boolean parse_success/generation_truncated: {where}")
    if truncated:
        return False, "generation_truncated"
    if not parse_success:
        return False, "parse_failed_or_unscorable"
    if candidate.get("scoring_error") is not None:
        return False, "scoring_error"
    score = candidate.get("generation_score")
    if score is None:
        return False, "missing_or_unscorable_score"
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError(f"generation_score must be finite numeric or null: {where}")
    score = float(score)
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"generation_score outside [0,1]: {where}")
    if score == 1.0:
        return True, "exact_positive"
    return False, "fractional_score" if score > 0.0 else "zero_score"


def _input_path(config: dict[str, Any], task: str, repo_root: Path, where: str) -> Path:
    roots = config.get("data_roots")
    if not isinstance(roots, dict) or not isinstance(roots.get(task), str) or not roots[task]:
        raise ValueError(f"run_config data_roots missing {task}: {where}")
    root = Path(roots[task])
    if not root.is_absolute():
        root = repo_root / root
    return (root / "test.jsonl").resolve()


def _prompt_from_input(row: dict[str, Any], where: str) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"profiling input messages must be a list: {where}")
    prompts = [message.get("content") for message in messages
               if isinstance(message, dict) and message.get("role") == "user"]
    if len(prompts) != 1 or not isinstance(prompts[0], str) or not prompts[0].strip():
        raise ValueError(f"profiling input must contain one nonempty user prompt: {where}")
    return prompts[0]


def _distribution(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def _exclusion_distribution(values: Counter[tuple[str, str]],
                            task: str | None = None) -> dict[str, int]:
    reasons = Counter()
    for (item_task, reason), count in values.items():
        if task is None or item_task == task:
            reasons[reason] += count
    return dict(sorted(reasons.items()))


def _validate_group(group: str, directory: Path, repo_root: Path,
                    input_records: dict[str, dict[str, Any]],
                    recovery_records: dict[str, dict[str, Any]]) -> tuple[
                        list[dict[str, Any]], list[dict[str, Any]], Counter[tuple[str, str]]]:
    candidate_path = directory / "cross_predictions.jsonl"
    routing_path = directory / "routing_labels.jsonl"
    summary_path = directory / "summary.json"
    config_path = directory / "run_config.json"
    candidates = read_jsonl(candidate_path)
    routing = read_jsonl(routing_path)
    summary = read_json(summary_path)
    config = read_json(config_path)
    if config.get("adapter_ids") != ADAPTER_MAPPING:
        raise ValueError(f"run_config does not preserve the fixed adapter mapping: {config_path}")
    for path in (candidate_path, routing_path, summary_path, config_path):
        input_records[str(path.resolve())] = fingerprint(path)
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for number, candidate in enumerate(candidates, 1):
        where = f"{candidate_path}:{number}"
        prompt_hash = _string(candidate, "prompt_hash", where)
        _string(candidate, "source_group", where)
        source_task = _string(candidate, "source_task", where)
        if source_task not in TASKS:
            raise ValueError(f"invalid source_task: {where}")
        adapter_id = candidate.get("adapter_id")
        if isinstance(adapter_id, bool) or not isinstance(adapter_id, int) or adapter_id not in ADAPTER_IDS:
            raise ValueError(f"invalid adapter_id: {where}")
        if candidate.get("adapter_task") != TASKS[adapter_id]:
            raise ValueError(f"adapter_task/adapter_id mapping mismatch: {where}")
        _exact_positive(candidate, where)
        mean_nll = candidate.get("mean_nll")
        if mean_nll is not None and (isinstance(mean_nll, bool) or
                                     not isinstance(mean_nll, (int, float))):
            raise ValueError(f"mean_nll must be finite numeric or null: {where}")
        by_hash[prompt_hash].append(candidate)
    label_by_hash: dict[str, dict[str, Any]] = {}
    for number, label in enumerate(routing, 1):
        where = f"{routing_path}:{number}"
        prompt_hash = _string(label, "prompt_hash", where)
        if prompt_hash in label_by_hash:
            raise ValueError(f"duplicate prompt_hash in profiling group {group}: {prompt_hash}")
        _string(label, "source_group", where)
        if _string(label, "source_task", where) not in TASKS:
            raise ValueError(f"invalid source_task: {where}")
        if not isinstance(label.get("resolved"), bool):
            raise ValueError(f"resolved must be boolean: {where}")
        _adapter_ids(label.get("ground_truth_adapter_ids"), where)
        label_by_hash[prompt_hash] = label
    if set(by_hash) != set(label_by_hash):
        raise ValueError(f"routing/candidate prompt_hash sets differ in {group}")

    input_indexes: dict[str, dict[str, dict[str, Any]]] = {}
    exact_rows, weak_rows = [], []
    exclusions: Counter[tuple[str, str]] = Counter()
    for prompt_hash, label in label_by_hash.items():
        values = by_hash[prompt_hash]
        ids = [value["adapter_id"] for value in values]
        if len(values) != 6 or sorted(ids) != list(ADAPTER_IDS) or len(set(ids)) != 6:
            raise ValueError(f"prompt {prompt_hash} must have adapter IDs 0 through 5 exactly once")
        source_groups = {value["source_group"] for value in values}
        source_tasks = {value["source_task"] for value in values}
        if len(source_groups) != 1 or len(source_tasks) != 1:
            raise ValueError(f"conflicting source metadata across six candidates: {prompt_hash}")
        source_group, source_task = next(iter(source_groups)), next(iter(source_tasks))
        if label["source_group"] != source_group or label["source_task"] != source_task:
            raise ValueError(f"candidate/routing metadata mismatch: {prompt_hash}")

        stored_prompts = [value.get("user_prompt") for value in values
                          if value.get("user_prompt") is not None]
        if any(not isinstance(value, str) or not value.strip() for value in stored_prompts):
            raise ValueError(f"empty or invalid user_prompt: {prompt_hash}")
        if len(set(stored_prompts)) > 1:
            raise ValueError(f"conflicting prompts across six candidates: {prompt_hash}")
        prompt = stored_prompts[0] if stored_prompts else None
        recovery_source = None
        if len(stored_prompts) != 6:
            input_path = _input_path(config, source_task, repo_root, str(config_path))
            key = str(input_path)
            if key not in input_indexes:
                if not input_path.is_file():
                    raise FileNotFoundError(f"missing profiling input for prompt recovery: {input_path}")
                recovery_records[key] = fingerprint(input_path)
                rows = read_jsonl(input_path)
                index: dict[str, dict[str, Any]] = {}
                for number, row in enumerate(rows, 1):
                    value = _string(row, "prompt_hash", f"{input_path}:{number}")
                    if value in index:
                        raise ValueError(f"duplicate prompt_hash in profiling input: {value}")
                    index[value] = row
                input_indexes[key] = index
            recovered_row = input_indexes[key].get(prompt_hash)
            if recovered_row is None:
                raise ValueError(f"prompt_hash absent from profiling input: {prompt_hash}")
            if recovered_row.get("source_group") != source_group:
                raise ValueError(f"profiling input source_group mismatch: {prompt_hash}")
            if recovered_row.get("task", source_task) != source_task:
                raise ValueError(f"profiling input source_task mismatch: {prompt_hash}")
            recovered = _prompt_from_input(recovered_row, f"{input_path}:{prompt_hash}")
            if prompt is not None and prompt != recovered:
                raise ValueError(f"stored/recovered prompt conflict: {prompt_hash}")
            prompt = recovered
            recovery_source = key
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"missing prompt: {prompt_hash}")
        for value in values:
            candidate_prompt = value.get("prompt")
            if candidate_prompt is not None and candidate_prompt != prompt:
                raise ValueError(f"conflicting prompt and user_prompt: {prompt_hash}")

        positives = []
        for value in values:
            positive, reason = _exact_positive(value, f"{candidate_path}:{prompt_hash}")
            if not positive:
                exclusions[(source_task, reason)] += 1
            if positive:
                positives.append(value["adapter_id"])
        positives.sort()
        routed_ids = _adapter_ids(label["ground_truth_adapter_ids"], prompt_hash)
        if positives != routed_ids or label["resolved"] is not bool(positives):
            raise ValueError(f"exact-label recomputation mismatch for {prompt_hash}: "
                             f"computed={positives}, routed={routed_ids}, resolved={label['resolved']}")
        finite_nll = [(float(value["mean_nll"]), int(value["adapter_id"])) for value in values
                      if value.get("mean_nll") is not None]
        weak = min(finite_nll) if finite_nll else None
        routed_weak_id = label.get("weak_label_adapter_id")
        routed_weak_nll = label.get("weak_label_mean_nll")
        if routed_weak_id is not None and (isinstance(routed_weak_id, bool) or
                                           not isinstance(routed_weak_id, int) or
                                           routed_weak_id not in ADAPTER_IDS):
            raise ValueError(f"invalid weak_label_adapter_id for {prompt_hash}")
        if weak is None:
            if routed_weak_id is not None or routed_weak_nll is not None:
                raise ValueError(f"weak-label recomputation mismatch for {prompt_hash}")
        elif (routed_weak_id != weak[1] or isinstance(routed_weak_nll, bool) or
              not isinstance(routed_weak_nll, (int, float)) or float(routed_weak_nll) != weak[0]):
            raise ValueError(f"weak-label recomputation mismatch for {prompt_hash}: "
                             f"computed={weak}, routed={(routed_weak_nll, routed_weak_id)}")
        common = {"prompt_hash": prompt_hash, "source_group": source_group,
                  "source_task": source_task, "prompt": prompt, "user_prompt": prompt,
                  "weak_label_adapter_id": weak[1] if weak else None,
                  "weak_label_mean_nll": weak[0] if weak else None,
                  "profiling_group": group}
        if recovery_source:
            common["prompt_recovery_source"] = recovery_source
        if positives:
            exact_rows.append({**common, "labels": [1 if index in positives else 0 for index in ADAPTER_IDS],
                               "ground_truth_adapter_ids": positives, "label_width": len(positives),
                               "label_type": "exact", "auxiliary_only": False})
        elif weak is not None:
            weak_rows.append({**common, "labels": None, "ground_truth_adapter_ids": [],
                              "label_width": 0, "label_type": "weak_only", "auxiliary_only": True,
                              "weak_label_reason": "lowest_finite_reference_mean_nll"})
    if summary.get("candidate_rows") is not None and summary["candidate_rows"] != len(candidates):
        raise ValueError(f"summary candidate_rows mismatch: {directory}")
    if summary.get("source_samples") is not None and summary["source_samples"] != len(routing):
        raise ValueError(f"summary source_samples mismatch: {directory}")
    return exact_rows, weak_rows, exclusions


def _check_split_uniqueness(rows: list[dict[str, Any]], split: str) -> tuple[set[str], set[str]]:
    hashes, groups = set(), set()
    for row in rows:
        if row["prompt_hash"] in hashes:
            raise ValueError(f"duplicate prompt_hash within {split}: {row['prompt_hash']}")
        if row["source_group"] in groups:
            raise ValueError(f"duplicate source_group within {split}: {row['source_group']}")
        hashes.add(row["prompt_hash"])
        groups.add(row["source_group"])
    return hashes, groups


def build(profiling_dirs: dict[str, Path], output_dir: Path, repo_root: Path,
          write_jsonl_fn: Callable[[Path, Iterable[dict[str, Any]]], None] = write_jsonl) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    directories = validate_directory_map(profiling_dirs)
    input_records: dict[str, dict[str, Any]] = {}
    for group in REQUIRED_GROUPS:
        for name in REQUIRED_FILES:
            path = directories[group] / name
            input_records[str(path.resolve())] = fingerprint(path)
    before = dict(input_records)
    recovery_records: dict[str, dict[str, Any]] = {}
    exact_by_split = {"train": [], "validation": []}
    weak_by_split = {"train": [], "validation": []}
    exclusion_by_split: dict[str, Counter[tuple[str, str]]] = {
        "train": Counter(), "validation": Counter()}
    for group in REQUIRED_GROUPS:
        split = "train" if group in TRAINING_GROUPS else "validation"
        exact, weak, exclusions = _validate_group(
            group, directories[group], repo_root.resolve(), input_records, recovery_records)
        for row in exact + weak:
            row["split"] = split
        exact_by_split[split].extend(exact)
        weak_by_split[split].extend(weak)
        exclusion_by_split[split].update(exclusions)
    before.update(recovery_records)
    train_all = exact_by_split["train"] + weak_by_split["train"]
    validation_all = exact_by_split["validation"] + weak_by_split["validation"]
    train_hashes, train_groups = _check_split_uniqueness(train_all, "training")
    validation_hashes, validation_groups = _check_split_uniqueness(validation_all, "validation")
    hash_overlap = train_hashes & validation_hashes
    group_overlap = train_groups & validation_groups
    if hash_overlap:
        raise ValueError(f"training/validation prompt_hash overlap: {sorted(hash_overlap)}")
    if group_overlap:
        raise ValueError(f"training/validation source_group overlap: {sorted(group_overlap)}")
    if any(row["profiling_group"] in EXCLUDED_GROUPS for row in train_all + validation_all):
        raise RuntimeError("excluded profiling group reached output rows")
    if any(len(row["labels"]) != 6 or sum(row["labels"]) != row["label_width"]
           or not 1 <= row["label_width"] <= 6
           for rows in exact_by_split.values() for row in rows):
        raise RuntimeError("invalid exact multi-hot output")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        payloads = {
            "train.jsonl": exact_by_split["train"],
            "validation.jsonl": exact_by_split["validation"],
            "train_weak_only.jsonl": weak_by_split["train"],
            "validation_weak_only.jsonl": weak_by_split["validation"],
        }
        output_records = []
        for name in OUTPUT_DATA_FILES:
            path = staging / name
            write_jsonl_fn(path, payloads[name])
            if read_jsonl(path) != payloads[name]:
                raise RuntimeError(f"UTF-8 JSONL reload mismatch: {name}")
            record = fingerprint(path)
            record["path"] = str((output_dir / name).resolve())
            output_records.append(record)
        current = {path: fingerprint(Path(path)) for path in before}
        if current != before:
            raise RuntimeError("input row count or SHA-256 changed during execution")
        overlap_checks = {
            "training_prompt_hash_duplicates": len(train_all) - len(train_hashes),
            "training_source_group_duplicates": len(train_all) - len(train_groups),
            "validation_prompt_hash_duplicates": len(validation_all) - len(validation_hashes),
            "validation_source_group_duplicates": len(validation_all) - len(validation_groups),
            "train_validation_prompt_hash_overlap": len(hash_overlap),
            "train_validation_source_group_overlap": len(group_overlap),
        }
        aggregate_exclusions = {split: _exclusion_distribution(exclusion_by_split[split])
                                for split in ("train", "validation")}
        per_task = {
            task: {
                split: {
                    "exact_rows": sum(row["source_task"] == task for row in exact_by_split[split]),
                    "weak_only_rows": sum(row["source_task"] == task for row in weak_by_split[split]),
                    "label_width_distribution": _distribution(
                        [row for row in exact_by_split[split] if row["source_task"] == task],
                        "label_width"),
                    "scoring_exclusion_reasons": _exclusion_distribution(
                        exclusion_by_split[split], task),
                }
                for split in ("train", "validation")
            }
            for task in TASKS
        }
        stats = {
            "counts": {split: {"exact": len(exact_by_split[split]),
                                "weak_only": len(weak_by_split[split])}
                       for split in ("train", "validation")},
            "exact_rows": sum(map(len, exact_by_split.values())),
            "weak_only_rows": sum(map(len, weak_by_split.values())),
            "label_width_distribution": {split: _distribution(exact_by_split[split], "label_width")
                                           for split in ("train", "validation")},
            "source_task_distribution": {
                split: {"exact": _distribution(exact_by_split[split], "source_task"),
                        "weak_only": _distribution(weak_by_split[split], "source_task")}
                for split in ("train", "validation")},
            "scoring_exclusion_reasons": aggregate_exclusions,
            "aggregate": {
                "exact_rows": sum(map(len, exact_by_split.values())),
                "weak_only_rows": sum(map(len, weak_by_split.values())),
                "label_width_distribution": _distribution(
                    exact_by_split["train"] + exact_by_split["validation"], "label_width"),
                "scoring_exclusion_reasons": _exclusion_distribution(
                    exclusion_by_split["train"] + exclusion_by_split["validation"]),
            },
            "per_task": per_task,
            "overlap_checks": overlap_checks,
        }
        write_json(staging / "stats.json", stats)
        stats_record = fingerprint(staging / "stats.json")
        stats_record["path"] = str((output_dir / "stats.json").resolve())
        manifest = {
            "dataset": "six_label_router_full_v2",
            "purpose": "frozen six-label Router training and validation data from completed cross-profiling",
            "fixed_adapter_mapping": ADAPTER_MAPPING,
            "exact_label_policy": ("Only resolved rows whose recomputed exact positives are nonempty; "
                                   "generation_score == 1.0 with successful parsing, no scoring error, "
                                   "and no truncation; preserve widths 1 through 6 as six-dimensional multi-hot labels."),
            "weak_only_policy": ("Rows without exact positives are auxiliary-only and written only when the "
                                  "lowest finite mean_nll weak label is valid; default Router data excludes them."),
            "training_groups": list(TRAINING_GROUPS),
            "validation_groups": list(VALIDATION_GROUPS),
            "excluded_groups": EXCLUDED_GROUPS,
            "resolved_input_paths": {group: str(directories[group]) for group in REQUIRED_GROUPS},
            "input_files_before": list(before.values()),
            "input_files_after": list(current.values()),
            "input_fingerprints_unchanged": True,
            "prompt_recovery_sources": list(recovery_records.values()),
            "output_files": output_records + [stats_record],
            "output_row_counts": {record["path"]: record["rows"] for record in output_records},
            "output_sha256": {record["path"]: record["sha256"] for record in output_records},
            "label_width_distributions": stats["label_width_distribution"],
            "source_task_distributions": stats["source_task_distribution"],
            "exact_and_weak_only_counts": stats["counts"],
            "overlap_check_results": overlap_checks,
            "parser_and_profiling_artifacts_are_frozen": True,
            "shard002_notice": "shard002 was used only for the already frozen e6 independent evaluation",
            "final_test_reservation": "shard012 and shard013 remain untouched for final testing",
            "evaluation_warning": "This dataset does not represent final test performance.",
            "directory_policy": ("shard000/shard001 use 20260828; shard003-shard011 use the completed "
                                 "no-CUBLAS 20260901 outputs; 20260831 is forbidden."),
        }
        write_json(staging / "manifest.json", manifest)
        read_json(staging / "manifest.json")
        read_json(staging / "stats.json")
        final_current = {path: fingerprint(Path(path)) for path in before}
        if final_current != before:
            raise RuntimeError("input row count or SHA-256 changed before publish")
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir


def parse_directory_values(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--profiling-dir must use GROUP=PATH")
        group, raw_path = value.split("=", 1)
        if not raw_path or group in result:
            raise ValueError(f"invalid or duplicate --profiling-dir: {value}")
        result[group] = Path(raw_path)
    return result


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluations-root", type=Path,
                        default=repo_root / "adapter-training/evaluations/cross-profile")
    parser.add_argument("--profiling-dir", action="append", default=[], metavar="GROUP=PATH")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "processed/six_label_router_full_v2")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")
    directories = (validate_directory_map(parse_directory_values(args.profiling_dir))
                   if args.profiling_dir else discover_profiling_directories(args.evaluations_root))
    build(directories, args.output_dir, args.repo_root)


if __name__ == "__main__":
    main()
