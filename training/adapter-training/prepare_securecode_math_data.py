#!/usr/bin/env python3
"""Prepare answer-bearing SecureCode and MathInstruct task-adapter data.

Only original task-training records are accepted.  Router prompt/label CSVs are
deliberately outside this program's input surface.  Inputs are read-only and
an existing output directory is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import shutil
import statistics
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

SEED = 42
DEFAULT_MATH_SPLIT_COUNTS = {"train": 8000, "validation": 500, "test": 500}
TASKS = ("securecode", "math")
SYSTEM = {
    "securecode": "Provide secure, correct code and explain the relevant security reasoning.",
    "math": "Solve the mathematics problem carefully and give the correct final answer.",
}
EVALUATION_SOURCE = re.compile(r"(?:^|[/_.-])(?:test|eval|evaluation|validation|valid)(?:[/_.-]|$)", re.I)


def normalize_prompt(value: str) -> str:
    return " ".join(str(value).split()).strip()


def prompt_hash(value: str) -> str:
    return hashlib.sha256(normalize_prompt(value).encode("utf-8")).hexdigest()


def make_row(task: str, dataset: str, group: str, user: str, assistant: str) -> dict[str, Any]:
    user, assistant = str(user).strip(), str(assistant).strip()
    if not user or not assistant:
        raise ValueError("empty user or assistant answer")
    return {
        "messages": [
            {"role": "system", "content": SYSTEM[task]},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "task": task,
        "source_dataset": dataset,
        "source_group": group,
        "prompt_hash": prompt_hash(user),
    }


def iter_json_array(path: Path, chunk_size: int = 1 << 20) -> Iterator[Any]:
    """Stream a top-level JSON array without loading it as a Python list."""
    decoder, buffer, pos, started, eof = json.JSONDecoder(), "", 0, False, False
    with path.open(encoding="utf-8") as handle:
        while True:
            if pos:
                buffer, pos = buffer[pos:], 0
            chunk = handle.read(chunk_size)
            if chunk:
                buffer += chunk
            else:
                eof = True
            while True:
                while pos < len(buffer) and (buffer[pos].isspace() or buffer[pos] == ","): pos += 1
                if not started:
                    if pos >= len(buffer): break
                    if buffer[pos] != "[": raise ValueError(f"{path}: expected top-level JSON array")
                    started, pos = True, pos + 1
                    continue
                while pos < len(buffer) and (buffer[pos].isspace() or buffer[pos] == ","): pos += 1
                if pos < len(buffer) and buffer[pos] == "]":
                    pos += 1
                    if buffer[pos:].strip() or (not eof and handle.read().strip()):
                        raise ValueError(f"{path}: trailing garbage")
                    return
                try:
                    value, end = decoder.raw_decode(buffer, pos)
                except json.JSONDecodeError as exc:
                    if eof: raise ValueError(f"{path}: invalid or truncated JSON array") from exc
                    break
                yield value
                pos = end
                while pos < len(buffer) and buffer[pos].isspace(): pos += 1
                if pos < len(buffer) and buffer[pos] == ",": pos += 1
                elif pos < len(buffer) and buffer[pos] != "]":
                    raise ValueError(f"{path}: expected comma or closing bracket")
            if eof: raise ValueError(f"{path}: missing closing bracket")


def prepare_securecode(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Use one answer-bearing first exchange per original conversation group."""
    rows, excluded = [], Counter()
    for domain in ("web", "aiml"):
        path = root / domain / "train.jsonl"
        if not path.is_file(): raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                record = json.loads(line)
                record_id = str(record.get("id", "")).strip()
                turns = record.get("conversations")
                if not record_id:
                    excluded["missing_id"] += 1; continue
                if not isinstance(turns, list):
                    excluded["missing_conversations"] += 1; continue
                user = next((m.get("content") for m in turns if m.get("role") in {"human", "user"}), None)
                answer = next((m.get("content") for m in turns if m.get("role") in {"assistant", "gpt"}), None)
                if not normalize_prompt(user or "") or not str(answer or "").strip():
                    excluded["missing_answer_pair"] += 1; continue
                rows.append(make_row("securecode", "scthornton/securecode",
                                     f"scthornton/securecode:{domain}:{record_id}", user, answer))
    return rows, {"raw_valid_rows": len(rows), "excluded": dict(excluded)}


def prepare_math(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, excluded, excluded_sources, sources = [], Counter(), Counter(), Counter()
    for index, record in enumerate(iter_json_array(path)):
        if not isinstance(record, dict):
            excluded["not_object"] += 1; continue
        source = normalize_prompt(record.get("source", ""))
        user = normalize_prompt(record.get("instruction", ""))
        answer = str(record.get("output", "")).strip()
        if not source or not user or not answer:
            excluded["missing_source_instruction_or_output"] += 1; continue
        if EVALUATION_SOURCE.search(source):
            excluded["evaluation_source"] += 1; excluded_sources[source] += 1; continue
        sources[source] += 1
        rows.append(make_row("math", "TIGER-Lab/MathInstruct",
                             f"TIGER-Lab/MathInstruct:{index}", user, answer))
        rows[-1]["_source"] = source
    return rows, {"raw_valid_rows": len(rows), "excluded": dict(excluded),
                  "excluded_source_counts": dict(sorted(excluded_sources.items())),
                  "source_counts_available": dict(sorted(sources.items()))}


def deduplicate(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen_hashes, seen_groups, result, removed = set(), set(), [], 0
    for row in rows:
        if row["prompt_hash"] in seen_hashes or row["source_group"] in seen_groups:
            removed += 1; continue
        seen_hashes.add(row["prompt_hash"]); seen_groups.add(row["source_group"]); result.append(row)
    return result, removed


def resolve_math_split_counts(train_count: int, validation_count: int, test_count: int,
                              limit: int | None) -> dict[str, int]:
    counts = {"train": train_count, "validation": validation_count, "test": test_count}
    if any(value < 1 for value in counts.values()):
        raise ValueError("--train-count, --validation-count, and --test-count must all be positive")
    total = sum(counts.values())
    if limit is not None and limit != total:
        raise ValueError(f"--limit {limit} conflicts with requested split-count sum {total}")
    return counts


def length_stats(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered: return {"mean":0.0, "p50":0.0, "p95":0.0, "p99":0.0, "max":0}
    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower, upper = math.floor(position), math.ceil(position)
        return ordered[lower] if lower == upper else ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return {"mean":statistics.fmean(ordered), "p50":percentile(.50), "p95":percentile(.95),
            "p99":percentile(.99), "max":max(ordered)}


def load_training_token_helper():
    path = Path(__file__).with_name("train_task_adapter.py")
    spec = importlib.util.spec_from_file_location("math_training_contract", path)
    if spec is None or spec.loader is None: raise RuntimeError(f"cannot load training contract: {path}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.tokenization_details


def load_local_tokenizer(tokenizer_path: Path | None, allow_skip: bool) -> Any | None:
    if tokenizer_path is None:
        if allow_skip: return None
        raise ValueError("--tokenizer-path is required for Math token filtering; use --allow-skip-token-validation only explicitly")
    if not tokenizer_path.exists(): raise FileNotFoundError(tokenizer_path)
    try:
        from transformers import AutoTokenizer
    except ImportError as exc: raise RuntimeError("transformers is required for local tokenizer filtering") from exc
    return AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)


def token_filter_math(rows: list[dict[str, Any]], tokenizer: Any | None,
                      requested_counts: dict[str, int], allow_skip: bool = False
                      ) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    if tokenizer is None:
        if not allow_skip: raise ValueError("tokenizer is required unless Math token validation is explicitly skipped")
        return rows, {"executed":False, "skip_reason":"explicit --allow-skip-token-validation",
                      "requested_split_counts":requested_counts}, []
    analyze = load_training_token_helper()
    kept, diagnostics, serialized, assistants = [], [], [], []
    truncated = fully_lost = partially_truncated = 0
    excluded_source, excluded_task = Counter(), Counter()
    for row in rows:
        details = analyze(tokenizer, row["messages"], 512)
        serialized.append(int(details["serialized_tokens"]))
        assistants.append(int(details["assistant_tokens_before"]))
        truncated += bool(details["truncated"])
        partially_truncated += bool(details["assistant_partially_truncated"])
        if details["assistant_fully_lost"]:
            fully_lost += 1; excluded_source[row.get("_source", "unknown")] += 1
            excluded_task["math"] += 1
            diagnostics.append({"prompt_hash":row["prompt_hash"], "source_group":row["source_group"],
                "source":row.get("_source", "unknown"), "task":"math",
                "serialized_token_count":details["serialized_tokens"],
                "assistant_token_count_before_truncation":details["assistant_tokens_before"],
                "assistant_token_count_after_truncation":details["assistant_tokens_after"],
                "exclusion_reason":"assistant_fully_lost_at_cutoff_512"})
        else: kept.append(row)
    requested = sum(requested_counts.values())
    if len(kept) < requested:
        raise ValueError(f"math: requested {requested} rows but only {len(kept)} valid unique candidates remain after token filtering")
    return kept, {"executed":True, "cutoff":512, "total_rows_checked":len(rows),
        "rows_truncated":truncated, "assistant_fully_lost_rows":fully_lost,
        "assistant_partially_truncated_rows":partially_truncated,
        "rows_kept_after_token_filter":len(kept), "excluded_by_source":dict(excluded_source),
        "excluded_by_task":{"math":excluded_task["math"]},
        "serialized_conversation_tokens":length_stats(serialized),
        "assistant_tokens":length_stats(assistants),
        "requested_split_counts":requested_counts}, diagnostics


def split_rows(rows: list[dict[str, Any]], task: str, limit: int | None = None,
               split_counts: dict[str, int] | None = None) -> dict[str, list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: hashlib.sha256(
        f"{SEED}:{task}:{row['source_group']}".encode()).digest())
    if split_counts is not None:
        total = sum(split_counts.values())
        if len(ordered) < total:
            raise ValueError(f"need {total} unique rows for requested splits, found {len(ordered)}")
        ordered = ordered[:total]
    elif limit is not None: ordered = ordered[:limit]
    random.Random(SEED).shuffle(ordered)
    if split_counts is None:
        total = len(ordered)
        test_count = total // 10
        validation_count = total // 10
        train_count = total - validation_count - test_count
        split_counts = {"train":train_count, "validation":validation_count, "test":test_count}
    train_count = split_counts["train"]
    validation_count = split_counts["validation"]
    return {"train": ordered[:train_count],
            "validation": ordered[train_count:train_count + validation_count],
            "test": ordered[train_count + validation_count:]}


def validate_splits(splits: dict[str, list[dict[str, Any]]], task: str) -> dict[str, bool]:
    hashes, groups = set(), set()
    for rows in splits.values():
        for row in rows:
            messages = row.get("messages")
            if [m.get("role") for m in messages or []] != ["system", "user", "assistant"]:
                raise ValueError("invalid message role order")
            if any(not str(m.get("content", "")).strip() for m in messages):
                raise ValueError("empty message content")
            if row.get("task") != task or prompt_hash(messages[1]["content"]) != row.get("prompt_hash"):
                raise ValueError("task or prompt hash mismatch")
            if task == "math" and EVALUATION_SOURCE.search(str(row.get("_source", ""))):
                raise ValueError("evaluation source survived Math candidate filtering")
            if row["prompt_hash"] in hashes or row["source_group"] in groups:
                raise ValueError("prompt/source_group overlap across splits")
            hashes.add(row["prompt_hash"]); groups.add(row["source_group"])
    return {"schema": True, "answers_nonempty": True, "prompt_hashes": True,
            "prompt_and_source_group_disjoint": True, "evaluation_sources_excluded": True,
            "utf8_reload": True}


def _write_task_directory(output: Path, task: str, rows: list[dict[str, Any]], stats: dict[str, Any],
                          limit: int | None, split_counts: dict[str, int] | None,
                          diagnostics: list[dict[str, Any]]) -> None:
    rows, duplicates = deduplicate(rows)
    splits = split_rows(rows, task, limit, split_counts)
    validation = validate_splits(splits, task)
    output.mkdir(parents=True, exist_ok=True)
    source_counts = Counter()
    for split, values in splits.items():
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in values:
                source_counts[row["source_dataset"]] += 1
                clean = {key:value for key,value in row.items() if not key.startswith("_")}
                handle.write(json.dumps(clean, ensure_ascii=False) + "\n")
        with (output / f"{split}.jsonl").open(encoding="utf-8") as handle:
            for line in handle: json.loads(line)
    actual_counts = {k: len(v) for k, v in splits.items()}
    if isinstance(stats.get("token_filter"), dict):
        stats["token_filter"]["split_counts"] = actual_counts
    result = {"task": task, "seed": SEED,
              "requested_split_counts": split_counts or actual_counts,
              "split_counts": actual_counts,
              "source_counts": dict(source_counts), "duplicates_removed": duplicates,
              "validation": validation, **stats}
    (output / "stats.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({
        "task": task, "seed": SEED, "schema_version": 1,
        "source_inputs": ["adapter-router/raw/securecode/data" if task == "securecode"
                          else "adapter-router/raw/mathinstruct/MathInstruct.json"],
        "router_csv_used": False,
    }, indent=2) + "\n", encoding="utf-8")
    with (output / "token_filter_exclusions.jsonl").open("w", encoding="utf-8") as handle:
        for row in diagnostics: handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_task(output: Path, task: str, rows: list[dict[str, Any]], stats: dict[str, Any],
               limit: int | None, split_counts: dict[str, int] | None = None,
               diagnostics: list[dict[str, Any]] | None = None) -> None:
    if output.exists(): raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        _write_task_directory(staging, task, rows, stats, limit, split_counts, diagnostics or [])
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--securecode-root", type=Path, default=repo / "adapter-router/raw/securecode/data")
    parser.add_argument("--math-source", type=Path, default=repo / "adapter-router/raw/mathinstruct/MathInstruct.json")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).parent / "processed/six_task_v1_r2")
    parser.add_argument("--limit", type=int,
                        help="deprecated total cap; for math it must equal the three requested counts")
    parser.add_argument("--train-count", type=int, default=8000)
    parser.add_argument("--validation-count", type=int, default=500)
    parser.add_argument("--test-count", type=int, default=500)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--allow-skip-token-validation", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 30: parser.error("--limit must be at least 30")
    if args.task == "securecode":
        rows, stats = prepare_securecode(args.securecode_root)
        write_task(args.output_root / args.task, args.task, rows, stats, args.limit)
    else:
        counts = resolve_math_split_counts(
            args.train_count, args.validation_count, args.test_count, args.limit)
        rows, stats = prepare_math(args.math_source)
        rows, duplicates = deduplicate(rows); stats["duplicates_removed"] = duplicates
        tokenizer = load_local_tokenizer(args.tokenizer_path, args.allow_skip_token_validation)
        rows, token_stats, diagnostics = token_filter_math(
            rows, tokenizer, counts, args.allow_skip_token_validation)
        stats["token_filter"] = token_stats
        write_task(args.output_root / args.task, args.task, rows, stats, args.limit, counts, diagnostics)


if __name__ == "__main__": main()
