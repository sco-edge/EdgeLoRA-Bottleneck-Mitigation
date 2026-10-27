#!/usr/bin/env python3
"""Build leakage-controlled SecureCode v2 data from local and SecuCoder inputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import statistics
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

REPOSITORY = "ivitopow/SecuCoder"
REVISION = "7d5149ab257ad938384cacef9fd1827174630d0d"
LICENSE = "CC-BY-NC-SA-4.0"
EXPECTED_SPLITS = {"train": 5708, "validation": 317, "test": 317}
SEED = 42
ROLES = ("system", "user", "assistant")
LOCAL_SYSTEM = "Provide secure, correct code and explain the relevant security reasoning."
OVERLAPPING_SOURCES = ("scthornton-securecode-web", "scthornton-securecode-mlai",
                       "scthornton-securecode-aiml")
TASK_BUCKETS = ("fix", "conversation", "classify", "prompt_to_code", "other")


def normalize_prompt(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split()).strip()


def prompt_hash(value: str) -> str:
    return hashlib.sha256(normalize_prompt(value).encode("utf-8")).hexdigest()


def percentile(values: list[int], fraction: float) -> float:
    if not values: return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper: return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def length_stats(values: list[int]) -> dict[str, float | int]:
    return {"mean": statistics.fmean(values) if values else 0.0,
            "p50": percentile(values, .50), "p95": percentile(values, .95),
            "p99": percentile(values, .99), "max": max(values, default=0)}


def canonical_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError("messages must contain exactly system/user/assistant")
    if tuple(message.get("role") for message in messages if isinstance(message, dict)) != ROLES:
        raise ValueError("messages role order must be system/user/assistant")
    result = [{"role": role, "content": str(message.get("content", "")).strip()}
              for role, message in zip(ROLES, messages)]
    if any(not message["content"] for message in result):
        raise ValueError("empty system/user/assistant content")
    return result


def row(task_dataset: str, group: str, messages: list[dict[str, str]],
        meta: dict[str, Any]) -> dict[str, Any]:
    messages = canonical_messages(messages)
    return {"messages": messages, "task": "securecode", "source_dataset": task_dataset,
            "source_group": group, "prompt_hash": prompt_hash(messages[1]["content"]),
            "source_task": task_bucket(meta.get("task")),
            "source": str(meta.get("source", "unknown")),
            "cwe": normalize_list(meta.get("cwe")),
            "language": normalize_prompt(meta.get("language", meta.get("lang", "unknown"))) or "unknown"}


def normalize_list(value: Any) -> list[str]:
    if value in (None, ""): return []
    values = value if isinstance(value, list) else [value]
    cleaned = [" ".join(unicodedata.normalize("NFKC", str(item)).split()).strip() for item in values]
    return [item for item in cleaned if item]


def task_bucket(value: Any) -> str:
    value = normalize_prompt(value).lower()
    return value if value in TASK_BUCKETS[:-1] else "other"


def local_rows(root: Path) -> tuple[list[dict[str, Any]], Counter]:
    result, excluded = [], Counter()
    for domain in ("web", "aiml"):
        path = root / domain / "train.jsonl"
        if not path.is_file(): raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    record = json.loads(line)
                    record_id = normalize_prompt(record.get("id", ""))
                    turns = record.get("conversations")
                    if not record_id or not isinstance(turns, list): raise ValueError("missing identity/conversation")
                    user = next(m for m in turns if m.get("role") in {"human", "user"})
                    assistant = next(m for m in turns if m.get("role") in {"assistant", "gpt"})
                    meta = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
                    meta = {**meta, "task": "conversation", "source": f"scthornton-securecode-{domain}"}
                    result.append(row("scthornton/securecode", f"scthornton/securecode:{domain}:{record_id}", [
                        {"role":"system", "content":LOCAL_SYSTEM},
                        {"role":"user", "content":user.get("content", "")},
                        {"role":"assistant", "content":assistant.get("content", "")},
                    ], meta))
                except (ValueError, StopIteration, TypeError, json.JSONDecodeError):
                    excluded["malformed_or_empty_role"] += 1
    return result, excluded


def iter_secucoder_splits(root: Path) -> Iterator[tuple[str, Iterable[dict[str, Any]]]]:
    """Read an RTX save_to_disk DatasetDict, with JSONL directories for fixtures."""
    fixture = {split: root / f"{split}.jsonl" for split in EXPECTED_SPLITS}
    if all(path.is_file() for path in fixture.values()):
        for split, path in fixture.items():
            def records(path: Path = path) -> Iterator[dict[str, Any]]:
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip(): yield json.loads(line)
            yield split, records()
        return
    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as exc:
        raise RuntimeError("SecuCoder save_to_disk input requires datasets on the RTX host") from exc
    loaded = load_from_disk(str(root))
    if not isinstance(loaded, DatasetDict) or set(loaded) != set(EXPECTED_SPLITS):
        raise ValueError("SecuCoder input must preserve train/validation/test DatasetDict splits")
    for split in EXPECTED_SPLITS: yield split, loaded[split]


def secucoder_rows(root: Path, enforce_manifest: bool = True) -> tuple[dict[str, list[dict[str, Any]]], Counter]:
    if enforce_manifest:
        manifest_path = root / "download_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("repository") != REPOSITORY or manifest.get("revision") != REVISION
                or manifest.get("license") != LICENSE
                or manifest.get("split_counts") != EXPECTED_SPLITS):
            raise ValueError("SecuCoder repository/revision manifest mismatch")
    splits, excluded = {split: [] for split in EXPECTED_SPLITS}, Counter()
    for split, records in iter_secucoder_splits(root):
        for index, record in enumerate(records):
            try:
                if not isinstance(record, dict) or not isinstance(record.get("meta"), dict):
                    raise ValueError("missing meta")
                messages, meta = canonical_messages(record.get("messages")), record["meta"]
                source = normalize_prompt(meta.get("source", ""))
                identity = normalize_prompt(meta.get("id", record.get("id", "")))
                if not source: raise ValueError("missing meta.source")
                # No stable row ID is guaranteed by the published schema.  The
                # source plus canonical message digest is therefore the finest
                # reproducible source identity available.
                digest = hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                group = f"{REPOSITORY}:{source}:{identity or digest}"
                splits[split].append(row(REPOSITORY, group, messages, meta))
            except (ValueError, TypeError): excluded[f"{split}_malformed_or_empty_role"] += 1
    if enforce_manifest:
        actual = {split:len(splits[split]) + excluded[f"{split}_malformed_or_empty_role"]
                  for split in EXPECTED_SPLITS}
        if actual != EXPECTED_SPLITS: raise ValueError(f"SecuCoder split count mismatch: {actual}")
    return splits, excluded


def source_overlaps_local(value: str) -> bool:
    normalized = value.replace("\\", "/").lower()
    return any(source in normalized for source in OVERLAPPING_SOURCES)


def unique(rows: Iterable[dict[str, Any]], blocked_hashes: set[str] | None = None,
           blocked_groups: set[str] | None = None) -> tuple[list[dict[str, Any]], Counter]:
    hashes, groups = set(blocked_hashes or ()), set(blocked_groups or ())
    result, removed = [], Counter()
    for value in rows:
        if value["prompt_hash"] in hashes:
            removed["prompt_duplicate"] += 1; continue
        if value["source_group"] in groups:
            removed["source_group_duplicate"] += 1; continue
        hashes.add(value["prompt_hash"]); groups.add(value["source_group"]); result.append(value)
    return result, removed


def deterministic_local_split(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda item: hashlib.sha256(
        f"{SEED}:{item['source_group']}".encode()).digest())
    random.Random(SEED).shuffle(ordered)
    holdout = len(ordered) // 10
    return {"train": ordered[2 * holdout:], "validation": ordered[:holdout],
            "test": ordered[holdout:2 * holdout]}


def combine(local: list[dict[str, Any]], secu: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    counts = Counter()
    # Upstream test wins over validation for any malformed upstream overlap.
    test, removed = unique(secu["test"]); counts.update(removed)
    test_hashes, test_groups = {r["prompt_hash"] for r in test}, {r["source_group"] for r in test}
    validation, removed = unique(secu["validation"], test_hashes, test_groups); counts.update(removed)
    holdout_hashes = test_hashes | {r["prompt_hash"] for r in validation}
    holdout_groups = test_groups | {r["source_group"] for r in validation}

    local, removed = unique(local, holdout_hashes, holdout_groups); counts["split_leakage_removed"] += sum(removed.values())
    local_split = deterministic_local_split(local)
    test, removed = unique([*test, *local_split["test"]]); counts.update(removed)
    validation, removed = unique([*validation, *local_split["validation"]],
                                 {r["prompt_hash"] for r in test}, {r["source_group"] for r in test})
    counts.update(removed)
    blocked_hashes = {r["prompt_hash"] for r in [*test, *validation]}
    blocked_groups = {r["source_group"] for r in [*test, *validation]}

    secu_train = []
    for value in secu["train"]:
        if source_overlaps_local(value["source"]): counts["source_duplicate"] += 1
        else: secu_train.append(value)
    train, removed = unique([*local_split["train"], *secu_train], blocked_hashes, blocked_groups)
    counts["prompt_duplicate"] += removed["prompt_duplicate"]
    counts["source_group_duplicate"] += removed["source_group_duplicate"]
    counts["split_leakage_removed"] += sum(removed.values())
    return {"train": train, "validation": validation, "test": test}, dict(counts)


def validate(splits: dict[str, list[dict[str, Any]]]) -> dict[str, bool]:
    hashes, groups = set(), set()
    for split, values in splits.items():
        for value in values:
            messages = canonical_messages(value["messages"])
            if prompt_hash(messages[1]["content"]) != value["prompt_hash"]: raise ValueError("prompt hash mismatch")
            if value["prompt_hash"] in hashes or value["source_group"] in groups: raise ValueError("split leakage")
            if split == "train" and value.get("upstream_split") in {"validation", "test"}:
                raise ValueError("SecuCoder holdout leaked into train")
            hashes.add(value["prompt_hash"]); groups.add(value["source_group"])
    return {"schema":True, "nonempty_roles":True, "answers_nonempty":True,
            "prompt_hashes":True, "split_prompt_disjoint":True, "split_source_group_disjoint":True,
            "secucoder_holdout_not_in_train":True, "utf8_reload":True, "router_csv_excluded":True}


def character_statistics(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    inputs = [len(row["messages"][1]["content"]) for values in splits.values() for row in values]
    answers = [len(row["messages"][2]["content"]) for values in splits.values() for row in values]
    return {"input_characters":length_stats(inputs), "assistant_characters":length_stats(answers)}


def load_training_token_helper():
    path = Path(__file__).with_name("train_task_adapter.py")
    spec = importlib.util.spec_from_file_location("securecode_training_contract", path)
    if spec is None or spec.loader is None: raise RuntimeError(f"cannot load training contract: {path}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.tokenization_details


def token_filter(splits: dict[str, list[dict[str, Any]]], tokenizer_path: Path | None,
                 allow_skip: bool = False, tokenizer: Any | None = None
                 ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], list[dict[str, Any]]]:
    if tokenizer_path is None and tokenizer is None:
        if not allow_skip:
            raise ValueError("--tokenizer-path is required for token filtering; use --allow-skip-token-validation only for fixtures")
        return splits, {"executed":False, "skip_reason":"explicit --allow-skip-token-validation"}, []
    if tokenizer is None and (tokenizer_path is None or not tokenizer_path.exists()):
        raise FileNotFoundError(tokenizer_path)
    if tokenizer is None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc: raise RuntimeError("transformers is required for local tokenizer filtering") from exc
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    analyze = load_training_token_helper()
    inputs, assistants, truncations, lost, partial = [], [], 0, 0, 0
    excluded_by_split, excluded_by_source, excluded_by_task = Counter(), Counter(), Counter()
    filtered, diagnostics = {split:[] for split in splits}, []
    for split, values in splits.items():
        for value in values:
            details = analyze(tokenizer, value["messages"], 512)
            inputs.append(int(details["serialized_tokens"])); assistants.append(int(details["assistant_tokens_before"]))
            truncations += bool(details["truncated"]); partial += bool(details["assistant_partially_truncated"])
            if details["assistant_fully_lost"]:
                lost += 1; excluded_by_split[split] += 1
                excluded_by_source[value["source"]] += 1; excluded_by_task[value["source_task"]] += 1
                diagnostics.append({"prompt_hash":value["prompt_hash"], "source_group":value["source_group"],
                    "source":value["source"], "task":value["source_task"],
                    "original_token_length":details["serialized_tokens"],
                    "assistant_token_count_before_truncation":details["assistant_tokens_before"],
                    "assistant_token_count_after_truncation":details["assistant_tokens_after"],
                    "exclusion_reason":"assistant_fully_lost_at_cutoff_512"})
            else: filtered[split].append(value)
    if any(not values for values in filtered.values()):
        raise ValueError("token filtering produced an empty train, validation, or test split")
    validate(filtered)
    return filtered, {"executed":True, "tokenizer_path":str(tokenizer_path or "fixture"), "cutoff":512,
            "total_rows_checked":len(inputs), "rows_truncated":truncations,
            "assistant_fully_lost_rows":lost, "assistant_partially_truncated_rows":partial,
            "rows_kept_after_token_filter":sum(map(len, filtered.values())),
            "excluded_by_split":{name:excluded_by_split[name] for name in splits},
            "excluded_by_source":dict(excluded_by_source),
            "excluded_by_task":{name:excluded_by_task[name] for name in TASK_BUCKETS},
            "truncation_rows":truncations, "assistant_full_loss_rows":lost,
            "serialized_conversation_tokens":length_stats(inputs),
            "assistant_tokens":length_stats(assistants)}, diagnostics


def write(output: Path, splits: dict[str, list[dict[str, Any]]], stats: dict[str, Any],
          diagnostics: list[dict[str, Any]] | None = None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    validation = validate(splits)
    source_counts, task_counts, cwes, languages = Counter(), Counter(), Counter(), Counter()
    for split, values in splits.items():
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for value in values:
                source_counts[value["source"]] += 1; task_counts[value["source_task"]] += 1
                cwes.update(value["cwe"] or ["unknown"]); languages[value["language"]] += 1
                clean = {key:item for key,item in value.items() if key not in {"source_task", "source", "cwe", "language", "upstream_split"}}
                handle.write(json.dumps(clean, ensure_ascii=False) + "\n")
        with (output / f"{split}.jsonl").open(encoding="utf-8") as handle:
            for line in handle: json.loads(line)
    payload = {**stats, "split_counts":{k:len(v) for k,v in splits.items()},
               "source_counts":dict(source_counts), "task_counts":{k:task_counts[k] for k in TASK_BUCKETS},
               "cwe_distribution":dict(cwes), "language_distribution":dict(languages), "validation":validation}
    (output / "stats.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({
        "schema_version":2, "seed":SEED,
        "sources":[{"repository":"scthornton/securecode", "input":"adapter-router/raw/securecode/data"},
                   {"repository":REPOSITORY, "revision":REVISION, "license":LICENSE,
                    "input":"adapter-training/raw/secucoder", "download_split_counts":EXPECTED_SPLITS}],
        "secucoder_training_split_only":True, "secucoder_holdouts_preserved":True,
        "router_csv_used":False,
    }, indent=2) + "\n", encoding="utf-8")
    with (output / "token_filter_exclusions.jsonl").open("w", encoding="utf-8") as handle:
        for value in diagnostics or (): handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def write_atomic(output: Path, splits: dict[str, list[dict[str, Any]]], stats: dict[str, Any],
                 diagnostics: list[dict[str, Any]]) -> None:
    if output.exists(): raise FileExistsError(f"refusing to overwrite existing processed data: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        write(staging, splits, stats, diagnostics)
        validate(splits)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def run(local_root: Path, secucoder_root: Path, output: Path, tokenizer_path: Path | None,
        enforce_manifest: bool = True, allow_skip_token_validation: bool = False) -> dict[str, Any]:
    local, local_bad = local_rows(local_root)
    secu, secu_bad = secucoder_rows(secucoder_root, enforce_manifest)
    for split, values in secu.items():
        for value in values: value["upstream_split"] = split
    splits, removals = combine(local, secu)
    splits, token_stats, diagnostics = token_filter(
        splits, tokenizer_path, allow_skip=allow_skip_token_validation)
    stats = {"repository":REPOSITORY, "revision":REVISION, "license":LICENSE,
             "existing_securecode_input_count":len(local) + sum(local_bad.values()),
             "secucoder_input_counts":{k:len(v) + secu_bad[f"{k}_malformed_or_empty_role"]
                                        for k,v in secu.items()},
             "secucoder_valid_counts":{k:len(v) for k,v in secu.items()},
             "source_duplicate_removed":removals.get("source_duplicate", 0),
             "prompt_duplicate_removed":removals.get("prompt_duplicate", 0),
             "source_group_duplicate_removed":removals.get("source_group_duplicate", 0),
             "duplicate_removed":removals.get("source_duplicate", 0) + removals.get("prompt_duplicate", 0)
                                  + removals.get("source_group_duplicate", 0),
             "split_leakage_removed":removals.get("split_leakage_removed", 0),
             "malformed_empty_role_removed":sum(local_bad.values()) + sum(secu_bad.values()),
             "malformed_reasons":dict(local_bad + secu_bad),
             "final_unique_rows":sum(map(len, splits.values())),
             "character_lengths":character_statistics(splits), "tokenizer_validation":token_stats}
    write_atomic(output, splits, stats, diagnostics)
    return stats


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-root", type=Path, default=repo / "adapter-router/raw/securecode/data")
    parser.add_argument("--secucoder-root", type=Path, default=Path(__file__).parent / "raw/secucoder")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "processed/six_task_v2/securecode")
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--allow-skip-token-validation", action="store_true",
                        help="explicitly skip token filtering (fixture/development only)")
    args = parser.parse_args()
    run(args.local_root, args.secucoder_root, args.output, args.tokenizer_path,
        allow_skip_token_validation=args.allow_skip_token_validation)


if __name__ == "__main__": main()
