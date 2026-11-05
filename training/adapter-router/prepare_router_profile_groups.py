#!/usr/bin/env python3
"""Deterministically freeze Router profile groups shard003 through shard013."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

TASKS = ("securecode", "math", "ifeval", "bbh", "gpqa", "mmlupro")
ADAPTERS = {task: index for index, task in enumerate(TASKS)}
GROUPS = tuple(f"shard{index:03d}" for index in range(3, 14))
INITIAL_EXCLUSIONS = tuple(f"frozen-validation-shard{index:03d}" for index in range(3))
HASH_RE = re.compile(r"^[0-9a-fA-F]{64}$")
SEED = 42
STANDARD_SAMPLES_PER_TASK = 32
FINAL_SAMPLES_PER_TASK = 15
FROZEN_EXISTING_ROWS_PER_TASK = 96
TOTAL_PER_TASK = 431
TOTAL_BALANCED_PROMPTS = 2586
NEW_GENERATED_ROWS = 2010
CAPACITY_REASON = (
    "Balanced final group is limited by 15 remaining eligible SecureCode validation rows "
    "after frozen exclusions and shard003-shard012 allocation."
)
SOURCE_PATHS = {
    "securecode": Path("adapter-training/processed/six_task_v2_r1/securecode/validation.jsonl"),
    "math": Path("adapter-training/processed/six_task_v2_r3/math/validation.jsonl"),
    **{task: Path(f"adapter-training/processed/six_task_v1_r3/{task}/validation.jsonl")
       for task in ("ifeval", "bbh", "gpqa", "mmlupro")},
}


def role_for(group: str) -> str:
    number = int(group.removeprefix("shard"))
    if 3 <= number <= 9: return "train_profile"
    if 10 <= number <= 11: return "validation_profile"
    if 12 <= number <= 13: return "final_test"
    raise ValueError(f"unsupported group: {group}")


def samples_for(group: str) -> int:
    return FINAL_SAMPLES_PER_TASK if group == "shard013" else STANDARD_SAMPLES_PER_TASK


def fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256(); rows = 0
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            if line.strip(): rows += 1
    return {"path": str(path.resolve()), "size": path.stat().st_size,
            "rows": rows, "sha256": digest.hexdigest()}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows: handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())


def validate_selection_metadata(row: dict[str, Any], task: str, where: str) -> None:
    if row.get("task") != task: raise ValueError(f"row task mismatch: {where}")
    if not isinstance(row.get("prompt_hash"), str) or not HASH_RE.fullmatch(row["prompt_hash"]):
        raise ValueError(f"invalid prompt_hash: {where}")
    if not isinstance(row.get("source_group"), str) or not row["source_group"].strip():
        raise ValueError(f"empty source_group: {where}")


def selection_key(seed: int, group: str, task: str, prompt_hash: str, source_group: str) -> str:
    """Selection intentionally depends only on allowed, answer-independent metadata."""
    return hashlib.sha256(f"{seed}|{group}|{task}|{prompt_hash}|{source_group}".encode()).hexdigest()


def collect_exclusions(roots: list[Path]) -> tuple[set[str], set[str]]:
    hashes, groups = set(), set()
    for root in roots:
        for task in TASKS:
            path = root / task / "test.jsonl"
            for index, row in enumerate(read_jsonl(path), 1):
                prompt_hash, source_group = row.get("prompt_hash"), row.get("source_group")
                if not isinstance(prompt_hash, str) or not HASH_RE.fullmatch(prompt_hash):
                    raise ValueError(f"invalid exclusion prompt_hash: {path}:{index}")
                if not isinstance(source_group, str) or not source_group.strip():
                    raise ValueError(f"invalid exclusion source_group: {path}:{index}")
                hashes.add(prompt_hash); groups.add(source_group)
    return hashes, groups


def generate(sources: dict[str, Path], exclusion_roots: list[Path], output_root: Path,
             seed: int = SEED, write_jsonl_fn: Callable = write_jsonl) -> list[Path]:
    if set(sources) != set(TASKS): raise ValueError(f"source tasks must be exactly {list(TASKS)}")
    targets = [output_root / f"frozen-validation-{group}" for group in GROUPS]
    existing = [str(path) for path in targets if path.exists()]
    if existing: raise FileExistsError("refusing to overwrite existing output directories: " + ", ".join(existing))
    source_paths = [sources[task] for task in TASKS]
    exclusion_paths = [root / task / "test.jsonl" for root in exclusion_roots for task in TASKS]
    missing = [str(path) for path in source_paths + exclusion_paths if not path.is_file()]
    if missing: raise FileNotFoundError("missing dataset inputs: " + ", ".join(missing))
    source_before = [fingerprint(path) for path in source_paths]
    exclusion_before = [fingerprint(path) for path in exclusion_paths]
    rows_by_task = {task: read_jsonl(sources[task]) for task in TASKS}
    for task in TASKS:
        for index, row in enumerate(rows_by_task[task], 1):
            validate_selection_metadata(row, task, f"{sources[task]}:{index}")
    used_hashes, used_groups = collect_exclusions(exclusion_roots)
    initial_hashes, initial_groups = set(used_hashes), set(used_groups)
    source_candidate_counts = {
        task: sum(row["prompt_hash"] not in initial_hashes and row["source_group"] not in initial_groups
                  for row in rows_by_task[task])
        for task in TASKS
    }
    output_root.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".six-label-profile-groups.staging-", dir=output_root))
    manifests: dict[str, dict[str, Any]] = {}; stats_by_group: dict[str, dict[str, Any]] = {}
    generated_records: list[dict[str, Any]] = []; remaining_before_final: dict[str, int] = {}
    published: list[Path] = []
    try:
        for group, target in zip(GROUPS, targets):
            count = samples_for(group); selected_by_task = {}; task_stats = {}
            prior_hashes, prior_groups = set(used_hashes), set(used_groups)
            group_hashes, group_groups = set(), set()
            if group == "shard013":
                remaining_before_final = {
                    task: sum(row["prompt_hash"] not in used_hashes and row["source_group"] not in used_groups
                              for row in rows_by_task[task])
                    for task in TASKS
                }
            for task in TASKS:
                ordered = sorted(rows_by_task[task], key=lambda row: selection_key(
                    seed, group, task, row["prompt_hash"], row["source_group"]))
                eligible = [row for row in ordered if row["prompt_hash"] not in used_hashes
                            and row["source_group"] not in used_groups]
                selected = []
                for row in eligible:
                    if row["prompt_hash"] in group_hashes or row["source_group"] in group_groups: continue
                    selected.append(row); group_hashes.add(row["prompt_hash"]); group_groups.add(row["source_group"])
                    if len(selected) == count: break
                if len(selected) != count:
                    raise ValueError(f"insufficient candidates for {task} in {group}: {len(selected)} < {count}")
                selected_by_task[task] = selected
                task_stats[task] = {"source_rows": len(ordered), "eligible_before_cross_task_dedup": len(eligible),
                                    "selected": len(selected)}
            used_hashes.update(group_hashes); used_groups.update(group_groups)
            stage = staging_root / target.name; output_records = []
            for task in TASKS:
                path = stage / task / "test.jsonl"; write_jsonl_fn(path, selected_by_task[task])
                reloaded = read_jsonl(path)
                if reloaded != selected_by_task[task]: raise RuntimeError(f"output reload mismatch: {group}/{task}")
                record = fingerprint(path); record["path"] = str((target / task / "test.jsonl").resolve())
                output_records.append(record)
            total = count * len(TASKS)
            checks = {"prompt_hash_overlap_with_shard000_001_002": len(group_hashes & initial_hashes),
                      "source_group_overlap_with_shard000_001_002": len(group_groups & initial_groups),
                      "prompt_hash_overlap_with_earlier_new_groups": len(
                          group_hashes & (prior_hashes - initial_hashes)),
                      "source_group_overlap_with_earlier_new_groups": len(
                          group_groups & (prior_groups - initial_groups)),
                      "prompt_hash_duplicates_within_group": total - len(group_hashes),
                      "source_group_duplicates_within_group": total - len(group_groups),
                      "per_task_counts_exact": all(len(rows) == count for rows in selected_by_task.values())}
            if any(checks[key] for key in checks if key != "per_task_counts_exact") or not checks["per_task_counts_exact"]:
                raise RuntimeError(f"integrity validation failed: {group}: {checks}")
            manifest = {"group_name": group, "role": role_for(group), "seed": seed,
                "selection_policy": ("Sort ascending by SHA-256(\"42|<shard_name>|<task>|<prompt_hash>|"
                                     "<source_group>\"); use validation rows and permitted metadata only."),
                "source_split": "validation", "samples_per_task": count, "total_rows": total,
                "capacity_reason": CAPACITY_REASON,
                "source_candidate_counts_after_frozen_exclusions": source_candidate_counts,
                "remaining_eligible_counts_after_shard003_through_shard012": remaining_before_final,
                "frozen_shard000_through_shard002_rows_per_task": FROZEN_EXISTING_ROWS_PER_TASK,
                "total_per_task": TOTAL_PER_TASK, "total_balanced_prompts": TOTAL_BALANCED_PROMPTS,
                "newly_generated_shard003_through_shard013_rows": NEW_GENERATED_ROWS,
                "sources": source_before, "initial_exclusions": exclusion_before,
                "earlier_generated_group_outputs": list(generated_records), "outputs": output_records,
                "checks": checks, "created_before_profiling": True,
                "adapter_mapping": ADAPTERS,
                "profiling_contract": {"split": "test", "max_samples": 32, "batch_size": 1,
                                       "max_new_tokens": 1024, "do_sample": False,
                                       "scoring": "frozen cross_profile_task_adapters.py",
                                       "exact_labels": "all adapters with exact task-aware generation score 1.0",
                                       "weak_labels": "finite assistant-reference mean NLL, reported separately"},
                "prohibitions": [
                    "Shard002 must not be used for retraining or model selection.",
                    "Shard012 and Shard013 are final tests and must not be used for training, tuning, parser changes, threshold selection, or model selection."]}
            stats = {"group_name": group, "role": role_for(group), "per_task": task_stats,
                     "samples_per_task": count,
                     "per_task_rows": {task: len(selected_by_task[task]) for task in TASKS},
                     "total_rows": total, "capacity_reason": CAPACITY_REASON,
                     "source_candidate_counts_after_frozen_exclusions": source_candidate_counts,
                     "remaining_eligible_counts_after_shard003_through_shard012": remaining_before_final,
                     "frozen_shard000_through_shard002_rows_per_task": FROZEN_EXISTING_ROWS_PER_TASK,
                     "total_per_task": TOTAL_PER_TASK, "total_balanced_prompts": TOTAL_BALANCED_PROMPTS,
                     "newly_generated_shard003_through_shard013_rows": NEW_GENERATED_ROWS,
                     "checks": checks, "outputs": output_records}
            write_json(stage / "manifest.json", manifest); write_json(stage / "stats.json", stats)
            manifests[group] = manifest; stats_by_group[group] = stats; generated_records.extend(output_records)
        source_after = [fingerprint(path) for path in source_paths]
        exclusion_after = [fingerprint(path) for path in exclusion_paths]
        if source_before != source_after: raise RuntimeError("source fingerprints changed during generation")
        if exclusion_before != exclusion_after: raise RuntimeError("exclusion fingerprints changed during generation")
        for group, target in zip(GROUPS, targets):
            stage = staging_root / target.name
            manifest_path = stage / "manifest.json"; manifest = manifests[group]; stats = stats_by_group[group]
            manifest["remaining_eligible_counts_after_shard003_through_shard012"] = remaining_before_final
            stats["remaining_eligible_counts_after_shard003_through_shard012"] = remaining_before_final
            manifest["source_fingerprints_after"] = source_after
            manifest["exclusion_fingerprints_after"] = exclusion_after
            manifest["fingerprints_unchanged"] = True
            write_json(manifest_path, manifest); write_json(stage / "stats.json", stats)
        for target in targets:
            (staging_root / target.name).rename(target); published.append(target)
        staging_root.rmdir()
        return targets
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        for target in published: shutil.rmtree(target, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("adapter-training/processed/router-profile"))
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    roots = [args.output_root / name for name in INITIAL_EXCLUSIONS]
    generate(SOURCE_PATHS, roots, args.output_root, args.seed)
