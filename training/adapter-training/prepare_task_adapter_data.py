#!/usr/bin/env python3
"""Prepare four leakage-controlled task-adapter datasets.

The implementation deliberately uses only the Python standard library.  It can
read JSON/JSONL exports and streams the very large CoT Collection JSON array.
It never modifies files below ``raw``.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib.util
import json
import math
import os
import random
import re
import shutil
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

SEED = 42
SPLIT_SIZES = {"train": 8000, "validation": 500, "test": 500}
SYSTEM = {
    "ifeval": "Follow the user's instructions precisely and satisfy every stated constraint.",
    "bbh": "Solve the problem carefully. Show useful reasoning and finish with the requested result.",
    "gpqa": "Answer the multiple-choice science question. Give the answer label and answer text.",
    "mmlupro": "Answer the multiple-choice question. Give the answer label and answer text.",
}
REVISIONS = {
    "HuggingFaceH4/no_robots": "e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b",
    "kaist-ai/CoT-Collection": "c9d352cdc119df4a4f7526d100e4acb4a72a7a5c",
    "allenai/ai2_arc": "210d026faf9955653af8916fad021475a3f00453",
    "allenai/sciq": "2c94ad3e1aafab77146f384e23536f97a4849815",
    "allenai/openbookqa": "388097ea7776314e93a529163e0fea805b8a6454",
    "tau/commonsense_qa": "94630fe30dad47192a8546eb75f094926d47e155",
    "Rowan/hellaswag": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
    "allenai/winogrande": "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
}
EXCLUDED_BBH = re.compile(r"bbh|bigbench|big_bench|mmlu|gpqa|ifeval|(?:^|[^a-z])math(?:[^a-z]|$)", re.I)


def normalize_prompt(value: str) -> str:
    return " ".join(str(value).split()).strip()


def prompt_hash(user_prompt: str) -> str:
    return hashlib.sha256(normalize_prompt(user_prompt).encode("utf-8")).hexdigest()


def source_group(dataset: str, row: dict[str, Any], index: int) -> str:
    raw_id = row.get("id", row.get("idx", row.get("question_id")))
    return f"{dataset}:{raw_id if raw_id not in (None, '') else index}"


def example(task: str, dataset: str, group: str, user: str, assistant: str) -> dict[str, Any]:
    user, assistant = str(user).strip(), str(assistant).strip()
    if not user or not assistant:
        raise ValueError("empty user or assistant content")
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
    """Stream a top-level JSON array without loading the full file."""
    decoder, buffer, pos, started, eof = json.JSONDecoder(), "", 0, False, False
    with path.open("r", encoding="utf-8") as handle:
        while True:
            if pos:
                buffer, pos = buffer[pos:], 0
            chunk = handle.read(chunk_size)
            if chunk:
                buffer += chunk
            else:
                eof = True
            while True:
                while pos < len(buffer) and buffer[pos].isspace():
                    pos += 1
                if not started:
                    if pos >= len(buffer):
                        break
                    if buffer[pos] != "[":
                        raise ValueError(f"{path}: expected a top-level JSON array")
                    started, pos = True, pos + 1
                    continue
                while pos < len(buffer) and (buffer[pos].isspace() or buffer[pos] == ","):
                    pos += 1
                if pos < len(buffer) and buffer[pos] == "]":
                    return
                try:
                    value, end = decoder.raw_decode(buffer, pos)
                except json.JSONDecodeError:
                    if eof:
                        raise ValueError(f"{path}: truncated or invalid JSON array")
                    break
                yield value
                pos = end
            if eof:
                raise ValueError(f"{path}: JSON array has no closing bracket")


def iter_json_object_records(path: Path, chunk_size: int = 1 << 20) -> Iterator[tuple[str, Any]]:
    """Stream ``(key, value)`` pairs from one top-level JSON object.

    Only one value is materialized at a time.  Duplicate keys are intentionally
    not detected: retaining every key from the 2.36 GB CoT object would defeat
    the bounded-memory design.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    decoder = json.JSONDecoder()
    buffer, pos, eof = "", 0, False

    with path.open("r", encoding="utf-8") as handle:
        def refill() -> bool:
            nonlocal buffer, pos, eof
            if pos:
                buffer, pos = buffer[pos:], 0
            chunk = handle.read(chunk_size)
            if chunk:
                buffer += chunk
                return True
            eof = True
            return False

        def skip_space() -> None:
            nonlocal pos
            while pos < len(buffer) and buffer[pos].isspace():
                pos += 1

        def need_char(context: str) -> str:
            while True:
                skip_space()
                if pos < len(buffer):
                    return buffer[pos]
                if not refill():
                    raise ValueError(f"{path}: unexpected end while {context}")

        def decode_one(context: str) -> Any:
            nonlocal pos
            while True:
                skip_space()
                try:
                    value, end = decoder.raw_decode(buffer, pos)
                except json.JSONDecodeError as exc:
                    if not eof and refill():
                        continue
                    raise ValueError(f"{path}: invalid JSON {context}: {exc.msg}") from exc
                pos = end
                return value

        if need_char("expecting top-level object") != "{":
            raise ValueError(f"{path}: expected a top-level JSON object")
        pos += 1
        expect_member = True
        while True:
            char = need_char("expecting an object key or closing brace")
            if char == "}":
                pos += 1
                break
            if not expect_member:
                raise ValueError(f"{path}: expected comma or closing brace")
            key = decode_one("object key")
            if not isinstance(key, str):
                raise ValueError(f"{path}: top-level object key must be a string")
            if need_char(f"expecting colon after key {key!r}") != ":":
                raise ValueError(f"{path}: expected colon after key {key!r}")
            pos += 1
            value = decode_one(f"value for key {key!r}")
            yield key, value
            char = need_char(f"expecting comma or closing brace after key {key!r}")
            if char == ",":
                pos += 1
                # A following closing brace is a forbidden trailing comma.
                if need_char("expecting an object key after comma") == "}":
                    raise ValueError(f"{path}: trailing comma before closing brace")
                expect_member = True
            elif char == "}":
                pos += 1
                break
            else:
                raise ValueError(f"{path}: expected comma or closing brace after key {key!r}")

        while True:
            skip_space()
            if pos < len(buffer):
                raise ValueError(f"{path}: trailing garbage after top-level object")
            if not refill():
                return


HF_METADATA_FILES = {"dataset_info.json", "state.json", "dataset_dict.json", "revisions.json"}


def is_hf_saved_dataset(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "state.json").is_file()
        and (path / "dataset_info.json").is_file()
        and any(path.glob("data-*.arrow"))
    )


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    if is_hf_saved_dataset(path):
        try:
            from datasets import DatasetDict, load_from_disk  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                f"{path} is a Hugging Face Dataset.save_to_disk directory; "
                "the execution environment must provide datasets==3.0.2"
            ) from exc
        loaded = load_from_disk(str(path))
        if isinstance(loaded, DatasetDict):
            if "train" not in loaded:
                raise ValueError(f"{path}: saved DatasetDict has no train split")
            loaded = loaded["train"]
        for row in loaded:
            yield dict(row)
        return
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as parquet  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                f"{path} is Parquet; use the training environment's existing pyarrow/datasets "
                "installation or export the raw split as JSONL (this script never installs packages)"
            ) from exc
        parquet_file = parquet.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=4096):
            yield from batch.to_pylist()
        return
    if path.suffix == ".arrow":
        try:
            import pyarrow.ipc as ipc  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                f"{path} is an Arrow data file; use datasets==3.0.2 or the "
                "training environment's existing pyarrow installation"
            ) from exc
        with path.open("rb") as handle:
            reader = ipc.open_stream(handle)
            for batch in reader:
                yield from batch.to_pylist()
        return
    # Most source exports are arrays.  Also accept the common {"data": [...]}
    # shape without weakening the streaming guarantee for the 2.36 GB CoT file.
    with path.open("r", encoding="utf-8") as handle:
        first = next((ch for ch in iter(lambda: handle.read(1), "") if not ch.isspace()), "")
    if first == "[":
        yield from iter_json_array(path)
    elif first == "{":
        value = json.loads(path.read_text(encoding="utf-8"))
        values = value.get("data", value.get("rows", value))
        if not isinstance(values, list):
            raise ValueError(f"{path}: JSON object must contain a data/rows array")
        yield from values
    else:
        raise ValueError(f"{path}: unsupported JSON shape")


def find_data_file(root: Path, hints: Sequence[str] = ()) -> Path:
    if root.is_file():
        if root.name in HF_METADATA_FILES:
            raise ValueError(f"metadata JSON is not record input: {root}")
        return root

    saved_dirs = sorted(
        (path for path in (root, *root.rglob("*")) if is_hf_saved_dataset(path)),
        key=lambda path: str(path),
    )
    arrow = sorted(root.rglob("data-*.arrow"))
    parquet = sorted(root.rglob("*.parquet"))
    jsonl = sorted(root.rglob("*.jsonl"))
    supported_json = sorted(
        path for path in root.rglob("*.json") if path.name not in HF_METADATA_FILES
    )
    for candidates in (saved_dirs, arrow, parquet, jsonl, supported_json):
        if not candidates:
            continue
        for hint in hints:
            match = next((path for path in candidates if hint.lower() in str(path).lower()), None)
            if match is not None:
                return match
        return candidates[0]
    raise FileNotFoundError(f"no supported record data under {root}")


def source_root(raw: Path, *names: str) -> Path:
    for name in names:
        candidate = raw / name
        if candidate.is_dir(): return candidate
    raise FileNotFoundError(f"source directory not found; tried: {', '.join(str(raw / n) for n in names)}")


def conversation_pair(row: dict[str, Any]) -> tuple[str, str]:
    messages = row.get("messages") or row.get("conversations")
    if isinstance(messages, list):
        user = next((m.get("content", m.get("value")) for m in messages if m.get("role", m.get("from")) in {"user", "human"}), None)
        assistant = next((m.get("content", m.get("value")) for m in messages if m.get("role", m.get("from")) in {"assistant", "gpt"}), None)
        if user and assistant:
            return str(user), str(assistant)
    user = row.get("prompt", row.get("instruction", row.get("input")))
    assistant = row.get("response", row.get("output", row.get("answer")))
    if user and assistant:
        return str(user), str(assistant)
    raise ValueError("row has no user/assistant pair")


def apply_constraint(user: str, target: str, kind: str, keyword: str = "verified") -> tuple[str, str]:
    if kind == "general":
        return user, target
    if kind == "exact_prefix":
        return user + '\nBegin the response with exactly "Answer: ".', "Answer: " + target
    if kind == "exact_suffix":
        return user + '\nEnd the response with exactly " [END]".', target + " [END]"
    if kind == "xml":
        return user + "\nWrap the entire response in <response> and </response> tags.", f"<response>{html.escape(target)}</response>"
    if kind == "json":
        return user + '\nReturn a JSON object with exactly one field named "answer".', json.dumps({"answer": target}, ensure_ascii=False)
    if kind == "keyword":
        return user + f'\nInclude the exact keyword "{keyword}".', target + f"\n{keyword}"
    raise ValueError(f"unknown constraint: {kind}")


def constraint_satisfied(target: str, kind: str, keyword: str = "verified") -> bool:
    if kind == "general": return bool(target.strip())
    if kind == "exact_prefix": return target.startswith("Answer: ")
    if kind == "exact_suffix": return target.endswith(" [END]")
    if kind == "xml": return target.startswith("<response>") and target.endswith("</response>")
    if kind == "keyword": return keyword in target
    if kind == "json":
        try: value = json.loads(target)
        except json.JSONDecodeError: return False
        return isinstance(value, dict) and list(value) == ["answer"] and isinstance(value["answer"], str)
    return False


def prepare_ifeval(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = find_data_file(root, ("train",))
    kinds = ("general", "exact_prefix", "exact_suffix", "xml", "json", "keyword")
    rows, counts, malformed = [], Counter(), 0
    for index, row in enumerate(iter_records(path)):
        try: user, target = conversation_pair(row)
        except ValueError:
            malformed += 1; continue
        kind = kinds[int(hashlib.sha256(f"{SEED}:{index}".encode()).hexdigest(), 16) % len(kinds)]
        user, target = apply_constraint(user, target, kind)
        item = example("ifeval", "HuggingFaceH4/no_robots", source_group("HuggingFaceH4/no_robots", row, index), user, target)
        item["_constraint"] = kind
        if not constraint_satisfied(target, kind): raise AssertionError("constraint transform failed")
        rows.append(item); counts[kind] += 1
    return rows, {"constraint_counts": dict(counts), "constraint_validation_failures": 0, "malformed": malformed}


def prepare_bbh(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, categories, configs, excluded, malformed = [], Counter(), Counter(), Counter(), Counter()
    for record_id, row in iter_json_object_records(path):
        if not isinstance(row, dict):
            malformed["value_not_object"] += 1
            continue
        missing = [field for field in ("source", "target", "rationale", "task") if not normalize_prompt(row.get(field, ""))]
        if missing:
            for field in missing:
                malformed[f"missing_or_empty_{field}"] += 1
            continue
        task = normalize_prompt(row["task"])
        source = normalize_prompt(row["source"])
        match = EXCLUDED_BBH.search(task + " " + source)
        if match:
            excluded[match.group(0).lower()] += 1; continue
        rationale, target = str(row["rationale"]).strip(), str(row["target"]).strip()
        config = normalize_prompt(row.get("config", "")) or "unspecified"
        rows.append(example("bbh", "kaist-ai/CoT-Collection", f"kaist-ai/CoT-Collection:{record_id}", source, f"{rationale}\n[RESULT]\n{target}"))
        rows[-1]["_category"] = task
        rows[-1]["_config"] = config
        categories[task] += 1
        configs[config] += 1
    return rows, {"category_available": dict(categories), "config_available": dict(configs),
                  "excluded_tasks": sum(excluded.values()), "excluded_reasons": dict(excluded),
                  "malformed_records": sum(malformed.values()), "malformed_reasons": dict(malformed)}


def choices_from(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    choices = row.get("choices")
    if isinstance(choices, dict):
        texts = choices.get("text", [])
        labels = choices.get("label") or [chr(65 + i) for i in range(len(texts))]
    elif isinstance(choices, list):
        texts, labels = choices, [chr(65 + i) for i in range(len(choices))]
    else:
        texts = row.get("endings", [])
        labels = [chr(65 + i) for i in range(len(texts))]
    return [str(x) for x in labels], [str(x) for x in texts]


def mcq_item(task: str, dataset: str, row: dict[str, Any], index: int, question: str,
             labels: list[str], texts: list[str], answer: Any, context: str = "") -> dict[str, Any]:
    if not labels or len(labels) != len(texts): raise ValueError("invalid choices")
    answer_s = str(answer)
    if answer_s in labels: pos = labels.index(answer_s)
    elif answer_s.isdigit() and 0 <= int(answer_s) < len(labels): pos = int(answer_s)
    else: raise ValueError(f"answer {answer!r} not in choices")
    shown = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
    prefix = f"Context: {context}\n" if context else ""
    user = f"{prefix}Question: {question}\nChoices:\n{shown}"
    item = example(task, dataset, source_group(dataset, row, index), user, f"{labels[pos]}. {texts[pos]}")
    item["_correct_answer"] = f"{labels[pos]}. {texts[pos]}"
    return item


def load_gpqa_source(root: Path, name: str) -> tuple[list[dict[str, Any]], int]:
    dataset = {"arc_challenge":"allenai/ai2_arc", "sciq":"allenai/sciq", "openbookqa":"allenai/openbookqa"}[name]
    path = find_data_file(root, (("challenge_train" if name == "arc_challenge" else "train"),))
    out, malformed = [], 0
    for i, row in enumerate(iter_records(path)):
        try:
            if name == "sciq":
                texts = [row["correct_answer"], row["distractor1"], row["distractor2"], row["distractor3"]]
                order = sorted(range(4), key=lambda x: hashlib.sha256(f"{SEED}:{i}:{x}".encode()).digest())
                texts = [texts[x] for x in order]; labels = list("ABCD"); answer = labels[order.index(0)]
                # support is intentionally assistant-side only: it can never leak into the user prompt.
                item = mcq_item("gpqa", dataset, row, i, row["question"], labels, texts, answer)
                if row.get("support"): item["messages"][2]["content"] = f"Reasoning: {row['support']}\nAnswer: {item['messages'][2]['content']}"
            else:
                labels, texts = choices_from(row)
                item = mcq_item("gpqa", dataset, row, i, row.get("question_stem", row.get("question", "")), labels, texts, row.get("answerKey", row.get("answer")))
            out.append(item)
        except (KeyError, ValueError, TypeError): malformed += 1
    return out, malformed


def load_mmlupro_source(root: Path, name: str) -> tuple[list[dict[str, Any]], int]:
    dataset = {"commonsense_qa":"tau/commonsense_qa", "hellaswag":"Rowan/hellaswag", "winogrande":"allenai/winogrande"}[name]
    path = find_data_file(root, ("train",))
    out, malformed = [], 0
    for i, row in enumerate(iter_records(path)):
        try:
            if name == "hellaswag":
                labels, texts = choices_from(row); question = f"{row.get('ctx', row.get('ctx_a', ''))} {row.get('ctx_b', '')}".strip(); answer = row["label"]
            elif name == "winogrande":
                labels, texts = ["A", "B"], [str(row["option1"]), str(row["option2"])]; question = row["sentence"]; answer = str(int(row["answer"]) - 1)
            else:
                labels, texts = choices_from(row); question = row["question"]; answer = row["answerKey"]
            out.append(mcq_item("mmlupro", dataset, row, i, question, labels, texts, answer))
        except (KeyError, ValueError, TypeError): malformed += 1
    return out, malformed


def deterministic_take(rows: Sequence[dict[str, Any]], count: int, salt: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda x: hashlib.sha256(f"{SEED}:{salt}:{x['source_group']}".encode()).digest())[:count]


def deduplicate(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen_hash, seen_group, out, removed = set(), set(), [], 0
    for row in rows:
        if row["prompt_hash"] in seen_hash or row["source_group"] in seen_group:
            removed += 1; continue
        seen_hash.add(row["prompt_hash"]); seen_group.add(row["source_group"]); out.append(row)
    return out, removed


def capped_categories(rows: Sequence[dict[str, Any]], total: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows: groups[row.get("_category", "unknown")].append(row)
    if not groups: return []
    cap = max(1, (total + len(groups) - 1) // len(groups) * 2)
    pool = [item for cat in sorted(groups) for item in deterministic_take(groups[cat], cap, cat)]
    return deterministic_take(pool, total, "bbh-balanced")


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
    spec = importlib.util.spec_from_file_location("four_task_training_contract", path)
    if spec is None or spec.loader is None: raise RuntimeError(f"cannot load training contract: {path}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.tokenization_details


def load_local_tokenizer(tokenizer_path: Path | None, allow_skip: bool) -> Any | None:
    if tokenizer_path is None:
        if allow_skip: return None
        raise ValueError("--tokenizer-path is required for token filtering; use --allow-skip-token-validation only explicitly")
    if not tokenizer_path.exists(): raise FileNotFoundError(tokenizer_path)
    try:
        from transformers import AutoTokenizer
    except ImportError as exc: raise RuntimeError("transformers is required for local tokenizer filtering") from exc
    return AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)


def token_filter_candidates(rows: list[dict[str, Any]], task: str, tokenizer: Any | None,
                            allow_skip: bool = False
                            ) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    if tokenizer is None:
        if not allow_skip: raise ValueError("tokenizer is required unless token validation is explicitly skipped")
        return rows, {"executed":False, "skip_reason":"explicit --allow-skip-token-validation"}, []
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
            fully_lost += 1; excluded_source[row["source_dataset"]] += 1; excluded_task[task] += 1
            diagnostics.append({"prompt_hash":row["prompt_hash"], "source_group":row["source_group"],
                "source":row["source_dataset"], "task":task,
                "serialized_token_count":details["serialized_tokens"],
                "assistant_token_count_before_truncation":details["assistant_tokens_before"],
                "assistant_token_count_after_truncation":details["assistant_tokens_after"],
                "exclusion_reason":"assistant_fully_lost_at_cutoff_512"})
        else: kept.append(row)
    return kept, {"executed":True, "cutoff":512, "total_rows_checked":len(rows),
        "rows_truncated":truncated, "assistant_fully_lost_rows":fully_lost,
        "assistant_partially_truncated_rows":partially_truncated,
        "rows_kept_after_token_filter":len(kept),
        # Filtering precedes split assignment so exclusions belong to the candidate pool.
        "excluded_by_split":{"candidate_pool":fully_lost, "train":0, "validation":0, "test":0},
        "excluded_by_source":dict(excluded_source), "excluded_by_task":dict(excluded_task),
        "serialized_conversation_tokens":length_stats(serialized),
        "assistant_tokens":length_stats(assistants)}, diagnostics


def require_candidates(rows: Sequence[dict[str, Any]], requested: int, task: str) -> None:
    if len(rows) < requested:
        raise ValueError(f"{task}: requested {requested} rows but only {len(rows)} valid candidates remain after token filtering")


def split_rows(rows: Sequence[dict[str, Any]], sizes: dict[str, int]) -> dict[str, list[dict[str, Any]]]:
    needed = sum(sizes.values())
    if len(rows) < needed: raise ValueError(f"need {needed} unique rows, found {len(rows)}")
    shuffled = list(rows); random.Random(SEED).shuffle(shuffled)
    result, offset = {}, 0
    for name in ("train", "validation", "test"):
        result[name] = shuffled[offset:offset + sizes[name]]; offset += sizes[name]
    return result


def validate_splits(splits: dict[str, list[dict[str, Any]]], sizes: dict[str, int]) -> dict[str, bool]:
    all_hashes, all_groups = set(), set()
    for name, rows in splits.items():
        if len(rows) != sizes[name]: raise AssertionError(f"{name}: wrong row count")
        for row in rows:
            if [m["role"] for m in row["messages"]] != ["system", "user", "assistant"]: raise AssertionError("bad roles")
            if any(not str(m["content"]).strip() for m in row["messages"]): raise AssertionError("empty content")
            if prompt_hash(row["messages"][1]["content"]) != row["prompt_hash"]: raise AssertionError("bad prompt_hash")
            if row["prompt_hash"] in all_hashes or row["source_group"] in all_groups: raise AssertionError("duplicate across dataset/splits")
            all_hashes.add(row["prompt_hash"]); all_groups.add(row["source_group"])
            if row["task"] == "ifeval" and not constraint_satisfied(row["messages"][2]["content"], row["_constraint"]): raise AssertionError("constraint violation")
            if row.get("_correct_answer") and not row["messages"][2]["content"].endswith(row["_correct_answer"]): raise AssertionError("wrong MCQ answer label/text")
    return {"row_counts": True, "message_schema": True, "nonempty_content": True, "prompt_hashes": True,
            "no_duplicates": True, "split_disjoint": True, "mcq_answers": True,
            "constraints": True, "utf8_reload": True}


def _write_adapter_directory(output: Path, task: str, rows: list[dict[str, Any]], prep_stats: dict[str, Any],
                             revisions: dict[str, str], diagnostics: list[dict[str, Any]],
                             sizes: dict[str, int]) -> None:
    rows, duplicates = deduplicate(rows)
    splits = split_rows(rows, sizes); validations = validate_splits(splits, sizes)
    output.mkdir(parents=True, exist_ok=True)
    source_counts, category_counts, constraint_counts = Counter(), Counter(), Counter()
    for split, values in splits.items():
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in values:
                source_counts[row["source_dataset"]] += 1
                if row.get("_category"): category_counts[row["_category"]] += 1
                if row.get("_constraint"): constraint_counts[row["_constraint"]] += 1
                clean = {k: v for k, v in row.items() if not k.startswith("_")}
                handle.write(json.dumps(clean, ensure_ascii=False) + "\n")
        with (output / f"{split}.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle: json.loads(line)
    prior_duplicates = int(prep_stats.pop("duplicates_removed", 0))
    stats = {"task": task, "seed": SEED, "split_counts": {k: len(v) for k, v in splits.items()},
             "source_counts": dict(source_counts), "category_distribution": dict(category_counts),
             "duplicates_removed": prior_duplicates + duplicates, "validation": validations, **prep_stats}
    if constraint_counts:
        stats["constraint_counts"] = dict(constraint_counts)
        stats["constraint_validation_failures"] = 0
    manifest = {"task": task, "seed": SEED, "split_sizes": sizes, "schema_version": 1,
                "source_revisions": {key: revisions[key] for key in sorted(source_counts)}}
    (output / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output / "token_filter_exclusions.jsonl").open("w", encoding="utf-8") as handle:
        for row in diagnostics: handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_adapter(output: Path, task: str, rows: list[dict[str, Any]], prep_stats: dict[str, Any],
                  revisions: dict[str, str], sizes: dict[str, int] = SPLIT_SIZES,
                  diagnostics: list[dict[str, Any]] | None = None) -> None:
    if output.exists(): raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        _write_adapter_directory(staging, task, rows, prep_stats, revisions, diagnostics or [], sizes)
        # Re-read validation happens inside the writer; validate the in-memory final split again there too.
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_revisions(raw: Path) -> dict[str, str]:
    path = raw / "revisions.json"
    if not path.exists(): raise FileNotFoundError(f"required revision record is missing: {path}")
    revisions = json.loads(path.read_text(encoding="utf-8"))
    for dataset, expected in REVISIONS.items():
        if revisions.get(dataset) != expected: raise ValueError(f"revision mismatch for {dataset}: expected {expected}, got {revisions.get(dataset)}")
    return revisions


def run(raw: Path, output: Path, tokenizer_path: Path | None,
        allow_skip_token_validation: bool = False) -> None:
    revisions = load_revisions(raw)
    tokenizer = load_local_tokenizer(tokenizer_path, allow_skip_token_validation)
    rows, stats = prepare_ifeval(source_root(raw, "no_robots"))
    rows, removed = deduplicate(rows); stats["duplicates_removed"] = removed
    rows, token_stats, diagnostics = token_filter_candidates(rows, "ifeval", tokenizer, allow_skip_token_validation)
    require_candidates(rows, 9000, "ifeval"); stats["token_filter"] = token_stats
    write_adapter(output / "ifeval", "ifeval", deterministic_take(rows, 9000, "ifeval"), stats, revisions, diagnostics=diagnostics)
    rows, stats = prepare_bbh(raw / "cot_collection/data/CoT_collection_en.json")
    rows, removed = deduplicate(rows); stats["duplicates_removed"] = removed
    rows, token_stats, diagnostics = token_filter_candidates(rows, "bbh", tokenizer, allow_skip_token_validation)
    require_candidates(rows, 9000, "bbh"); stats["token_filter"] = token_stats
    selected = capped_categories(rows, 9000); require_candidates(selected, 9000, "bbh balanced selection")
    write_adapter(output / "bbh", "bbh", selected, stats, revisions, diagnostics=diagnostics)
    arc, am = load_gpqa_source(source_root(raw, "ai2_arc", "arc_challenge"), "arc_challenge")
    obqa, om = load_gpqa_source(source_root(raw, "openbookqa"), "openbookqa")
    sciq, sm = load_gpqa_source(source_root(raw, "sciq"), "sciq")
    arc, ad = deduplicate(arc); obqa, od = deduplicate(obqa); sciq, sd = deduplicate(sciq)
    candidates, cross_removed = deduplicate(arc + obqa + sciq)
    candidates, token_stats, diagnostics = token_filter_candidates(
        candidates, "gpqa", tokenizer, allow_skip_token_validation)
    arc = [row for row in candidates if row["source_dataset"] == "allenai/ai2_arc"]
    obqa = [row for row in candidates if row["source_dataset"] == "allenai/openbookqa"]
    sciq = [row for row in candidates if row["source_dataset"] == "allenai/sciq"]
    gpqa = deterministic_take(arc, len(arc), "arc") + deterministic_take(obqa, min(3000, len(obqa)), "obqa")
    gpqa += deterministic_take(sciq, max(0, 9000 - len(gpqa)), "sciq")
    require_candidates(gpqa, 9000, "gpqa")
    write_adapter(output / "gpqa", "gpqa", gpqa, {"malformed": am + om + sm,
                  "duplicates_removed": ad + od + sd + cross_removed,
                  "token_filter":token_stats}, revisions, diagnostics=diagnostics)
    available, malformed = [], 0
    source_duplicates = 0
    for name in ("commonsense_qa", "hellaswag", "winogrande"):
        values, bad = load_mmlupro_source(source_root(raw, name), name); malformed += bad
        values, removed = deduplicate(values); source_duplicates += removed
        available += values
    available, cross_removed = deduplicate(available); source_duplicates += cross_removed
    available, token_stats, diagnostics = token_filter_candidates(
        available, "mmlupro", tokenizer, allow_skip_token_validation)
    combined = []
    source_names = {"commonsense_qa":"tau/commonsense_qa", "hellaswag":"Rowan/hellaswag",
                    "winogrande":"allenai/winogrande"}
    for name, dataset in source_names.items():
        values = [row for row in available if row["source_dataset"] == dataset]
        require_candidates(values, 3000, f"mmlupro/{name}")
        combined += deterministic_take(values, 3000, name)
    write_adapter(output / "mmlupro", "mmlupro", combined, {"malformed": malformed,
                  "duplicates_removed": source_duplicates, "token_filter":token_stats},
                  revisions, diagnostics=diagnostics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path(__file__).parent / "raw")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "processed")
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--allow-skip-token-validation", action="store_true")
    args = parser.parse_args(); run(args.raw_dir, args.output_dir, args.tokenizer_path,
                                    args.allow_skip_token_validation)


if __name__ == "__main__": main()
