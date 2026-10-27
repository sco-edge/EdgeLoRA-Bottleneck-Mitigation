#!/usr/bin/env python3
"""Download one pinned SecuCoder revision on the RTX training host."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

REPOSITORY = "ivitopow/SecuCoder"
REVISION = "7d5149ab257ad938384cacef9fd1827174630d0d"
LICENSE = "CC-BY-NC-SA-4.0"
EXPECTED_SPLITS = {"train": 5708, "validation": 317, "test": 317}


def download(destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing raw data: {destination}")
    try:
        from datasets import DatasetDict, load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required on the RTX download host") from exc
    dataset = load_dataset(REPOSITORY, revision=REVISION)
    if not isinstance(dataset, DatasetDict) or set(dataset) != set(EXPECTED_SPLITS):
        raise ValueError(f"unexpected splits: {list(dataset)}")
    counts = {split: len(dataset[split]) for split in EXPECTED_SPLITS}
    if counts != EXPECTED_SPLITS:
        raise ValueError(f"split count mismatch: expected {EXPECTED_SPLITS}, got {counts}")
    dataset.save_to_disk(str(destination))
    (destination / "download_manifest.json").write_text(json.dumps({
        "repository": REPOSITORY,
        "revision": REVISION,
        "license": LICENSE,
        "split_counts": counts,
        "downloaded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "format": "datasets.DatasetDict.save_to_disk",
    }, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "raw/secucoder")
    args = parser.parse_args()
    download(args.output)


if __name__ == "__main__": main()
