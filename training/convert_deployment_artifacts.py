#!/usr/bin/env python3
"""Convert local PEFT checkpoints to the Jetson runtime's deployment format.

Release wrapper around the converter used by the experiment. It does not copy
training data, model weights, or validation prompts into this repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

TASKS = ("securecode", "math", "ifeval", "bbh", "gpqa", "mmlupro")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True,
                        help="Patched EdgeLoRA edgelora/ directory")
    parser.add_argument("--base", type=Path, required=True,
                        help="Local Hugging Face base-model snapshot")
    parser.add_argument("--task-adapters", type=Path, required=True,
                        help="Directory with the six task-named PEFT subdirectories")
    parser.add_argument("--router-adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    converter = args.runtime_dir.resolve() / "convert_lora_to_gguf.py"
    if not converter.is_file() or not args.base.is_dir():
        parser.error("a patched converter and a local base-model snapshot are required")
    output_names = [f"{name}.gguf" for name in (*TASKS, "router")]
    output_names += ["router-classifier.bin"]
    if args.output_dir.exists() and not args.output_dir.is_dir():
        parser.error("output destination is not a directory")
    collisions = [name for name in output_names + ["adapter-manifest.json"]
                  if (args.output_dir / name).exists()]
    if collisions:
        parser.error("refusing to overwrite: " + ", ".join(collisions))
    inputs = {name: (args.task_adapters / name).resolve() for name in TASKS}
    inputs["router"] = args.router_adapter.resolve()
    before = {}
    for name, directory in inputs.items():
        for filename in ("adapter_config.json", "adapter_model.safetensors"):
            path = directory / filename
            if not path.is_file():
                parser.error(f"missing {name}/{filename}")
            before[f"{name}/{filename}"] = sha256(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir.resolve()
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               CUDA_VISIBLE_DEVICES="")
    for name, directory in inputs.items():
        command = [sys.executable, str(converter), "--base", str(args.base.resolve()),
                   "--outtype", "f32", "--outfile", str(out / f"{name}.gguf")]
        if name == "router":
            command += ["--router-classifier-out", str(out / "router-classifier.bin")]
        subprocess.run(command + [str(directory)], check=True, env=env)

    # Verify the classifier export against its saved PEFT tensor, as in the
    # historical export, including the 6-label / 3072-hidden-dimension contract.
    from safetensors.torch import load_file
    weights = load_file(str(inputs["router"] / "adapter_model.safetensors"),
                        device="cpu")["base_model.model.score.weight"]
    expected = weights.float().contiguous().numpy().astype("<f4").tobytes()
    actual = (out / "router-classifier.bin").read_bytes()
    if actual[:8] != b"ELRTRSCR" or len(actual) < 24:
        raise RuntimeError("invalid classifier header")
    if struct.unpack("<IIII", actual[8:24]) != (1, 1, 6, 3072):
        raise RuntimeError("classifier shape or format does not match the experiment")
    if actual[24:] != expected:
        raise RuntimeError("exported classifier differs from the PEFT checkpoint")
    for key, digest in before.items():
        name, filename = key.split("/", 1)
        if sha256(inputs[name] / filename) != digest:
            raise RuntimeError(f"input changed during conversion: {key}")
    manifest = {
        "source_inputs": before, "source_inputs_unchanged": True,
        "task_ids": dict(enumerate(TASKS)), "router_adapter_id": 6,
        "adapter_storage": "f32", "classifier_bit_exact": True,
        "outputs": {name: sha256(out / name) for name in output_names},
    }
    (out / "adapter-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                        encoding="utf-8")
    print(f"Wrote seven GGUF adapters, classifier, and manifest to {out}")


if __name__ == "__main__":
    main()
