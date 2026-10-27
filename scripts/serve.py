#!/usr/bin/env python3
"""Launch one paper variant; asynchronous loading is also a per-request option."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODES = ("baseline", "async", "inplace", "combined")


def make_command(config, mode, host, port):
    def path(value):
        p = Path(value).expanduser()
        return str(p if p.is_absolute() else ROOT / p)

    tasks = config["task_adapters"]
    if len(tasks) != 6:
        raise ValueError("Expected six task adapters in ID order 0 through 5.")
    files = [path(config[k]) for k in ("binary", "base_model", "router_adapter", "router_classifier")]
    files += [path(p) for p in tasks]
    command = [files[0], "-m", files[1], "-ngl", str(config.get("gpu_layers", 99)),
               "-c", str(config.get("context_size", 2048)),
               "-b", str(config.get("batch_size", 512)),
               "-ub", str(config.get("micro_batch_size", 512)),
               "-np", str(config.get("parallel_slots", 2)),
               "-t", str(config.get("threads", 4)), "--host", host, "--port", str(port),
               "--log-format", "json", "--batch-lora", "true", "--lora-init-without-apply",
               "--adapter-cache-size", str(config.get("adapter_cache_size", 2)),
               "--router-lora-id", "6", "--router-classifier", files[3],
               "--router-label-map", "0,1,2,3,4,5", "--router-top-k", "1",
               "--slot-save-path", path(config.get("slot_save_path", "results/slots"))]
    for p in files[4:] + [files[2]]:
        command += ["--lora", p]
    flags = {"EDGELORA_ACC_AUTO": "1" if mode in ("inplace", "combined") else "0",
             "EDGELORA_ACC_SLICE": "0", "EDGELORA_META_DETAIL": "0"}
    policy = "backend-async" if mode in ("async", "combined") else "sync"
    return command, flags, policy, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/runtime.json")
    parser.add_argument("--mode", choices=MODES, default="combined")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--dry-run", action="store_true", help="Print command without starting a server.")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    command, flags, policy, files = make_command(config, args.mode, args.host, args.port)
    print(json.dumps({"mode": args.mode, "argv": command, "environment": flags,
                      "required_request_field": {"adapter_load_policy": policy}}, indent=2), flush=True)
    if args.dry_run:
        return 0
    missing = [p for p in files if not Path(p).is_file()]
    if missing:
        parser.error("Missing runtime/model artifacts:\n" + "\n".join(missing))
    slot_dir = Path(command[command.index("--slot-save-path") + 1])
    slot_dir.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        try:
            sock.bind((args.host, args.port))
        except OSError as exc:
            parser.error(f"Port is unavailable: {exc}")
    # Match the recorded runs: clear inherited experimental overrides.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("EDGELORA_", "GGML_", "CAUSE_", "LCP_"))}
    env.update(flags)
    # Foreground execution keeps Ctrl+C and server ownership visible to the user.
    try:
        return subprocess.call(command, cwd=ROOT, env=env)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
