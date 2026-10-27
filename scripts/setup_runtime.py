#!/usr/bin/env python3
"""Fetch pinned EdgeLoRA and apply the paper runtime patch without building it."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "patches" / "runtime_manifest.json"


def run(*args):
    subprocess.run([str(arg) for arg in args], check=True)


def verify_files(destination, manifest):
    mismatches = []
    for relative, expected in manifest["files"].items():
        path = destination / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            mismatches.append(relative)
    if mismatches:
        raise RuntimeError("Runtime source hash mismatch: " + ", ".join(mismatches))
    changed = subprocess.check_output(
        ["git", "-C", str(destination), "diff", "--name-only", "HEAD", "--"], text=True
    ).splitlines()
    unexpected = set(changed) - set(manifest["files"])
    if unexpected:
        raise RuntimeError("Unexpected upstream source changes: " + ", ".join(sorted(unexpected)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "vendor" / "EdgeLoRA")
    parser.add_argument("--check", action="store_true", help="verify an already prepared runtime without modifying it")
    args = parser.parse_args()
    destination = args.destination.resolve()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    patch = MANIFEST.parent / manifest["patch_file"]
    if hashlib.sha256(patch.read_bytes()).hexdigest() != manifest["patch_sha256"]:
        raise RuntimeError("Patch checksum does not match the manifest")

    if destination.exists():
        head = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
        if head != manifest["upstream_commit"]:
            raise RuntimeError("Existing destination is not at the pinned upstream commit; choose a new --destination")
        verify_files(destination, manifest)
        print("Verified runtime:", destination / "edgelora")
        return
    if args.check:
        raise RuntimeError("Runtime is absent; run setup_runtime.py without --check first")

    destination.parent.mkdir(parents=True, exist_ok=True)
    run("git", "init", "--quiet", destination)
    run("git", "-C", destination, "config", "core.autocrlf", "false")
    run("git", "-C", destination, "remote", "add", "origin", manifest["upstream_url"])
    run("git", "-C", destination, "fetch", "--depth", "1", "origin", manifest["upstream_commit"])
    run("git", "-C", destination, "checkout", "--detach", manifest["upstream_commit"])
    run("git", "-C", destination, "apply", "--check", patch)
    run("git", "-C", destination, "apply", patch)
    verify_files(destination, manifest)
    print("Prepared runtime:", destination / "edgelora")
    print("Next: bash scripts/build_runtime.sh")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError, OSError) as exc:
        print("setup_runtime:", exc, file=sys.stderr)
        sys.exit(1)
