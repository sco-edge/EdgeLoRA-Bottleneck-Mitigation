#!/usr/bin/env python3
"""Print a small environment record without hostnames, usernames or env variables."""
import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
ROOT = Path(__file__).resolve().parents[1]

def report():
    packages = {}
    for name in ('torch','transformers','peft','datasets','safetensors','huggingface_hub'):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    try:
        revision = subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--untracked-files=no'],text=True))
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return dict(python=platform.python_version(),system=platform.system(),machine=platform.machine(),
                packages=packages,repository_commit=revision,tracked_changes=dirty)

def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(report(), indent=2))

if __name__ == '__main__': main()
