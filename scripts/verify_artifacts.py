#!/usr/bin/env python3
"""Verify exported adapter files against a local conversion manifest (CPU only)."""
import argparse
import hashlib
import json
from pathlib import Path
import re

def verify(directory, manifest):
    outputs = manifest.get('outputs')
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError('Manifest requires a nonempty outputs mapping')
    checked = []
    for name, expected in outputs.items():
        if not isinstance(name, str) or '/' in name or '\\' in name or name in ('.', '..') or Path(name).is_absolute():
            raise ValueError('Manifest output names must be simple filenames')
        if not isinstance(expected, str) or not re.fullmatch('[0-9a-fA-F]{64}', expected):
            raise ValueError(f'Invalid SHA-256 for {name}')
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Missing regular artifact file: {name}')
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''): digest.update(chunk)
        if digest.hexdigest() != expected.lower():
            raise ValueError(f'Checksum mismatch: {name}')
        checked.append(name)
    return checked

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--manifest', type=Path, help='Default: DIRECTORY/adapter-manifest.json')
    args = parser.parse_args()
    try:
        manifest = json.loads((args.manifest or args.directory/'adapter-manifest.json').read_text(encoding='utf-8'))
        if not isinstance(manifest, dict): raise ValueError('Manifest must be an object')
        checked = verify(args.directory, manifest)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({'verified': checked}, indent=2))

if __name__ == '__main__': main()
