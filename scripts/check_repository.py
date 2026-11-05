#!/usr/bin/env python3
"""Check tracked Python syntax, JSON and local Markdown links without GPU imports."""
import argparse
import ast
import json
from pathlib import Path
import re
import subprocess
import urllib.parse
ROOT = Path(__file__).resolve().parents[1]

def check(root):
    names = subprocess.check_output(['git','-C',str(root),'ls-files','-z'], text=True).split('\0')
    errors = []
    for name in filter(None,names):
        p = root/name
        try:
            if p.suffix == '.py': ast.parse(p.read_text(encoding='utf-8'), filename=name)
            elif p.suffix == '.json': json.loads(p.read_text(encoding='utf-8'))
            elif p.suffix == '.md':
                text = re.sub(r'```.*?```', '', p.read_text(encoding='utf-8'), flags=re.S)
                for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text):
                    target = target.strip().split(' "')[0].strip('<>')
                    url = urllib.parse.urlsplit(target)
                    if url.scheme or url.netloc or not url.path: continue
                    local = (p.parent/urllib.parse.unquote(url.path)).resolve()
                    if not local.is_relative_to(root.resolve()) or not local.exists():
                        errors.append(f'{name}: missing local link {target}')
        except (OSError, SyntaxError, ValueError) as exc:
            errors.append(f'{name}: {exc}')
    return errors

def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    errors = check(ROOT)
    if errors: raise SystemExit('\n'.join(errors))
    print('Tracked Python, JSON and local Markdown targets are valid.')

if __name__ == '__main__': main()
