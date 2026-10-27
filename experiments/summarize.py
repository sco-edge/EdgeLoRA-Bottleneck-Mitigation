#!/usr/bin/env python3
"""Aggregate valid portable experiment runs; never substitute HTTP gaps for Max TBT."""
import argparse
import collections
import json
from pathlib import Path
import statistics


def aggregate(root):
    groups = collections.defaultdict(list)
    failures = []
    for file in sorted(root.rglob('summary.json')):
        if not (file.parent / 'config.json').exists():
            continue
        run = json.loads(file.read_text(encoding='utf-8'))
        config = json.loads((file.parent / 'config.json').read_text(encoding='utf-8'))
        if not run.get('valid'):
            failures.append(str(file.relative_to(root)))
            continue
        key = tuple(config.get(k) for k in ('experiment', 'mode', 'rate', 'case', 'state', 'input', 'ongoing', 'cache'))
        groups[key].append((config['block'], run['metrics']))
    result = []
    for key, records in groups.items():
        if len({r[0] for r in records}) != len(records):
            raise ValueError(f'Duplicate block in group {key}; do not count reruns as independent repeats')
        fields = [k for k, v in records[0][1].items() if isinstance(v, (int, float))]
        result.append(dict(configuration=dict(zip(('experiment', 'mode', 'rate', 'case', 'state', 'input', 'ongoing', 'cache'), key)),
             n=len(records), blocks=sorted(r[0] for r in records), metrics={field:dict(
                 median=statistics.median(r[1][field] for r in records),
                 minimum=min(r[1][field] for r in records), maximum=max(r[1][field] for r in records)) for field in fields}))
    return dict(groups=result, invalid_runs=failures, error_bars='minimum and maximum; not confidence intervals')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    if not args.results.is_dir():
        parser.error('Results folder does not exist')
    print(json.dumps(aggregate(args.results), indent=2))


if __name__ == '__main__':
    main()
