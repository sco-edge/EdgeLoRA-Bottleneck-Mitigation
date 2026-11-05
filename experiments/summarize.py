#!/usr/bin/env python3
"""Aggregate valid portable experiment runs; never substitute HTTP gaps for Max TBT."""
import argparse
import collections
import json
import math
from pathlib import Path
import statistics


def aggregate(root):
    groups = collections.defaultdict(list)
    failures = []
    incomplete = [str(p.parent.relative_to(root)) for p in sorted(root.rglob('config.json'))
                  if not (p.parent / 'summary.json').is_file()]
    for file in sorted(root.rglob('summary.json')):
        if not (file.parent / 'config.json').exists():
            continue
        run = json.loads(file.read_text(encoding='utf-8'))
        config = json.loads((file.parent / 'config.json').read_text(encoding='utf-8'))
        if not run.get('valid'):
            failures.append(str(file.relative_to(root)))
            continue
        key = tuple(config.get(k) for k in ('experiment', 'mode', 'rate', 'case', 'state', 'input', 'ongoing', 'cache'))
        metrics = run.get('metrics')
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError(f'{file}: valid run requires a nonempty metrics object')
        if type(config.get('block')) is not int or config['block'] < 0:
            raise ValueError(f'{file}: block must be a nonnegative integer')
        for field, value in metrics.items():
            if isinstance(value, bool) or (isinstance(value, (int, float)) and not math.isfinite(value)):
                raise ValueError(f'{file}: invalid numeric metric {field}')
        groups[key].append((config['block'], metrics))
    result = []
    for key, records in groups.items():
        if len({r[0] for r in records}) != len(records):
            raise ValueError(f'Duplicate block in group {key}; do not count reruns as independent repeats')
        fields = [k for k, v in records[0][1].items() if isinstance(v, (int, float))]
        expected_fields = set(fields)
        for block, metrics in records:
            actual_fields = {k for k, v in metrics.items() if type(v) in (int, float)}
            if actual_fields != expected_fields:
                raise ValueError(f'Inconsistent numeric metrics in group {key}, block {block}')
        result.append(dict(configuration=dict(zip(('experiment', 'mode', 'rate', 'case', 'state', 'input', 'ongoing', 'cache'), key)),
             n=len(records), blocks=sorted(r[0] for r in records), metrics={field:dict(
                 median=statistics.median(r[1][field] for r in records),
                 minimum=min(r[1][field] for r in records), maximum=max(r[1][field] for r in records)) for field in fields}))
    return dict(groups=result, invalid_runs=failures, incomplete_runs=incomplete, error_bars='minimum and maximum; not confidence intervals')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    if not args.results.is_dir():
        parser.error('Results folder does not exist')
    print(json.dumps(aggregate(args.results), indent=2))


if __name__ == '__main__':
    main()
