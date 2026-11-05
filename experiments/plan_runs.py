#!/usr/bin/env python3
"""Print the balanced 36-run throughput order without starting experiments."""
import argparse
import json

def plan():
    rates = [0.3, 0.6, 0.9]
    modes = ['baseline', 'async', 'inplace', 'combined']
    rows = []
    for block in range(3):
        ordered_rates = rates[block:] + rates[:block]
        for index, rate in enumerate(ordered_rates):
            shift = (block + index) % len(modes)
            for mode in modes[shift:] + modes[:shift]:
                rows.append(dict(block=block, rate=rate, mode=mode,
                                 run_name=f'{mode}-r{round(rate*10):02d}-b{block}'))
    return rows

def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(plan(), indent=2))

if __name__ == '__main__': main()
