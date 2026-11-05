#!/usr/bin/env python3
"""Render published CSV aggregates; does not run benchmarks or estimate missing uncertainty."""
import argparse
import csv
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
COLORS = ['#4B5563', '#0072B2', '#E69F00', '#009E73']
MODES = ['Baseline', 'Async', 'In-place', 'Combined']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.titleweight':'bold','svg.hashsalt':'edgelora-results'})

def rows(name):
    with (ROOT/'experiments/paper_results'/name).open(encoding='utf-8-sig',newline='') as handle:
        return list(csv.DictReader(handle))

def one(data, **match):
    found = [r for r in data if all(r[k] == v for k,v in match.items())]
    if len(found) != 1: raise ValueError(f'Expected one CSV row for {match}; got {len(found)}')
    return found[0]

def number(row, key):
    value = float(row[key])
    if not math.isfinite(value): raise ValueError(f'Nonfinite {key}')
    return value

def finish(fig, out, name, caption):
    out.mkdir(parents=True,exist_ok=True)
    fig.text(.5,.015,caption,ha='center',va='bottom',fontsize=9,color='#374151')
    fig.subplots_adjust(bottom=.22,top=.78,wspace=.28)
    for ax in fig.axes:
        ax.grid(axis='y',alpha=.18)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
    fig.savefig(out/f'{name}.png',dpi=180,facecolor='white')
    fig.savefig(out/f'{name}.svg',facecolor='white',metadata={'Date':None})
    plt.close(fig)

def serving(out):
    data = rows('serving.csv')
    fig, axes = plt.subplots(1,2,figsize=(11.5,4.6))
    rates = [.3,.6,.9]
    for ax,metric,title,ylabel in zip(axes,['throughput','latency'],
            ['Completed throughput (higher is better)','Request completion time (lower is better)'],
            ['Requests / second','Seconds']):
        for index,(source,label) in enumerate(zip(['sync','async','inplace','async_acc'],MODES)):
            records = [one(data,mode=source,metric=metric,offered_rps=str(rate)) for rate in rates]
            y = [number(r,'median') for r in records]
            low = [v-number(r,'min') for v,r in zip(y,records)]
            high = [number(r,'max')-v for v,r in zip(y,records)]
            if any(v<0 for v in low+high): raise ValueError('Invalid min/median/max ordering')
            ax.errorbar(rates,y,yerr=[low,high],label=label,color=COLORS[index],marker=['o','s','^','D'][index],capsize=3,linewidth=1.8)
        ax.set(title=title,xlabel='Offered load (requests / second)',ylabel=ylabel,xticks=rates)
    fig.suptitle('Recorded serving results on Jetson Orin Nano Super',fontsize=16,y=.98)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.91),ncol=4,frameon=False)
    finish(fig,out,'serving','24 requests/run; 3 runs/point. Median with min-max bars (not confidence intervals).\nCompletion: median of run means. Finite traces include drain; output equivalence remains unresolved.')

PLOTS = {'serving': serving}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('figure',choices=list(PLOTS))
    parser.add_argument('--output-dir',type=Path,default=ROOT/'docs/figures')
    args = parser.parse_args()
    PLOTS[args.figure](args.output_dir)

if __name__ == '__main__': main()
