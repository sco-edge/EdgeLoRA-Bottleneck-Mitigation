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
    svg = out/f'{name}.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines()) + '\n',
                   encoding='utf-8', newline='\n')
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

def latency(out):
    data = rows('latency.csv')
    conditions = ['Repeat 6','A2 / 16','A4 / 16','A2 / 128','A4 / 128','A2 / 256','A4 / 256']
    fig, axes = plt.subplots(1,2,figsize=(12,5.2))
    for ax,metric,title in zip(axes,['A Max TBT','B TTFT'],['Ongoing requests: Max TBT','Incoming requests: TTFT']):
        for index,label in enumerate(['Baseline','Combined']):
            y = [number(one(data,experiment='serving',condition=c,metric=metric,configuration=label),'median_ms')/1000 for c in conditions]
            ax.bar([i+(index-.5)*.36 for i in range(len(y))],y,width=.36,label=label,color=COLORS[0 if index==0 else 3])
        ax.set(title=title,ylabel='Seconds (lower is better)',xticks=range(len(conditions)),xticklabels=conditions)
        ax.tick_params(axis='x',rotation=35,labelsize=8)
    fig.suptitle('Matched-arrival latency measurements',fontsize=16,y=.98)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.91),ncol=2,frameon=False)
    finish(fig,out,'latency','A2/A4: 2/4 ongoing requests; suffix: incoming input tokens, cold adapter file. Repeat 6: warm file, cache 2.\nMedians over 30 runs. Fanout Max TBT averages per-request maxima within each run. Dispersion is not in the CSV.')

def ablation(out):
    data = rows('latency.csv')
    fig, axes = plt.subplots(1,2,figsize=(11.5,4.8))
    for ax,state,title in zip(axes,['R','M'],['Resident task adapter','Nonresident task adapter; warm file']):
        for index,label in enumerate(MODES):
            y = [number(one(data,experiment='ablation',condition=f'{n} {state}',metric='A Max TBT',configuration=label),'median_ms')/1000 for n in (16,128,256)]
            ax.bar([i+(index-1.5)*.19 for i in range(3)],y,width=.19,color=COLORS[index],label=label)
        ax.set(title=title,ylabel='Ongoing Max TBT (seconds)',xlabel='Incoming input tokens',xticks=range(3),xticklabels=['16','128','256'],ylim=(0,9))
    fig.suptitle('Separating adapter preparation and accumulation costs',fontsize=16,y=.98)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.91),ncol=4,frameon=False)
    finish(fig,out,'ablation','One ongoing request; 2 slots; cache 2; adapter files warm. Lower is better.\nMedian of 12 matched runs; CSV contains no dispersion. Same vertical scale in both panels.')

def motivation(out):
    data = rows('motivation.csv')
    fig,axes = plt.subplots(1,2,figsize=(11.5,4.7))
    left = [number(one(data,panel='a',condition=c),'median') for c in ('resident','sync loading')]
    bars = axes[0].bar(['Resident','Synchronous loading'],left,color=[COLORS[0],COLORS[1]],width=.55)
    axes[0].bar_label(bars,fmt='%.1f ms',padding=4)
    axes[0].set(title='Loading-related decode stall',ylabel='Max TBT (ms)',ylim=(0,650))
    right = [number(one(data,panel='b',condition=str(n)),'median') for n in (16,128,256)]
    bars = axes[1].bar(['16','128','256'],right,color=COLORS[2],width=.55)
    axes[1].bar_label(bars,fmt='%.2f%%',padding=4)
    axes[1].set(title='Accumulation share of kernel-duration sums',ylabel='Percent of total kernel-duration sum',xlabel='Incoming input tokens',ylim=(0,100))
    fig.suptitle('Two measured sources of serving delay',fontsize=16,y=.98)
    finish(fig,out,'motivation','Left: baseline, 30 runs. Right: 6 profiled runs/length, both adapters resident, CUDA Graphs disabled.\nDifferent experiments and units; kernel-duration shares are not wall-time fractions or measured bandwidth.')

PLOTS = {'serving': serving, 'latency': latency, 'ablation': ablation, 'motivation': motivation}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('figure',choices=list(PLOTS))
    parser.add_argument('--output-dir',type=Path,default=ROOT/'docs/figures')
    args = parser.parse_args()
    PLOTS[args.figure](args.output_dir)

if __name__ == '__main__': main()
