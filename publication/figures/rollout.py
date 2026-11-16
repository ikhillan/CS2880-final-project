"""Pointwise partial-rollout effects for the two higher-income scoring rules."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

args=argparse.ArgumentParser(description=__doc__)
args.add_argument('evidence',type=Path)
args.add_argument('output',type=Path)
a=args.parse_args()
series=json.loads((a.evidence/'rollout.json').read_text())['curves']
slope=json.loads((a.evidence/'results.json').read_text())['panels']['enriched_4']['isolated']['pp']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'pdf.fonttype':42})
fig,(left,right)=plt.subplots(1,2,figsize=(5.7,2.68),gridspec_kw={'width_ratios':[1.4,1]},layout='constrained')
for ax in (left,right):
    ax.axhline(0,color='#70818b',lw=.75)
    ax.spines[['top','right']].set_visible(False)
    ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    ax.tick_params(length=3)
    for j,(w,color,mark) in enumerate([(1,'#934418','s'),(4,'#245b79','o')]):
        if ax is right and w==1:
            continue
        rows=series[f'enriched_{w}']
        p=np.array([r['p'] for r in rows]);vals=np.array([r['pp'] for r in rows]);ci=np.array([r['pointwise95'] for r in rows])
        ix=p<=.2 if ax is right else p<=1
        ax.fill_between(p[ix],ci[ix,0],ci[ix,1],color=color,alpha=.17,lw=0)
        ax.plot(p[ix],vals[ix],marker=mark,ms=2.7,color=color,lw=1.6,label=f'Price penalty {w}')
    ax.set_xlabel('Share receiving the new list')
left.set(title='Full range of rollout',ylabel='Lower-income match change (pp)',xlim=(-.02,1.02),xticks=[0,.25,.5,.75,1])
left.legend(frameon=False,loc='upper left',fontsize=8)
right.set(title='Early rollout, price penalty 4',xlim=(-.005,.205),xticks=[0,.05,.10,.15,.20])
right.plot([0,.2],[0,slope*.2],ls=':',lw=1.5,color='#314955',label='Initial tangent')
right.legend(frameon=False,loc='upper left',fontsize=8)
fig.savefig(a.output,metadata={'Title':'Partial adoption of higher-income shortlists','Author':'','CreationDate':None,'ModDate':None})
