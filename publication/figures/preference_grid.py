"""Draw all 36 fresh-block estimates from the published summary."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

cli = argparse.ArgumentParser(description=__doc__)
cli.add_argument('summary', type=Path)
cli.add_argument('out', type=Path)
args = cli.parse_args()
data = json.loads(args.summary.read_text())
rows = data['estimates']
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8.7, 'pdf.fonttype':42})
fig, axes = plt.subplots(2, 2, figsize=(3.25, 2.75), layout='constrained')
limit = np.ceil(max(abs(r['UT_minus_UU_lower']['estimate']) for r in rows))
for i,(design,title) in enumerate([('ordinary','Ordinary'),('enriched','Enriched')]):
    for j,(budget,budget_title) in enumerate([('k5','Five listings'),('q40','40% of options')]):
        ax=axes[i,j]
        if design == 'ordinary':
            subset=[r for r in rows if r['design'] in ('ordinary','primary','uniform') and r['budget']==budget]
        else:
            subset=[r for r in rows if r['design']==design and r['budget']==budget]
        assert len(subset)==9
        matrix=np.empty((3,3))
        for r in subset:
            matrix[[1.,2.,4.].index(r['w_p']),[.5,.75,1.25].index(r['sigma_r'])]=r['UT_minus_UU_lower']['estimate']
        ax.imshow(matrix,cmap='PuOr_r',norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),aspect='auto')
        for y,x in np.ndindex(matrix.shape):
            val=matrix[y,x]
            ax.text(x,y,f'{val:+.2f}'.replace('-', '−'),ha='center',va='center',color='white' if abs(val)>.58*limit else '#171717',fontsize=8.6)
        ax.set_xticks(range(3),['0.50','0.75','1.25'])
        ax.set_yticks(range(3),['1','2','4'])
        ax.tick_params(length=0,labelbottom=i==1,labelleft=j==0,pad=2)
        if i==0:
            ax.set_title(budget_title,fontsize=9.2,pad=3)
        if j==0:
            ax.set_ylabel(title,fontsize=9.1)
        for sp in ax.spines.values():
            sp.set_visible(False)
fig.supxlabel('Private taste noise (σ)',fontsize=8.9)
fig.savefig(args.out,metadata={'Title':'Lower-income match-rate spillover across preference settings','CreationDate':None,'ModDate':None,'Author':''})
