"""Rebuild slide tables and scientific figures from the packaged result extract."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
D=json.loads((ROOT/'data/results.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,
    'axes.spines.right':False,'pdf.fonttype':42,'savefig.facecolor':'white'})
BLUE='#304E68'; RUST='#A94722'; GREY='#73797E'

# Core table, using the underlying means and medians without intermediate rounding.
labels={'enumeration':'Exhaustive search','assignment':'Assignment cuts',
        'component':'Component cuts','prefix':'Prefix cuts','reverse_delete':'Reverse-delete',
        'greedy':'Blocker greedy','prefix_cover':'Prefix cover','full':'Full disclosure'}
rows=[]
for k,label in labels.items():
    v=D['methods'][k]
    timing='---' if k=='full' else f"{1000*v['elapsed_median']:.1f}"
    rows.append(f"{label} & {v['cost']['mean']:.2f} & {160-v['suboptimal_count']} & {timing}\\\\")
(ROOT/'data/core-table.tex').write_text(
 '\\begin{center}\n\\begin{tabular}{@{}lrrr@{}}\n\\toprule\nMethod & Mean cost & At min. & Median (ms)\\\\\n\\midrule\n'
 +'\n'.join(rows)+'\n\\bottomrule\n\\end{tabular}\n\\end{center}\n')
rows=[]
for k,label in [('prefix','With support'),('prefix_no_support','Without support')]:
    v=D['ablation'][k]
    rows.append(f"{label} & {v['certified_count']}/40 & {v['median_seconds']:.3f}\\\\")
(ROOT/'data/ablation-table.tex').write_text('\\begin{tabular}{@{}lrr@{}}\n\\toprule\nModel & Certified optima & Median seconds\\\\\n\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\n\\end{tabular}\n')
rows=[]
for k,label in [('1','One connected market'),('8','Eight local markets')]:
    v=D['scale'][k]['prefix']
    rows.append(f"{label} & {v['certified_count']}/16 & {v['median_seconds']:.3f} & {v['gap_max']:g}\\\\")
(ROOT/'data/scale-table.tex').write_text('\\begin{tabular}{@{}lrrr@{}}\n\\toprule\nRegime & Certified & Median (s) & Max. $U-L$\\\\\n\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\n\\end{tabular}\n')

# Report calls to certification only for completed runs. Timeout counts are
# censored observations, not measurements of the calls required to finish.
sizes=sorted({r['n'] for r in D['adversarial']})
records={(r['n'],r['variant']):r for r in D['adversarial']}
assert len(records)==len(D['adversarial'])==3*len(sizes)
rows=[]
for n in sizes:
    cells=[str(n)]
    for variant in ['assignment','component','prefix']:
        result=records[n,variant]
        if result['optimal']:
            assert result['rounds']>0
            assert abs(result['cost']-result['lower_bound'])<1e-7
            cells.append(str(result['rounds']))
        else:
            cells.append(r'\textit{Timeout}')
    rows.append(' & '.join(cells)+r'\\')
(ROOT/'data/rounds-table.tex').write_text(
    '\\begin{tabular}{@{}rrrr@{}}\n\\toprule\n'
    '$N$ & Assignment cuts & Component cuts & Prefix cuts\\\\\n\\midrule\n'
    +'\n'.join(rows)+'\n\\bottomrule\n\\end{tabular}\n')

# Separate axes keep gains and losses from being read as interchangeable units.
fig,axs=plt.subplots(1,2,figsize=(8.4,2.7))
for j,(metric,title,xmax) in enumerate([('buyer_change','Mean change in rank score',.35),('buyer_harmed_fraction','Fraction receiving a worse rank',.25)]):
    ax=axs[j]
    for i,(g,col) in enumerate([('lower',BLUE),('higher',GREY)]):
        v=D['social'][f'prefix_selected_{g}_access_{metric}']
        mean=v['mean'];lo,hi=v['ci95']
        ax.errorbar(mean,1-i,xerr=[[mean-lo],[hi-mean]],fmt='o',color=col,ms=7,capsize=4,lw=1.5)
    ax.set_yticks([1,0],['Lower access','Higher access'] if j==0 else [])
    ax.set_ylim(-.6,1.6);ax.set_xlim(0,xmax);ax.set_title(title,fontsize=12,pad=13)
    ax.set_xticks([0,.15,.30] if j==0 else [0,.10,.20]);ax.grid(axis='x',alpha=.16)
fig.subplots_adjust(left=.17,right=.97,top=.75,bottom=.20,wspace=.28)
fig.savefig(ROOT/'figures/distribution.pdf');plt.close(fig)
print('Rebuilt four tables and one figure from recorded data.')
