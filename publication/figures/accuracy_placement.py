"""Plot group-specific paired contrasts on two explicitly labeled scales."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def main():
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path);args=p.parse_args()
    data=pd.read_csv(args.input)
    view=data[(data.policy=='context')&(data.rollout=='all')&(data.contrast=='minus_uniform')].set_index(['split','design','metric'])
    panels=[('summer','uniform'),('winter','uniform'),('summer','enriched'),('winter','enriched')]
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.22),sharey=True)
    fig.subplots_adjust(left=.185,right=.985,bottom=.24,top=.79,wspace=.20)
    for ax,metric,heading,xlim,step in zip(axes,['oracle_hit','total_rate'],['Full-market partner on shortlist','Renters placed by DA'],[(-.9,25),(-4.25,.5)],[5,1]):
        for i,(season,design) in enumerate(panels):
            row=view.loc[(season,design,metric)]
            color='#264B63' if design=='uniform' else '#985132'
            ax.errorbar(row.estimate,3-i,xerr=np.array([[row.estimate-row.lo],[row.hi-row.estimate]]),fmt='o' if design=='uniform' else 's',color=color,ms=4.2,lw=1.3,capsize=3,zorder=3)
        ax.axvline(0,color='#606060',lw=.85)
        ax.grid(axis='x',color='#DFDFDF',lw=.55)
        ax.set_title(heading,fontsize=9.7);ax.set_xlim(*xlim);ax.set_ylim(-.5,3.5)
        ax.xaxis.set_major_locator(MultipleLocator(step));ax.tick_params(axis='both',length=0,labelsize=8)
        ax.set_xlabel('Change, percentage points',fontsize=8)
        for side in ('top','right','left'):ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color('#B0B0B0')
    axes[0].set_yticks([3,2,1,0],['Summer ordinary','Winter ordinary','Summer enriched','Winter enriched'])
    fig.savefig(args.output)
if __name__=='__main__':main()
