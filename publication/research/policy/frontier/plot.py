"""Plot the reachable additional placements under a stated-rank requirement."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('rank_csv',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
    f=pd.read_csv(a.rank_csv);fig,ax=plt.subplots(figsize=(5.35,2.25));fig.subplots_adjust(left=.115,bottom=.255,right=.965,top=.94)
    ax.step(f.max_rank,f.maximum,color='#205474',where='post',lw=1.8)
    for x in (5,10,15,20):
        y=int(f.loc[f.max_rank==x,'maximum'].iloc[0]);ax.plot(x,y,'o',ms=4,color='#205474')
        ax.annotate(f'{y}',(x,y),xytext=((-5 if x==20 else 3),8),textcoords='offset points',ha='right' if x==20 else 'left',fontsize=8)
    ax.set(xlim=(.7,20.5),ylim=(-3,177),xticks=[1,5,10,15,20],yticks=[0,50,100,150],xlabel='Lowest acceptable position on the originally submitted list',ylabel='Additional placements')
    ax.tick_params(length=0,labelsize=8)
    ax.xaxis.label.set_size(9);ax.yaxis.label.set_size(9);ax.grid(axis='y',linewidth=.5,color='#ddd')
    for k in ['top','right','left']:ax.spines[k].set_visible(False)
    ax.spines['bottom'].set_color('#aaa')
    fig.savefig(a.out);plt.close(fig)
