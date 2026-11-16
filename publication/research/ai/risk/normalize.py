"""Translate higher-income-only paired count files into calibration input."""
import argparse
from pathlib import Path
import pandas as pd
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    refs=('transparent','pointwise','context')
    g=pd.concat(pd.read_csv(a.source/f'{s}-counts.csv.gz') for s in ('cal','summer','winter'))
    g=g[g.rollout.eq('higher') & g.policy.isin([x+s for x in refs for s in ('','_anchor')])].copy()
    g['policy']=g.policy.map(lambda x:x+'_top' if x in refs else x)
    assert (g.matched_lower==g.baseline_matched_lower+g.joint_gain-g.joint_loss).all()
    g=g.rename(columns={'joint_gain':'gain','joint_loss':'loss'})
    g[['split','design','month','borough','seed','policy','lower_n','gain','loss']].to_csv(a.output,index=False,compression='gzip')
