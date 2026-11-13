"""Join the primary k=5 assignment results to the matching capacity of the identical menus."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from benchmark import FEATURE_NAMES, features, mask_order, partner_frequency, read_profiles, rng
from tie_check import max_size


def main(source,out):
    s=2
    m=read_profiles(source)[s];e=m[2]
    xx=features(m)
    scores={'own':-m[0],'conditional':partner_frequency(s,m,220)[:,:-1]}
    for f in json.loads((out/'fits.json').read_text()):
        if f['test']==3:
            keys=f['coefficients']
            scores[f['model']]=xx[:,:,[FEATURE_NAMES.index(k) for k in keys]] @ np.array(list(keys.values()))
    with (out/'blocks.csv').open() as f:
        baseline={(r['arm'],int(r['block'])):int(r['placed']) for r in csv.DictReader(f) if r['k']=='5' and r['cohort']=='3'}
    stats={p:[0,0,0] for p in [*scores,'uniform']}
    sums=[]
    fixed={p:mask_order(e,sco,rng(s,240).random(e.shape),5) for p,sco in scores.items()}
    for b in range(32):
        u=rng(s,440,b).random((256,*e.shape))
        U=e & (np.argsort(np.argsort(np.where(e,u,2),axis=2),axis=2)<5)
        for p in stats:
            if p=='uniform':
                maximum=sum(max_size(g) for g in U)
                distinct=int(U.any(1).sum())
                if b==0:
                    assert all(max_size(g)==(maximum_bipartite_matching(csr_matrix(g),perm_type="column")>=0).sum() for g in U[:40])
            else:
                maximum=256*max_size(fixed[p])
                distinct=256*int(fixed[p].any(0).sum())
            actual=baseline[p,b]
            assert maximum>=actual
            stats[p][0]+=distinct;stats[p][1]+=maximum;stats[p][2]+=actual
            sums.append(dict(block=b,arm=p,n_refinements=256,distinct=distinct,max_cardinality=maximum,placed=actual))
    report=[]
    for p,(distinct,maximum,actual) in stats.items():
        report.append(dict(arm=p,mean_distinct=distinct/8192,mean_maximum=maximum/8192,mean_placed=actual/8192,
                           gap_count=(maximum-actual)/8192,capacity_delta_pp=100*(maximum-stats['uniform'][1])/(8192*len(e)),
                           gap_delta_pp=100*((maximum-actual)-(stats['uniform'][1]-stats['uniform'][2]))/(8192*len(e)),
                           placed_delta_pp=100*(actual-stats['uniform'][2])/(8192*len(e))))
    for file,rows in [('primary-capacity-blocks.csv',sums),('primary-capacity.csv',report)]:
        with (out/file).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(report)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=Path(__file__).parent)
    a=parser.parse_args();main(a.source,a.out)
