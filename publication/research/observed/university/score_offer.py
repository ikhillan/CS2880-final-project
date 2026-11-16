"""Compare the fixed real-offer ensembles against the same support and assignments."""
import argparse
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from run import Kernel,Market,features,history,ordering,measures

def evaluate(root,data,fit,out):
    m=Market(root,2026,positive_scores=True)
    status=Market(root,2026)
    assert np.array_equal(status.a,m.a) and np.array_equal(status.school,m.school) and np.array_equal(status.r,m.r)
    assert np.array_equal(status.recorded,m.recorded) and np.array_equal(status.score,m.score)
    cols=np.load(data/'test-col.npy')
    full_rows=[]
    for name in ('blind','informed'):
        ensemble=fit/f'{name}-ensemble.npz'
        if ensemble.exists():
            with np.load(ensemble) as saved:prob=saved['p']
        else:
            prob=sum(np.load(fit/f'{name}-{s}.npy').astype(float) for s in [8431,8432,8433])/3
        assert prob.shape==(m.n,21) and np.allclose(prob.sum(1),1.,atol=2e-5)
        prob=prob[m.a,cols]
        # Use the same seeded tie lottery for the two learned models independently of submitted rank.
        tie=np.random.default_rng(118844).permutation(len(cols))
        sorted_i=np.lexsort((tie,-prob,m.a));rank=np.empty(len(cols),np.uint8)
        rank[sorted_i]=(np.arange(len(cols))-m.offset[m.a[sorted_i]]).astype(np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            solver=Kernel(tmp);full=solver.solve(m,m.cap,rank,255)
            for k in [1,3,5,10]:
                selected=solver.solve(m,m.cap,rank,k);m.verify(m.cap,rank,k,selected)
                full_rows.append(dict(capacity='published_plus_overbook',policy=name,k=k,draw=-1,**measures(m,rank,k,selected,full)))
    refs=pd.read_csv(out/'counts.csv')
    refs=refs[(refs.policy!='trained')&(refs.capacity=='published_plus_overbook')&((refs.draw<32))]
    # These identical menus use the same deterministic control/ties and conditional support as the primary analysis.
    pd.concat([pd.DataFrame(full_rows),refs],ignore_index=True).to_csv(out/'offer-counts.csv',index=False)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','data','fit','out'):p.add_argument(n,type=Path)
    a=p.parse_args();evaluate(a.root,a.data,a.fit,a.out)
