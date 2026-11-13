"""Evaluate one frozen enrollment-trained ensemble against earlier offer-trained lists."""
import argparse
import json
import hashlib
from pathlib import Path
import sys
import tempfile
import numpy as np
import pandas as pd

def main(args):
    sys.path[:0]=[str(args.previous.resolve()),str(args.frontier.resolve())]
    from run import Market,Kernel,measures
    from solve import model_menu
    import check
    check.main(args)
    # model_menu reproduces archived seeded tie rules for the learned rows.
    m=Market(args.inputs,2026,positive_scores=True)
    rows=[]
    with tempfile.TemporaryDirectory() as temp:
        kern=Kernel(temp)
        full=m.batch(m.cap)
        for name in ('blind','informed'):
            _,mask,_=model_menu(m,args.features,args.fit,name)
            rank=(~mask).astype('uint8')
            z=kern.solve(m,m.cap,rank,1);m.verify(m.cap,rank,1,z)
            rows.append(dict(model=name,**measures(m,rank,1,z,full)))
            # Independent simultaneous DA on the explicitly compressed menu.
            import copy
            idx=np.flatnonzero(mask);reduced=copy.copy(m)
            reduced.a=m.a[idx];reduced.school=m.school[idx];reduced.priority=m.priority[idx]
            reduced.degree=np.bincount(reduced.a,minlength=m.n);reduced.offset=np.r_[0,reduced.degree.cumsum()]
            match=reduced.batch(m.cap);v=match>=0;match[v]=idx[match[v]]
            assert np.array_equal(z,match)
    out=args.out
    pd.DataFrame(rows).to_csv(out/'da.csv',index=False)
    h=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    for f in (args.out.parent/'models').glob('*.json'):
        a=json.loads(f.read_text())
        assert h(f.with_suffix('.pt'))==a['weights']
        assert h(args.retarget/'features2025.npz')==a['train_features_sha'] and h(args.features/'features2026.npz')==a['test_features_sha']
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('previous','frontier','inputs','enrollment','features','fit','out','retarget'):p.add_argument(n,type=Path)
    main(p.parse_args())
