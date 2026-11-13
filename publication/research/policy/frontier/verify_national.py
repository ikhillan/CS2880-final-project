"""Reconstruct the solver witnesses and certify them by independently derived dual bounds."""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from solve import model_menu, witness, SCALE
from dual import certify
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['previous','inputs','features','fit','out']:p.add_argument(name,type=Path)
    args=p.parse_args();sys.path.insert(0,str(args.previous.resolve()))
    from run import Market
    m=Market(args.inputs,2026,positive_scores=True);orig=pd.read_csv(args.out/'frontier.csv')
    results=[]
    for name in ('informed','blind'):
        prob,_,worst=model_menu(m,args.features,args.fit,name)
        c=np.rint(np.maximum(0.,prob[worst[m.a]]-prob)*SCALE).astype(np.int64)
        for r in orig[(orig.model==name)&(orig.permutation==0)&(orig.fraction>0)].itertuples():
            selected,objective=witness(m,c,r.target,0)
            assert objective==int(r.cost_integer)
            checks=certify(m,c,selected,r.target)
            results.append(dict(model=name,target=r.target,**checks))
            print(name,r.target,checks['dual'],flush=True)
    pd.DataFrame(results).to_csv(args.out/'national-duals.csv',index=False)
