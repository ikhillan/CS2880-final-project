"""Independent round-synchronous DA plus direct Monte Carlo/result sanity checks."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import replay as R

def synchronous(order, priorities, graph):
    n,m=graph.shape
    row_rank=np.argsort(order,axis=1)
    used=np.zeros_like(graph);held=np.zeros_like(graph)
    for _ in range(graph.sum()+1):
        free=~held.any(1);avail=graph&~used&free[:,None]
        i=np.where(avail.any(1))[0]
        if not len(i): break
        j=np.argmin(np.where(avail[i],row_rank[i],m),axis=1)
        used[i,j]=True;held[i,j]=True
        for j in np.where(held.any(0))[0]:
            cand=np.flatnonzero(held[:,j]);winner=cand[np.argmin(priorities[cand,j])]
            held[cand,j]=False;held[winner,j]=True
    return np.where(held.any(1),held.argmax(1),-1)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();df=R.read_data(a.data);checked=0;baseline_errors=[];edges=[]
    counts=pd.read_csv(a.out/'wave-results.csv')
    for side in [0,1]:
        markets=R.build_markets(df,side)
        for known in [0,1]:
            pools,scores=R.fit(markets,known)
            for m in markets:
                if m.wave not in R.TEST:continue
                pool=pools[m.wave];deg=pool.sum(1)
                edges.append([side,known,m.wave,len(deg),int((deg<=5).sum()),int((deg>0).sum())])
                expected=0.
                for seed in range(256):
                    order,priority=R.refine(m,seed)
                    ref=synchronous(order,priority,m.acceptable)
                    assert np.array_equal(ref,R.allocate(order,priority,m.acceptable))
                    checked+=1
                    winners=np.flatnonzero(ref>=0)
                    expected+=np.minimum(1,5/deg[winners]).mean()/256
                    if seed in (0,42,99,180,255):
                        for k in (1,3,5):
                            graph=R.slates(scores[m.wave],pool,k)&m.acceptable
                            assert np.array_equal(synchronous(order,priority,graph),R.allocate(order,priority,graph))
                            checked+=1
                saved=counts.query('side==@side and known==@known and k==5 and policy=="uniform" and wave==@m.wave')
                baseline_errors.append(abs(100*expected-saved.recovery.item()))
    info={'independent_assignment_vector_checks':checked,
          'max_abs_uniform_recovery_monte_carlo_error_by_wave_pp':float(max(baseline_errors)),
          'protocol_sha256':hashlib.sha256(Path(__file__).with_name('PROTOCOL.md').read_bytes()).hexdigest()}
    (a.out/'verification.json').write_text(json.dumps(info,indent=2)+'\n')
    pd.DataFrame(edges,columns=['side','known','wave','proposers','degree_le_5','degree_positive']).to_csv(a.out/'degrees.csv',index=False)
    print(info)
if __name__=='__main__':main()
