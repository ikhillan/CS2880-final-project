"""Maximum cardinality in the displayed graph with responsive seat capacities."""
import argparse
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow,breadth_first_order
from run import Market,Kernel,history,features,ordering

def visible_capacity(m,cap,mask):
    # Source -> applicants -> programs -> sink. A unit of flow is one placement.
    i=m.a[mask]+1;j=m.school[mask]+m.n+1;t=m.n+m.m+1
    rows=np.concatenate([np.zeros(m.n,int),i,np.arange(m.m)+m.n+1])
    cols=np.concatenate([np.arange(m.n)+1,j,np.full(m.m,t)])
    values=np.r_[np.ones(m.n+len(i),np.int32),cap]
    G=csr_matrix((values,(rows,cols)),shape=(t+1,t+1))
    flow=maximum_flow(G,0,t,method='dinic')
    residual=(G-flow.flow).tocsr();residual.eliminate_zeros()
    reached=breadth_first_order(residual,0,directed=True,return_predecessors=False)
    S=np.zeros(t+1,bool);S[reached]=True
    cut=int(values[S[rows]&~S[cols]].sum())
    assert not S[t] and cut==flow.flow_value
    return cut

def main(root,results,out):
    m=Market(root,2026);X,x,_=features(m,history(Market(root,2025)))
    mod=json.loads((results/'metrics.json').read_text())['fitted']
    w=np.array(mod['weights_intercept_last']); y=(X-np.array(mod['feature_mean']))@(w[:-1]/np.array(mod['feature_sd']))+w[-1]
    del X
    candidates={'trained':y,'own_rank':-m.r,'historical_cutoff':(x>=0).astype(float)*40-m.r}
    rows=[]
    with tempfile.TemporaryDirectory() as tmp:
        kernel=Kernel(tmp)
        for j in range(-3,16):
            name=list(candidates)[j+3] if j<0 else 'uniform'
            scores=candidates[name] if j<0 else np.random.default_rng(14012026+j).random(len(m.a))
            rank=ordering(m,scores)
            da=kernel.solve(m,m.cap,rank,5);n=int((da>=0).sum()); nu=visible_capacity(m,m.cap,rank<5)
            assert nu>=n
            rows.append(dict(policy=name,draw=j if j>=0 else -1,da=n,capacity=nu,shortfall=nu-n))
    pd.DataFrame(rows).to_csv(out,index=False)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('inputs',type=Path);p.add_argument('results',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();main(a.inputs,a.results,a.out)
