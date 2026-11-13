"""Maximize applicants placed while preserving each provisional recipient's ordinal floor.

For an initial matching M, construct its residual network with source->matched
arcs unavailable in the forward direction. Every augmentation keeps the
source arc for each current recipient. Its value is certified by an equal cut.
"""
from pathlib import Path
import argparse
import json
import sys
import tempfile
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow, breadth_first_order
from repair import clear, learned

def ordinal_bound(m,cap,start):
    old=start>=0;oldedge=start[old]
    new=np.flatnonzero((~old[m.a])|(np.arange(len(m.a))<=start[m.a]))
    # Residual network of known matching with full unit applicant-program edges.
    isused=np.zeros(len(m.a),bool);isused[oldedge]=True
    fwd=new[~isused[new]];n=m.n;p=m.n+1;sink=m.n+m.m+1
    used=np.bincount(m.school[oldedge],minlength=m.m);js=np.arange(m.m);un=np.flatnonzero(~old)
    rows=np.r_[np.zeros(len(un),int),1+m.a[fwd],p+m.school[oldedge],p+js]
    cols=np.r_[1+un,p+m.school[fwd],1+m.a[oldedge],np.repeat(sink,m.m)]
    values=np.r_[np.ones(len(un)+len(fwd)+len(oldedge),int),cap-used]
    graph=csr_matrix((values,(rows,cols)),shape=(sink+1,sink+1));graph.eliminate_zeros()
    flow=maximum_flow(graph,0,sink,method='dinic');diff=flow.flow
    resid=(graph-diff).tocsr();resid.eliminate_zeros()
    reached=np.zeros(sink+1,bool);reached[breadth_first_order(resid,0,directed=True,return_predecessors=False)]=True
    cut=int(values[reached[rows]&~reached[cols]].sum());assert cut==flow.flow_value
    keeps=np.asarray(diff[p+m.school[oldedge],1+m.a[oldedge]]).ravel()==0
    adds=np.asarray(diff[1+m.a[fwd],p+m.school[fwd]]).ravel()>0
    chosen=np.r_[oldedge[keeps],fwd[adds]];final=np.full(m.n,-1,np.int32)
    final[m.a[chosen]]=chosen
    assert np.unique(m.a[chosen]).size==len(chosen)
    assert np.all(final[old]>=0) and np.all(final[old]<=start[old])
    assert np.all(np.bincount(m.school[chosen],minlength=m.m)<=cap)
    assert len(chosen)==old.sum()+cut
    return dict(gain_upper=cut,retained_initial=int(old.sum()),moved_to_strictly_better=int((final[old]<start[old]).sum()),
                occupied=int(len(chosen)),eligible_edges=int(len(new)))

def main(a):
    sys.path.insert(0,str(a.university.resolve()))
    from run import Market,Kernel,features,history,ordering
    m=Market(a.inputs,2026,positive_scores=True);prev=Market(a.inputs,2025,positive_scores=True)
    _,x,_=features(m,history(prev))
    ranks={'first':m.pos,'cutoff':ordering(m,40.*(x>=0)-m.r),'blind':learned(m,a.features,a.fit,'blind'),
           'informed':learned(m,a.features,a.fit,'informed')}
    out=[]
    with tempfile.TemporaryDirectory() as t:
        kernel=Kernel(t)
        for name,rank in ranks.items():
            initial=clear(m,kernel,m.cap,rank<5)
            out.append(dict(baseline=name,**ordinal_bound(m,m.cap,initial)))
    pd.DataFrame(out).to_csv(a.out/'upper.csv',index=False)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['university','inputs','features','fit','out']:p.add_argument(n,type=Path)
    main(p.parse_args())
