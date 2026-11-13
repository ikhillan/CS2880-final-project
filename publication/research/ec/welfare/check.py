"""Exact burden floors under stability, matched-participant and placement-count restrictions."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching, maximum_flow

CELLS = [(b, m) for b in ('Manhattan', 'Brooklyn', 'Queens', 'Bronx')
         for m in ('2026-06', '2026-07', '2026-08')]


@dataclass
class Market:
    key: tuple
    inst: object
    low: np.ndarray
    burden: np.ndarray
    da: np.ndarray
    held: np.ndarray
    stable: list


def inverse(match, n):
    held = np.full(n, -1, int)
    b = np.flatnonzero(match >= 0)
    assert len(np.unique(match[b])) == len(b)
    held[match[b]] = b
    return held


def stable_options(inst, da, dh, other, oh):
    """Complete enumeration with fixed recipients, listings and both sides' stable rank intervals."""
    from hiddenoptions.matching import blocking_mask
    assert np.array_equal(da >= 0, other >= 0) and np.array_equal(dh >= 0, oh >= 0)
    v = np.flatnonzero(da != other)
    available = np.flatnonzero(dh != oh)
    assert len(v) == len(available)
    options = {}
    for b in v:
        options[b] = [s for s in available if inst.E[b, s]
                      and inst.rank_b[b, da[b]] <= inst.rank_b[b, s] <= inst.rank_b[b, other[b]]
                      and inst.rank_s[oh[s], s] <= inst.rank_s[b, s] <= inst.rank_s[dh[s], s]]
    order = sorted(v, key=lambda b: len(options[b]))
    current = da.copy()
    result = []
    def visit(j, used):
        if j == len(order):
            held = inverse(current, inst.n_s)
            if not blocking_mask(inst, current, held).any():
                result.append(current.copy())
            return
        b = order[j]
        for s in options[b]:
            if s not in used:
                current[b] = s
                visit(j+1, used | {s})
    visit(0, set())
    assert any(np.array_equal(da, a) for a in result) and any(np.array_equal(other, a) for a in result)
    return result


def extract(repo):
    sys.path.insert(0,str(repo.resolve()))
    from hiddenoptions.markets import Catalog, MarketSpec, build_market
    from hiddenoptions.matching import Instance, deferred_acceptance
    cat = Catalog()
    rows = []
    for b,m in CELLS:
        for seed in range(20):
            spec=MarketSpec(b,m,seed,64,64,donors='calibration',w_p=4.,sigma_r=.5)
            market = build_market(spec,cat)
            inst = market.inst
            da,held = deferred_acceptance(inst,inst.E)
            rev = Instance(inst.rank_s.T,inst.rank_b.T,inst.firm_s,inst.firm_b,inst.cost)
            oh,other = deferred_acceptance(rev,rev.E)
            stable = stable_options(inst,da,held,other,oh)
            low = market.renters.lower_income.to_numpy(bool)
            burden = np.full(inst.E.shape,np.inf)
            income = market.renters.income.to_numpy(float)[:,None]
            np.divide(12*market.listings.rent.to_numpy()[None,:],income,out=burden,where=inst.E)
            rows.append(Market((b,m,seed),inst,low,burden,da,held,stable))
    return rows


def low_burdens(m,a):
    ix=np.flatnonzero(m.low & (a>=0))
    return m.burden[ix,a[ix]]


def at_scope(m,t,scope):
    """Return the feasible assignment maximizing the cheap lower-income count in this market."""
    E=m.inst.E
    if scope=='stable_min' or scope=='stable_max':
        f=max if scope=='stable_min' else min
        a=f(m.stable,key=lambda a: np.count_nonzero(low_burdens(m,a)<=t))
        return a
    if scope in ('both_renter_ids','both_sides_ids'):
        active=np.flatnonzero(m.da>=0)
        offered=np.flatnonzero(m.held>=0) if scope=='both_sides_ids' else np.arange(E.shape[1])
        visible=E[np.ix_(active,offered)]
        cheap=(m.low[active,None] & (m.burden[np.ix_(active,offered)]<=t))
        cost=np.where(visible,~cheap,np.inf)
        b,s=linear_sum_assignment(cost)
        assert len(b)==len(active)
        out=np.full(E.shape[0],-1,int)
        out[active[b]]=offered[s]
        return out
    assert scope=='both_group_market_counts'
    n,mcols=E.shape
    nlow=int(np.count_nonzero(m.low&(m.da<0)))
    nhigh=int(np.count_nonzero(~m.low&(m.da<0)))
    # Each private group of dummy vacancies absorbs its exact number of unplaced renters.
    cheap=m.low[:,None] & (m.burden<=t)
    real=np.where(E,(n+1)+(~cheap),np.inf)
    low_dummy=np.broadcast_to(np.where(m.low,0.,np.inf)[:,None],(n,nlow))
    high_dummy=np.broadcast_to(np.where(~m.low,0.,np.inf)[:,None],(n,nhigh))
    b,s=linear_sum_assignment(np.hstack([real,low_dummy,high_dummy]))
    out=np.where(s<mcols,s,-1)
    assert np.array_equal(b,np.arange(n))
    assert np.count_nonzero(m.low & (out>=0))==np.count_nonzero(m.low & (m.da>=0))
    assert np.count_nonzero(~m.low & (out>=0))==np.count_nonzero(~m.low & (m.da>=0))
    return out


def qfloor(markets,scope,N,k):
    # Full feasible edges for lower-income occupants are a complete set of possible breakpoints.
    cuts=np.unique(np.concatenate([m.burden[m.inst.E & m.low[:,None]] for m in markets]))
    def evaluate(t):
        assignments=[at_scope(m,t,scope) for m in markets]
        count=sum(np.count_nonzero(low_burdens(m,a)<=t) for m,a in zip(markets,assignments))
        return int(count),assignments
    lo,hi=-1,len(cuts)-1
    while hi-lo>1:
        mid=(lo+hi)//2
        if evaluate(cuts[mid])[0]>=k:
            hi=mid
        else:
            lo=mid
    opt=float(cuts[hi]); cut0=np.nextafter(opt,-np.inf)
    nminus,am=evaluate(cut0)
    nplus,ap=evaluate(opt)
    witness=am if scope=='stable_max' else ap
    vals=np.concatenate([low_burdens(m,a) for m,a in zip(markets,witness)])
    assert len(vals)==N and nminus<k<=nplus and np.quantile(vals,.25)==opt
    return dict(q25=opt,strictly_below=nminus,at_value=nplus), witness


def pooled_feasibility(markets,N,NH,k,opt):
    """Extend a minimum-burden matching with a residual two-group max flow at the relaxed lower bound."""
    # Preserve every low-cost pair to prove attainability after exactly meeting both volumes.
    fixed=[]
    for m in markets:
        mask=m.inst.E & m.low[:,None] & (m.burden<=opt)
        a=maximum_bipartite_matching(csr_matrix(mask),perm_type='column')
        fixed.append([(int(b),int(s)) for b,s in enumerate(a) if s>=0])
    assert sum(map(len,fixed))==k
    # Vertices 0 source, 1/2 income groups, 3 sink; remaining disjoint renter/listing pairs.
    r=[];c=[];v=[]
    def arcs(row,col,val):
        r.extend(row);c.extend(col);v.extend(val)
    arcs([0,0],[1,2],[N-k,NH])
    base=4
    offsets=[]
    for m,pairs in zip(markets,fixed):
        nb,ns=m.inst.E.shape
        br=np.arange(nb)+base
        sr=np.arange(ns)+base+nb
        offsets.append((br,sr))
        usedb=np.zeros(nb,bool);useds=np.zeros(ns,bool)
        for b,s in pairs: usedb[b]=True;useds[s]=True
        bs,ss=np.where(m.inst.E & ~usedb[:,None] & ~useds[None,:])
        arcs(np.where(m.low[~usedb],1,2),br[~usedb],np.ones((~usedb).sum(),int))
        arcs(br[bs],sr[ss],np.ones(len(bs),int))
        arcs(sr[~useds],np.full((~useds).sum(),3),np.ones((~useds).sum(),int))
        base+=nb+ns
    network=coo_matrix((np.array(v,int),(r,c)),shape=(base,base)).tocsr()
    f=maximum_flow(network,0,3)
    assert f.flow_value==N-k+NH, (f.flow_value,N-k+NH)
    assignments=[]
    for m,pairs,(br,sr) in zip(markets,fixed,offsets):
        bs,ss=f.flow[br][:,sr].nonzero()
        out=np.full(len(br),-1,int);out[bs]=ss
        for b,s in pairs: out[b]=s
        assignments.append(out)
    assert sum(np.count_nonzero(~m.low & (a>=0)) for m,a in zip(markets,assignments))==NH
    vals=np.concatenate([low_burdens(m,a) for m,a in zip(markets,assignments)])
    assert len(vals)==N and np.quantile(vals,.25)==opt
    return assignments


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    args=p.parse_args()
    m=extract(args.repo)
    import scipy
    N=sum(np.count_nonzero(z.low&(z.da>=0)) for z in m)
    NH=sum(np.count_nonzero(~z.low&(z.da>=0)) for z in m)
    k=(N-1)//4+1
    assert (N,NH,k)==(217,7755,55)
    results={'markets':len(m),'n_low':N,'n_higher':NH,'k':k,'da_q25':float(np.quantile(np.concatenate([low_burdens(z,z.da) for z in m]),.25)),
             'enumerated_stable_outcomes_by_market':[len(z.stable) for z in m],
             'variable_stable_markets':sum(len(z.stable)>1 for z in m),
             'total_low_recipients_with_multiple_stable_partners':sum(int(np.any(np.stack(z.stable)!=z.da,axis=0)[z.low].sum()) for z in m),
             'low_recipients_with_multiple_stable_partners_markets':sum(bool(np.any(np.stack(z.stable)[:,z.low]!=z.da[z.low])) for z in m),
             'scopes':{},'scipy':scipy.__version__}
    witnesses={'market_order':[z.key for z in m],'recipient_order':'64 entries per market. Entries are matched listing (0-based) or -1 for unmatched.', 'scopes':{}}
    for scope in ['stable_min','stable_max','both_sides_ids','both_renter_ids','both_group_market_counts']:
        summary, a=qfloor(m,scope,N,k)
        results['scopes'][scope]=summary
        witnesses['scopes'][scope]=[x.tolist() for x in a]
    cuts=np.unique(np.concatenate([z.burden[z.inst.E & z.low[:,None]] for z in m]))
    def upper(t):
        return int(sum(np.count_nonzero(maximum_bipartite_matching(csr_matrix(z.inst.E & z.low[:,None] & (z.burden<=t)),perm_type='column')>=0) for z in m))
    lo,hi=-1,len(cuts)-1
    while hi-lo>1:
        mid=(hi+lo)//2
        if upper(cuts[mid])>=k:hi=mid
        else:lo=mid
    opt=float(cuts[hi]);nminus=upper(np.nextafter(opt,-np.inf));nplus=upper(opt)
    w=pooled_feasibility(m,N,NH,k,opt)
    results['scopes']['both_group_pooled_counts']={'q25':opt,'upper_bound_strictly_below':nminus,'at_value':nplus}
    witnesses['scopes']['both_group_pooled_counts']=[x.tolist() for x in w]
    results['market_keys']=witnesses['market_order']
    args.out.mkdir(exist_ok=True,parents=True)
    args.out.joinpath('results.json').write_text(json.dumps(results,indent=2)+'\n')
    import gzip
    with open(args.out/'witnesses.json.gz','wb') as f:
        with gzip.GzipFile(filename='',fileobj=f,mode='wb',mtime=0) as gz:
            gz.write(json.dumps(witnesses,separators=(',',':')).encode())
    print(json.dumps({k:v for k,v in results.items() if k not in ['market_keys','enumerated_stable_outcomes_by_market']},indent=2))

if __name__=='__main__':
    main()
