"""Independently solve the fixed-volume threshold problem on a unit-capacity flow network."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from nested_support import extract


def expensive_edge_cost(markets,threshold,N,scope):
    vals=[]
    offsets=0
    for m in markets:
        keep = m.initial_matched[m.renters] if scope=='original_recipients' else np.ones(len(m.renters),bool)
        b,s=m.renters[keep],m.slots[keep]
        how_many=int(m.initial_matched.sum())
        if not len(b):
            assert scope=='pooled' or how_many==0
            continue
        # Compact unit capacity rows on both sides.
        br,bi=np.unique(b,return_inverse=True)
        sj,si=np.unique(s,return_inverse=True)
        degree_rows=len(br)
        vals.append((bi+offsets,si+offsets+degree_rows,(m.burden[keep]>threshold).astype(float),how_many))
        offsets+=len(br)+len(sj)
    edge_cost=np.concatenate([v[2] for v in vals])
    ii=np.concatenate([np.concatenate([v[0] for v in vals]),np.concatenate([v[1] for v in vals])])
    indices=np.arange(len(edge_cost))
    incidence=coo_matrix((np.ones(len(ii)),(ii,np.concatenate([indices,indices]))),shape=(offsets,len(edge_cost))).tocsr()
    if scope=='pooled':
        e_rows=np.zeros(len(edge_cost),int)
        demand=[N]
    else:
        e_rows=np.repeat(np.arange(len(vals)),[len(v[2]) for v in vals])
        demand=[v[3] for v in vals]
        assert sum(demand)==N
    A_eq=coo_matrix((np.ones(len(edge_cost)),(e_rows,indices)),shape=(len(demand),len(edge_cost))).tocsr()
    sol=linprog(edge_cost,A_ub=incidence,b_ub=np.ones(offsets),A_eq=A_eq,b_eq=demand,bounds=(0,1),method='highs')
    if not sol.success:
        raise RuntimeError(sol.message)
    v=np.rint(sol.x)
    # Cardinality-constrained bipartite matching is integral via a min-cost unit flow.
    assert np.max(np.abs(sol.x-v))<1e-6
    assert (incidence@v<=1).all() and (A_eq@v==demand).all()
    return int(v.sum()-np.dot(v,edge_cost))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--calibration',type=Path,required=True)
    ap.add_argument('--evidence',type=Path,required=True)
    args=ap.parse_args()
    saved=json.loads((args.evidence/'results.json').read_text())
    calibration=json.loads(args.calibration.read_text())
    data=extract(args.repo,calibration['selected'])
    tau=saved['target_burden']
    out=[]
    for row in saved['sizes']:
        if row['size'] not in [64,'whole']:
            continue
        scoped = [m.select(row['size'],row['price_floor']) for m in data]
        for scope in (['pooled'] if row['size']==64 else ['pooled','same_market_counts','original_recipients']):
            local=row if scope=='pooled' else row[scope]
            q=local['exact_minimum_q25']
            values={}
            for key,t in [('survey_target',tau),('strictly_below',np.nextafter(q,-np.inf)),('at_optimum',q)]:
                values[key]=expensive_edge_cost(scoped,t,217,scope)
            assert values['strictly_below']<55<=values['at_optimum']
            out.append(dict(size=row['size'],price_floor=row['price_floor'],scope=scope,
                            fixed_volume=217,max_cheap_edges=values))
            print(out[-1],flush=True)
    (args.evidence/'flow_check.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':
    main()
