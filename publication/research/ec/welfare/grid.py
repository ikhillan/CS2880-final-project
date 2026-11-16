"""Exact interpolated quantiles from products of distinct lower-income stable outcomes."""
import argparse
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
from check import CELLS, stable_options


def quantile_from_blocks(fixed, variable, N):
    # Equality is on the sorted burden vectors: local allocations changing solely
    # a higher-group partner (or permuting an equal burden) contribute one vector.
    fixed=np.sort(np.array(fixed))
    rank=(N-1)//4
    frac=((N-1)%4)/4
    total=1
    for choices in variable: total*=len(choices)
    if total>2_000_000: raise RuntimeError('Product exceeds safe enumeration bound: '+str(total))
    best=(np.inf,None);worst=(-np.inf,None)
    for indices in product(*[range(len(v)) for v in variable]):
        additions=[v[i] for v,i in zip(variable,indices)]
        vals=np.sort(np.concatenate([fixed,*additions]))
        assert len(vals)==N
        val=vals[rank] if not frac else vals[rank]+frac*(vals[rank+1]-vals[rank])
        if val<best[0]:best=(val,indices)
        if val>worst[0]:worst=(val,indices)
    return best,worst,total


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    sys.path.insert(0,str(args.repo.resolve()))
    from hiddenoptions.markets import Catalog, MarketSpec, build_market
    from hiddenoptions.matching import Instance, deferred_acceptance
    cat=Catalog()
    report=[]
    calibrated=json.loads(args.calibration.read_text())
    for price,noise in product([1.,2.,4.],[.5,.75,1.25]):
        fixed,variable,observed,groups=[],[],[],[]
        nflex,mflex,nstable=0,0,0
        for b,month in CELLS:
            for seed in range(20):
                market=build_market(MarketSpec(b,month,seed,64,64,donors='calibration',w_p=price,sigma_r=noise),cat)
                inst=market.inst;r=market.renters
                p,ph=deferred_acceptance(inst,inst.E)
                rev=Instance(inst.rank_s.T,inst.rank_b.T,inst.firm_s,inst.firm_b,inst.cost)
                sh,s=deferred_acceptance(rev,rev.E)
                low=np.flatnonzero(r.lower_income.to_numpy(bool) & (p>=0))
                variance=np.count_nonzero(p[low]!=s[low])
                nflex+=int(variance);mflex+=int(variance>0)
                prices=market.listings.rent.to_numpy(float)
                income=r.income.to_numpy(float)
                current=12*prices[p[low]]/income[low]
                observed.extend(current.tolist())
                if variance:
                    opts=stable_options(inst,p,ph,s,sh)
                    unique={}
                    for a in opts:
                        key=tuple(np.sort(12*prices[a[low]]/income[low]).tolist())
                        if key not in unique:unique[key]=a.tolist()
                    if len(unique)>1:
                        variable.append([np.array(v) for v in unique])
                        groups.append(dict(market=[b,month,seed], assignments=list(unique.values())))
                    else:fixed.extend(current.tolist())
                else:fixed.extend(current.tolist())
                nstable+=int(np.any(p!=s))
        n=len(observed);best,worst,nout=quantile_from_blocks(fixed,variable,n)
        fitted=next(row for row in calibrated['fits'] if row['w_p']==price and row['sigma_r']==noise)
        da=float(np.quantile(observed,.25)); assert n==fitted['matched_lower'] and abs(da-fitted['simulated']['lower'][0])<1e-13
        assert best[0]-1e-14<=da<=worst[0]+1e-14
        row=dict(price=price,noise=noise,low_placed=n,DA_q25=da,stable_min_q25=float(best[0]),stable_max_q25=float(worst[0]),width_pp=100*float(worst[0]-best[0]),
                 markets_with_multiple_stable_outcomes=nstable,flexible_low_renters=nflex,markets_with_flexible_low=mflex,
                 combinations_of_burden_outcomes=nout)
        for label,extreme in [('min',best),('max',worst)]:
            row[label+'_nonconstant_markets_witness']=[dict(market=g['market'],assignment=g['assignments'][i]) for g,i in zip(groups,extreme[1])]
        report.append(row)
        print({k:v for k,v in row.items() if not k.endswith('witness')},flush=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
