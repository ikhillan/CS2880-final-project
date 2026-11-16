"""Cross the two design choices in the calibrated rental simulator, with common input streams."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

CELLS = [(b,m) for b in ['Bronx','Brooklyn','Manhattan','Queens'] for m in ['2026-06','2026-07','2026-08']]
DESIGNS = ['ordinary listings, ordinary renters','ordinary listings, selected renters',
           'band listings, ordinary renters','band listings, selected renters']
METRICS = ['n_lower','uu_low','ut_low','n_higher','uu_high','ut_high']
SEEDS = list(range(84000,84040))


def rows_of_market(market, listing_mode, donor_mode):
    from hiddenoptions.exposure import transparent_exposure, uniform_exposure
    from hiddenoptions.matching import deferred_acceptance, blocking_mask
    low = market.renters.lower_income.to_numpy(bool)
    for quota, args in [('k5', {'k':5}), ('q40', {'q':0.4})]:
        base = uniform_exposure(market, **args)
        alternative = transparent_exposure(market, **args)
        mixed = np.where(low[:,None],base,alternative)
        assert np.array_equal(mixed.sum(1),base.sum(1)) and np.array_equal(base[low],mixed[low])
        matched = []
        for graph in [base,mixed]:
            applicants,listings = deferred_acceptance(market.inst,graph)
            assert not (blocking_mask(market.inst,applicants,listings) & graph).any()
            matched.append(applicants>=0)
        yield {'seed':market.spec.seed, 'borough':market.spec.borough, 'month':market.spec.month,
               'design':DESIGNS[2*listing_mode+donor_mode],'quota':quota,
               **dict(zip(METRICS,[low.sum(),matched[0][low].sum(),matched[1][low].sum(),
                                  (~low).sum(),matched[0][~low].sum(),matched[1][~low].sum()]))}


def simulate(repo):
    sys.path.insert(0,str(repo.resolve()))
    from hiddenoptions import markets as mod
    from hiddenoptions.data import borough_weights
    cat = mod.Catalog()
    records, missed = [], []
    def assemble(spec, catalog, renters, listings):
        # Reuse the complete ranking, priority and tie logic under the common seed.
        with patch.object(mod,'sample_people',return_value=(renters.reset_index(drop=True), listings.reset_index(drop=True), {})):
            return mod.build_market(spec,catalog)
    for borough,month in CELLS:
        for seed in SEEDS:
            spec=mod.MarketSpec(borough,month,seed,64,64,donors='calibration',design='uniform')
            frame,pool = cat.frame(spec)
            cap=mod._affordability_cap(frame,spec)
            band = pool.rent <= cap
            for supply in (0,1):
                lr=mod._rng(spec,'listings')
                if supply:
                    if min(int(band.sum()),int((~band).sum()))<32:
                        raise ValueError('Band sampling unavailable in a planned four-borough cell.')
                    listings=pd.concat([mod._draw_listings(lr,pool[band],32,spec),mod._draw_listings(lr,pool[~band],32,spec)])
                else:
                    listings=mod._draw_listings(lr,pool,64,spec)
                for selected in (0,1):
                    rr=mod._rng(spec,'renters')
                    if selected:
                        donors=frame[frame.lower_income]
                        any_feasible=mod._feasible(donors,listings,'rent').any(axis=1)
                        if not any_feasible.any():
                            missed.append([seed,borough,month,DESIGNS[2*supply+selected]])
                            continue
                        renters=pd.concat([mod._draw_renters(rr,donors[any_feasible],32),mod._draw_renters(rr,frame[~frame.lower_income],32)])
                    else:
                        renters=mod._draw_renters(rr,frame,64)
                    market=assemble(spec,cat,renters,listings)
                    if seed==SEEDS[0] and not supply and not selected:
                        native=mod.build_market(spec,cat)
                        for key in ['rank_b','rank_s']:
                            assert np.array_equal(getattr(native.inst,key),getattr(market.inst,key))
                    records.extend(rows_of_market(market,supply,selected))
    return pd.DataFrame(records), missed, borough_weights(cat.renters)


def summarize(frame,shares):
    indexed=frame.set_index(['seed','borough','month','design','quota'])[METRICS]
    index=pd.MultiIndex.from_tuples([(seed,b,m,d,q) for seed in SEEDS for b,m in CELLS
                                     for d in DESIGNS for q in ['k5','q40']],names=indexed.index.names)
    cube=indexed.reindex(index).to_numpy().reshape(40,12,4,2,6)
    common=np.isfinite(cube).all(axis=(2,3,4))
    weights=np.array([shares[b] for b,_ in CELLS])
    weights/=weights.sum()
    freqs=np.vstack([np.ones(40),np.random.default_rng(812649).multinomial(40,[1/40]*40,size=5000)])
    reports={}
    for target,values in [('available',cube),('joint',np.where(common[:,:,None,None,None],cube,np.nan))]:
        tab=np.einsum('rs,scdqm->rcdqm',freqs,np.nan_to_num(values))
        with np.errstate(divide='ignore',invalid='ignore'):
            effect=(tab[...,2]-tab[...,1])/tab[...,0]
        estimates=100*np.einsum('c,rcdq->rdq',weights,effect)
        def stat(v):
            valid=np.isfinite(v[1:])
            return {'estimate_pp':float(v[0]) if np.isfinite(v[0]) else None,
                    'ci95':np.quantile(v[1:],[.025,.975]).tolist() if valid.all() else None,
                    'undefined_draws':int((~valid).sum())}
        reports[target]={q:{d:stat(estimates[:,i,j]) for i,d in enumerate(DESIGNS)} for j,q in enumerate(['k5','q40'])}
        if target=='joint':
            for j,q in enumerate(['k5','q40']):
                reports[target][q]['listing_shift_at_ordinary_renters']=stat(estimates[:,2,j]-estimates[:,0,j])
                reports[target][q]['listing_shift_at_selected_renters']=stat(estimates[:,3,j]-estimates[:,1,j])
                reports[target][q]['renter_shift_on_band_listings']=stat(estimates[:,3,j]-estimates[:,2,j])
    reports['nmarkets_joint']=int(common.sum())
    reports['joint_by_cell']={f'{b}_{m}':int(common[:,i].sum()) for i,(b,m) in enumerate(CELLS)}
    return reports


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    args=p.parse_args()
    data,missing,shares=simulate(args.repo)
    args.out.mkdir(exist_ok=True,parents=True)
    data.to_csv(args.out/'block_counts.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    report={'design_sha256':hashlib.sha256(Path(__file__).with_name('design.md').read_bytes()).hexdigest(),
            'estimand':'UT minus UU, lower-income match rate in percentage points; four-borough standardized',
            'markets_by_design':data[data.quota=='k5'].groupby('design').size().to_dict(),
            'missing_count':len(missing),'missing_instances':missing,'outcomes':summarize(data,shares)}
    (args.out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
