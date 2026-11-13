"""Build split-isolated, integer-indexed simulator markets without identifiers."""
import argparse
from collections import Counter
import hashlib
import importlib.util
from itertools import product
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np

SPLITS = {'train': (2026120100,80), 'val': (2026120300,16),
          'summer': (2026120500,48), 'winter': (2026120700,48), 'cal': (2026120900,32)}
BOROUGHS = ['Bronx','Brooklyn','Manhattan','Queens']
DESIGNS = ['uniform','enriched']


def build(task):
    seed,design,month,borough=task
    from hiddenoptions import markets, matching, exposure
    spec=markets.MarketSpec(borough,month,seed,donors='calibration',design=design)
    try:
        market=markets.build_market(spec,catalog)
    except markets.InsufficientSupply as error:
        return task,str(error)
    E=market.inst.E
    assert (market.renters.donors=='calibration').all()
    cap=market.renters.cap.to_numpy(float)[:,None]
    rent=market.listings[spec.rent_column].to_numpy(float)[None,:]
    inc=market.renters.income.to_numpy(float)[:,None]
    beds=market.listings.bedrooms.to_numpy()[None,:]
    need=market.renters.need_bedrooms.to_numpy()[:,None]
    q=markets.listing_quality(market.listings,catalog,spec.quality)[None,:]
    frac=np.divide(rent,cap,out=np.zeros(E.shape),where=cap>0)
    ratio=np.minimum(np.divide(inc,12*rent,out=np.zeros(E.shape),where=rent>0),markets.SCREEN_RATIO)
    features=np.stack(np.broadcast_arrays(frac,np.minimum(beds-need,2),market.distance_km/5,q,ratio),-1)
    features=np.where(E[...,None],features,0).astype(np.float32)
    u=np.where(E,market.u_observable,0)
    v=np.where(E,spec.w_income*ratio,0)
    assert np.isfinite(features).all() and np.isfinite(u).all() and np.isfinite(v).all()
    uniform=np.full(E.shape,127,np.uint8)
    for r in range(len(E)):
        order=exposure.fallback_order(market,r)
        uniform[r,order]=np.arange(len(order))
    transparent=np.where(E,np.argsort(np.argsort(-market.u_observable,axis=1,kind='stable'),axis=1),127).astype(np.uint8)
    y,ys=matching.deferred_acceptance(market.inst,E)
    assert not matching.blocking_mask(market.inst,y,ys).any()
    return task,dict(x=features,e=E,u=u.astype(np.float32),v=v.astype(np.float32),
        pref=np.where(E,market.inst.rank_b,127).astype(np.uint8),seller=np.where(E,market.inst.rank_s,127).astype(np.uint8),
        uniform=uniform,transparent=transparent,lower=market.renters.lower_income.to_numpy(bool),y=np.where(y<0,64,y).astype(np.uint8))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--split',choices=list(SPLITS),nargs='+',default=list(SPLITS));p.add_argument('--workers',type=int,default=5);
    p.add_argument('--repo',type=Path);p.add_argument('--winter-script',type=Path);p.add_argument('--summer-raw',type=Path);p.add_argument('--winter-raw',type=Path);a=p.parse_args()
    root=a.root.resolve(); repo=(a.repo or root/'work/todd-project').resolve(); sys.path[:0]=[str(repo),str(repo/'scripts')]
    from hiddenoptions.markets import Catalog
    from hiddenoptions.data import borough_weights
    global catalog
    catalog=Catalog();weights=borough_weights(catalog.renters)
    wpath=(a.winter_script or root/'outputs/ec-revision/winter/run.py').resolve()
    wm=importlib.util.spec_from_file_location('winter_clean',wpath);winter=importlib.util.module_from_spec(wm);wm.loader.exec_module(winter)
    a.out.mkdir(exist_ok=True,parents=True)
    for split in a.split:
        months=winter.MONTHS if split=='winter' else winter.SUMMER
        if split=='winter':
            winter.prepare(catalog,repo,(a.summer_raw or root/'work/nyc-scarcity/data/firstmover').resolve(),(a.winter_raw or root/'work/nyc-scarcity/winter-inputs').resolve())
        base,n=SPLITS[split];keys=[]; data={};missing=[]
        tasks=list(product(range(base,base+n),DESIGNS,months,BOROUGHS))
        with mp.get_context('fork').Pool(a.workers) as pool:
            for task,rec in pool.imap(build,tasks,chunksize=4):
                if isinstance(rec,str):
                    missing.append([*task,rec]);continue
                index=len(keys);keys.append(task)
                for name,arr in rec.items():
                    if name not in data: data[name]=np.empty((len(tasks),*arr.shape),arr.dtype)
                    data[name][index]=arr
        target=a.out/f'{split}.npz';np.savez_compressed(target,**{k:v[:len(keys)] for k,v in data.items()})
        meta=dict(keys=keys,key_order=['seed','design','month','borough'],missing=missing,weights={b:weights[b] for b in BOROUGHS},
                  feature_order=['rent_over_cap','spare_bedrooms_capped','distance_over_5km','quality','capped_income_over_annual_rent'],
                  seed_start=base,blocks=n,sha256=hashlib.file_digest(target.open('rb'),'sha256').hexdigest())
        target.with_suffix('.json').write_text(json.dumps(meta,separators=(',',':'))+'\n');print(split,len(keys),len(missing),target.stat().st_size,flush=True)


if __name__=='__main__': main()
