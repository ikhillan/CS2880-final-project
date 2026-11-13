"""Prepare anonymized integer ranks and shortlist orders using the team's development generator."""
import argparse
import hashlib
import json
import sys
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SEED_START = 2026111300
BLOCKS = 100
BOROUGHS = ['Bronx','Brooklyn','Manhattan','Queens']
MONTHS = ['2026-06','2026-07','2026-08']
PRICES = [1.,2.,4.]
NOISES = [.5,.75,1.25]
N = 64
SENTINEL = 127
DESIGNS = ['uniform','enriched']


def order_rank(scores, eligible):
    order = np.argsort(-np.where(eligible, scores, -np.inf), kind='stable', axis=1)
    return np.where(eligible, np.argsort(order, axis=1), SENTINEL).astype(np.uint8)


def initialize(repo):
    global catalog, markets, exposure, matching
    sys.path.insert(0,str(repo))
    from hiddenoptions import markets, exposure, matching
    catalog = markets.Catalog()


def sample(task):
    index, block, design, month, borough = task
    spec = markets.MarketSpec(borough, month, SEED_START+block, 64,64,design=design,donors='calibration')
    market = markets.build_market(spec,catalog)
    # sample_people() supplies the label after resampling from the selected donor frame.
    assert set(market.renters.donors)=={'calibration'}
    E, rb, rs = market.inst.E, market.inst.rank_b, market.inst.rank_s
    low = market.renters.lower_income.to_numpy(bool)
    cap = market.renters.cap.to_numpy(float)[:,None]
    rent = market.listings.rent.to_numpy(float)[None,:]
    beds = market.listings.bedrooms.to_numpy()[None,:]
    need = market.renters.need_bedrooms.to_numpy()[:,None]
    quality = markets.listing_quality(market.listings,catalog,spec.quality)[None,:]
    rng = markets._rng(spec,'preferences')
    xi = rng.standard_normal((1,64))
    eps = rng.standard_normal(E.shape)
    rng.standard_normal(E.shape)  # listing noise, already used in the reference instance
    tie = rng.random(E.shape) * 1e-9
    with np.errstate(divide='ignore', invalid='ignore'):
        price_share = np.where(cap>0,rent/cap,1.)
    ranks = np.empty((9,64,64),np.uint8)
    transparent = np.empty((3,64,64),np.uint8)
    uniform = np.full((64,64),SENTINEL,np.uint8)
    for b in range(64):
        order = exposure.fallback_order(market,b)
        uniform[b,order] = np.arange(len(order))
    references = np.full((9,2,3,64),SENTINEL,np.uint8) if block==0 else None
    for pi,price_weight in enumerate(PRICES):
        observed = spec.w_q*quality-price_weight*price_share+spec.w_s*np.minimum(beds-need,2)-spec.w_d*market.distance_km/5
        transparent[pi] = order_rank(observed,E)
        for ni,sigma in enumerate(NOISES):
            gp = pi*3+ni
            rentscore = observed+spec.w_q*spec.common_shock*xi+sigma*eps+tie
            ranks[gp] = order_rank(rentscore,E)
            if block==0:
                ref = markets.build_market(replace(spec,w_p=price_weight,sigma_r=sigma),catalog)
                assert np.array_equal(ref.inst.rank_b, np.where(E,ranks[gp].astype(int),matching.NOT_EDGE))
                assert np.array_equal(ref.inst.rank_s,rs)
                assert np.array_equal(ref.renters.income.to_numpy(),market.renters.income.to_numpy())
                for qi,(k,q) in enumerate([(5,None),(None,.4)]):
                    u = exposure.uniform_exposure(market,k,q)
                    t = exposure.transparent_exposure(ref,k,q)
                    c = exposure.show_count(E.sum(1),k,q)[:,None]
                    assert np.array_equal(u,uniform<c) and np.array_equal(t,transparent[pi]<c)
                    for ai,g in enumerate([u,np.where(low[:,None],u,t),t]):
                        matched,_ = matching.deferred_acceptance(ref.inst,g)
                        references[gp,qi,ai] = np.where(matched<0,64,matched)
    assert np.array_equal(ranks[6],np.where(E,rb,SENTINEL))
    return index, (ranks, np.where(E,rs,SENTINEL).astype(np.uint8), uniform, transparent, low),references


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--workers',default=4,type=int)
    a=p.parse_args()
    from itertools import product
    tasks = [(i,*vals) for i,vals in enumerate(product(range(BLOCKS), DESIGNS,MONTHS,BOROUGHS))]
    pref = np.empty((len(tasks),9,N,N),np.uint8)
    seller = np.empty((len(tasks),N,N),np.uint8)
    uniform = np.empty_like(seller)
    transparent = np.empty((len(tasks),3,N,N),np.uint8)
    low = np.empty((len(tasks),N),np.bool_)
    ref = np.empty((24,9,2,3,N), np.uint8)
    with Pool(a.workers, initializer=initialize,initargs=(a.repo.resolve(),)) as pool:
        for i,values,check in pool.imap_unordered(sample,tasks,chunksize=4):
            pref[i],seller[i],uniform[i],transparent[i],low[i] = values
            if check is not None:
                ref[i] = check
    np.savez_compressed(a.out,pref=pref,seller=seller,uniform=uniform,transparent=transparent,lower=low,reference=ref)
    sys.path.insert(0,str(a.repo.resolve()))
    from hiddenoptions.data import borough_weights,load_renters
    weights=borough_weights(load_renters())
    evidence=dict(blocks=BLOCKS,seed_start=SEED_START,shape=list(pref.shape),n_market=len(tasks),
        order=['block','design','month','borough'],designs=DESIGNS,months=MONTHS,boroughs=BOROUGHS,
        raw_borough_weights={b:weights[b] for b in BOROUGHS},prices=PRICES,noises=NOISES,
        reference_matched_vectors=len(ref)*9*2*3,
        sha256=hashlib.file_digest(a.out.open('rb'),'sha256').hexdigest(),numpy=np.__version__)
    print(json.dumps(evidence,indent=2))
    a.out.with_suffix('.json').write_text(json.dumps(evidence,indent=2)+'\n')


if __name__=='__main__':
    main()
