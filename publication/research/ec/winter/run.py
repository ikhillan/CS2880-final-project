"""Prospectively specified score-only transfer using a previously analyzed catalogue period."""
import argparse
from collections import Counter
from dataclasses import replace
from hashlib import file_digest
import json
from pathlib import Path
import platform
import re
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MONTHS = ['2026-01', '2026-02', '2026-03']
SUMMER = ['2026-06', '2026-07', '2026-08']
BOROUGHS = ['Bronx', 'Brooklyn', 'Manhattan', 'Queens']
DESIGNS = ['uniform','enriched']
SEED = 2026111700
NBLOCK = 80
MISSING = {'', 'NA', 'NONE', 'UNKNOWN', 'NULL'}


def digest(path):
    with open(path,'rb') as f:
        return file_digest(f,'sha256').hexdigest()


def normalized(value):
    return re.sub('[^A-Z0-9]', '', str(value).strip().upper()) if pd.notna(value) else ''


def physical(borough, street, unit):
    street, unit = normalized(street), normalized(unit)
    if street in MISSING or unit in MISSING or pd.isna(borough) or borough not in BOROUGHS:
        return None
    return (borough, street, unit)


def prepare(cat, repo, summer, winter):
    import firstmover_clean as cleaner
    from hiddenoptions import data
    pins = json.loads((HERE/'sources.json').read_text())
    snapshot = HERE/'BLS-snapshot.json'
    assert digest(snapshot)==pins[snapshot.name]['sha256']
    cpi_data=json.loads(snapshot.read_text())['Results']['series'][0]
    assert cpi_data['seriesID']=='CUUR0000SA0'
    cpi={r['year']+'-'+r['period'][1:]:float(r['value']) for r in cpi_data['data']}
    for month in MONTHS+['2026-07']:
        assert data.cpi_value(month)==(cpi[month], 'observed')
    summer_ids, summer_keys = set(), set()
    inputs={}
    for m in SUMMER:
        path=summer/f'{m}.csv'
        sha=digest(path)
        assert sha==json.loads((repo/f'data/firstmover/raw/{m}.manifest.json').read_text())['sha256']
        inputs[m]=sha
        raw=pd.read_csv(path,dtype=str,keep_default_na=False,usecols=['id','borough','street','unit'])
        for row in raw.itertuples():
            summer_ids.add('fm:'+row.id)
            key=physical(row.borough,row.street,row.unit)
            if key: summer_keys.add(key)
    seen_ids, seen_keys=set(),set()
    summaries={}
    cleaner.RAW_DIR=winter
    # Native post-CPI/order/PUMA loader expects a gzip CSV; only the temporary filtered winter files use this path.
    with tempfile.TemporaryDirectory() as td:
        original=data.FIRSTMOVER_CLEAN
        try:
            data.FIRSTMOVER_CLEAN=Path(td)
            for m in MONTHS:
                inputs[m]=digest(winter/f'{m}.csv')
                assert inputs[m]==pins[f'{m}.csv']['sha256']
                cleaned=cleaner.clean(m)
                selected=cleaned[cleaned.in_analysis_catalog & cleaned.borough.isin(BOROUGHS)]
                ledger=Counter()
                keep=[]
                for row in selected.itertuples():
                    key=physical(row.borough,row.street,row.unit)
                    source_key=physical(row.borough_source,row.street,row.unit)
                    if not key:
                        ledger['ambiguous_unit']+=1
                    elif row.listing_uid in summer_ids or key in summer_keys or source_key in summer_keys:
                        ledger['summer_overlap']+=1
                    elif row.listing_uid in seen_ids or key in seen_keys:
                        ledger['winter_repeat']+=1
                    else:
                        seen_keys.add(key);seen_ids.add(row.listing_uid);keep.append(row.Index)
                final=cleaned.loc[keep]
                assert len(selected)==len(final)+sum(ledger.values())
                final.to_csv(Path(td)/f'firstmover_listings_{m}.csv.gz',index=False,compression='gzip')
                cat._listings[m]=data.load_listings(m)
                assert len(cat._listings[m])==len(final)
                summaries[m]={'raw':len(cleaned),'team_clean_four_borough':len(selected),'retained':len(final),
                              'exclusions':dict(ledger),'by_borough':dict(final.borough.value_counts().items())}
        finally:
            data.FIRSTMOVER_CLEAN=original
    assert seen_ids.isdisjoint(summer_ids) and seen_keys.isdisjoint(summer_keys)
    return summaries,inputs,{m:cpi[m] for m in MONTHS+['2026-07']}


def independent_da(inst, graph):
    """Round proposals with immediate per-listing choice among offers and its previous holder."""
    n,m=graph.shape
    prefs=np.argsort(np.where(graph,inst.rank_b,999999),axis=1,kind='stable')
    degree=graph.sum(1);ptr=np.zeros(n,int);partner=np.full(n,-1,int);held=np.full(m,-1,int)
    while True:
        free=np.flatnonzero((partner<0)&(ptr<degree))
        if not len(free): break
        offers={}
        for i in free:
            j=prefs[i,ptr[i]];ptr[i]+=1
            offers.setdefault(j,[]).append(i)
        for j,people in offers.items():
            old=held[j]
            if old>=0:
                people.append(old);partner[old]=-1
            win=min(people,key=lambda i:inst.rank_s[i,j])
            partner[win]=j;held[j]=win
    return partner,held


def build(repo,summer,winter,out):
    sys.path[:0]=[str(repo),str(repo/'scripts')]
    from hiddenoptions.markets import Catalog,MarketSpec,build_market,InsufficientSupply
    from hiddenoptions.data import borough_weights
    from hiddenoptions.exposure import uniform_exposure,transparent_exposure
    from hiddenoptions.matching import deferred_acceptance,blocking_mask
    cat=Catalog()  # Summer feature references are constructed before any earlier listing is supplied.
    borough_shares=borough_weights(cat.renters)
    cleaning,hashes,cpi=prepare(cat,repo,summer,winter)
    rows=[];missing=Counter();checks=Counter();missing_keys=[]
    for d in DESIGNS:
        for m in MONTHS:
            for b in BOROUGHS:
                for s in range(SEED,SEED+NBLOCK):
                    spec=MarketSpec(b,m,s,64,64,donors='calibration',design=d,w_p=4.,sigma_r=.5)
                    try:
                        market=build_market(spec,cat)
                    except InsufficientSupply as e:
                        missing[d,m,b,str(e)]+=1;missing_keys.append([d,m,b,s]);continue
                    low=market.renters.lower_income.to_numpy(bool)
                    inst=market.inst
                    assert (market.renters.donors=='calibration').all()
                    # score4 already subtracts four times price divided by the budget.
                    rent=market.listings.rent.to_numpy(float)
                    caps=market.renters.cap.to_numpy(float)
                    share=np.divide(rent[None,:],caps[:,None],out=np.zeros(inst.E.shape),where=caps[:,None]>0)
                    alt_market=replace(market,u_observable=np.where(inst.E,market.u_observable+3*share,-np.inf))
                    if s==SEED:
                        native_one=build_market(replace(spec,w_p=1.),cat)
                        assert np.array_equal(native_one.inst.E,inst.E)
                        assert np.array_equal(native_one.listings.listing_uid,market.listings.listing_uid)
                        assert np.allclose(native_one.u_observable[inst.E],alt_market.u_observable[inst.E])
                    for quota,kw in [('k5',{'k':5}),('q40',{'q':.4})]:
                        U=uniform_exposure(market,**kw)
                        t1=transparent_exposure(alt_market,**kw)
                        t4=transparent_exposure(market,**kw)
                        if s==SEED: assert np.array_equal(t1,transparent_exposure(native_one,**kw))
                        graphs=[U,np.where(low[:,None],U,t1),np.where(low[:,None],U,t4)]
                        rec={'design':d,'month':m,'borough':b,'seed':s,'quota':quota,'lower':int(low.sum()),'higher':int((~low).sum())}
                        for arm,graph in zip(['UU','UT1','UT4'],graphs):
                            assert np.array_equal(U[low],graph[low]) and np.array_equal(U.sum(1),graph.sum(1))
                            mb,ms=deferred_acceptance(inst,graph)
                            assert not (blocking_mask(inst,mb,ms)&graph).any()
                            if s==SEED or s==SEED+NBLOCK-1:
                                rb,rs=independent_da(inst,graph)
                                assert np.array_equal(rb,mb) and np.array_equal(rs,ms)
                                checks['independent_complete_assignment_vectors']+=1
                            rec[f'{arm}_lower']=int(((mb>=0)&low).sum())
                            rec[f'{arm}_higher']=int(((mb>=0)&~low).sum())
                            checks['valid_DA_allocations']+=1
                        rows.append(rec)
    out.mkdir(exist_ok=True,parents=True)
    pd.DataFrame(rows).to_csv(out/'counts.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    details={'cleaning':cleaning,'input_sha256':hashes,'CPI_U':cpi,'borough_shares':borough_shares,
             'seed_start':SEED,'seed_blocks':NBLOCK,'paired_allocation_checks':dict(checks),
             'planned_markets':NBLOCK*len(MONTHS)*len(BOROUGHS)*len(DESIGNS),
             'unavailable':len(missing_keys),'unavailable_keys':missing_keys,
             'availability':[{'design':d,'month':m,'borough':b,'reason':r,'count':n} for (d,m,b,r),n in missing.items()],
             'protocol_sha256':digest(HERE/'DESIGN.md'),
             'code_sha256':{f:digest(repo/f) for f in ['hiddenoptions/markets.py','hiddenoptions/data.py','hiddenoptions/exposure.py',
                                  'hiddenoptions/matching.py','scripts/firstmover_clean.py','scripts/common.py']},
             'summer_reference_sha256':{m:digest(repo/f'data/firstmover/clean/firstmover_listings_{m}.csv.gz') for m in SUMMER},
             'CPI_reference_sha256':digest(repo/'data/reference/cpi_u_us_city_average.csv'),
             'census_cache_sha256':digest(repo/'data/derived/renter_profiles_v2.csv.gz'),
             'python':platform.python_version(),
             'numpy':np.__version__,'pandas':pd.__version__}
    (out/'inputs.json').write_text(json.dumps(details,indent=2,default=int)+'\n')


def summarize(out):
    raw=pd.read_csv(out/'counts.csv.gz')
    info=json.loads((out/'inputs.json').read_text())
    rng=np.random.default_rng(871136)
    f=np.vstack([np.ones(NBLOCK),rng.multinomial(NBLOCK,np.ones(NBLOCK)/NBLOCK,size=10000)])
    cells=[(m,b) for m in MONTHS for b in BOROUGHS]
    weights=np.array([info['borough_shares'][b] for m,b in cells])/3
    total_weight=weights.sum()
    joint=[];info_panels={}
    for (d,q),g in raw.groupby(['design','quota']):
        cellindex=pd.MultiIndex.from_tuples([(s,m,b) for s in range(SEED,SEED+NBLOCK) for m,b in cells],names=['seed','month','borough'])
        vars=['lower','higher','UU_lower','UT1_lower','UT4_lower','UU_higher','UT1_higher','UT4_higher']
        mat=g.set_index(cellindex.names).reindex(cellindex)[vars].to_numpy().reshape(NBLOCK,len(cells),len(vars))
        available=np.isfinite(mat[:,:,0])
        support=available.any(0)
        target_w=weights[support]/weights[support].sum()
        absent=[list(c) for c,yes in zip(cells,support) if not yes]
        tab=np.einsum('rs,scm->rcm',f,np.nan_to_num(mat[:,support,:]))
        assert (tab[:,:,:2]>0).all()
        levels={}
        for i,k in enumerate(vars[2:],2):
            dn=0 if k.endswith('lower') else 1
            levels[k]=100*np.einsum('c,rc->r',target_w,tab[:,:,i]/tab[:,:,dn])
        ci=lambda v: {'estimate':float(v[0]),'pointwise95':np.quantile(v[1:],[.025,.975]).tolist()}
        key=d+'_'+q
        info_panels[key]={'raw_low':int(g.lower.sum()),'raw_high':int(g.higher.sum()),'markets':len(g),
                          'cells_absent':absent,'original_geographic_weight':float(weights[support].sum()),
                          'fraction_four_borough_target':float(weights[support].sum()/total_weight),
                          'levels_pct':{k:ci(v) for k,v in levels.items()},'low_contrasts_pp':{},
                          'cell_counts':g.groupby(['month','borough']).size().reset_index(name='n').to_dict('records')}
        for nm,a,b in [('T1_minus_U','UT1','UU'),('T4_minus_U','UT4','UU'),('T1_minus_T4','UT1','UT4')]:
            change=levels[a+'_lower']-levels[b+'_lower']
            info_panels[key]['low_contrasts_pp'][nm]=ci(change)
            joint.append((key,nm,change))
    v=np.array([x[2] for x in joint]);sd=v[:,1:].std(axis=1,ddof=1)
    assert np.all(sd>0)
    crit=np.quantile(np.max(abs(v[:,1:]-v[:,:1])/sd[:,None],axis=0),.95)
    for (k,nm,y),s in zip(joint,sd):
        info_panels[k]['low_contrasts_pp'][nm]['simultaneous95']=[float(y[0]-crit*s),float(y[0]+crit*s)]
    r={'panels':info_panels,'primary':'uniform_k5 T1_minus_T4 (percentage points)','bootstrap':{'reps':10000,'seed':871136,'simultaneous_critical':crit,'family':len(joint)},
       'status':'New score-only contrast on previously used January–March 2026 catalogue period; summer model, donors and feature references; not new field outcomes.'}
    (out/'results.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--summer-raw',type=Path,required=True)
    p.add_argument('--winter-raw',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--summarize-only',action='store_true')
    args=p.parse_args()
    if not args.summarize_only: build(args.repo.resolve(),args.summer_raw.resolve(),args.winter_raw.resolve(),args.out)
    summarize(args.out)
