"""Development-only diagnostic. No recommendation calls or inferred preference scores."""
import argparse
import bisect
import csv
import json
import math
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from statistics import median
from degree import degree_probs, budget_steps

MONTHS = ('2026-06', '2026-07', '2026-08')
BOROUGHS = ('Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island')
PROBS = lru_cache(maxsize=20000)(degree_probs)


def normalized(s):
    return re.sub(r'[^A-Z0-9]', '', s.upper())


def load_catalogue(path):
    seen_ids, seen_units, rows = set(), set(), []
    ledger = Counter()
    with path.open(encoding='utf-8-sig', newline='') as stream:
        records = sorted(csv.DictReader(stream),key=lambda row:(row['created_at_utc'],row['id']))
    for row in records:
        ledger['raw'] += 1
        borough = row['borough'].strip()
        if row['state'].strip() != 'NY' or borough not in BOROUGHS:
            ledger['outside_or_unknown_geography'] += 1
            continue
        try:
            price, beds = float(row['price']), float(row['bedrooms'])
            if not math.isfinite(price) or price<=0 or beds<0 or not beds.is_integer() or not row['id'].strip():
                raise ValueError
        except (ValueError, OverflowError):
            ledger['invalid_fields'] += 1
            continue
        if row['id'] in seen_ids:
            ledger['duplicated_id'] += 1
            continue
        seen_ids.add(row['id'])
        street,unit = normalized(row['street']), normalized(row['unit'])
        key=(borough,row['zip_code'].strip(),street,unit)
        if street and unit and unit not in {'NA','NONE','UNKNOWN','NULL'}:
            if key in seen_units:
                ledger['duplicate_unit'] += 1
                continue
            seen_units.add(key)
        else:
            ledger['ambiguous_unit_retained'] += 1
        rows.append((borough, price, int(beds), row['id']))
    ledger['kept']=len(rows)
    return rows, dict(ledger)


def save_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def main():
    arg=argparse.ArgumentParser()
    arg.add_argument('--data', type=Path, required=True)
    arg.add_argument('--out', type=Path, required=True)
    args=arg.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    meta=json.loads((args.data/'acs/summary.json').read_text())
    cpi=meta['cpi_u']
    cutoff=float(meta['median_income_2024'])
    donors=defaultdict(list)
    with (args.data/'acs/recent_donors.csv').open() as stream:
        for r in csv.DictReader(stream):
            if r['donor_split']!='calibration':
                continue
            income=float(r['income_july2026'])
            size=int(r['household_size'])
            group='lower' if float(r['income_2024'])<=cutoff else 'higher'
            donors[r['borough'],group].append((max(0,(size-1)//2),max(0,0.4*income/12),float(r['weight'])))
    assert all(donors[b,g] for b in BOROUGHS for g in ('lower','higher'))
    listing_rows,aff_rows,enrich_rows,ledgers=[],[],[],{}
    for month in MONTHS:
        rows,ledgers[month]=load_catalogue(args.data/'firstmover'/f'{month}.csv')
        multiplier=float(cpi['2026-07'])/float(cpi[month])
        for borough in BOROUGHS:
            cellrows=sorted((p,h,uid) for b,p,h,uid in rows if b==borough)
            L=len(cellrows)
            prices=[r[0] for r in cellrows]
            assert L
            listing_rows.append(dict(month=month,borough=borough,L=L,median_asking_nominal=median(prices),
                                     p25_nominal=prices[math.ceil(.25*L)-1],under_2286_nominal=bisect.bisect_right(prices,2286),
                                     under_500_nominal=bisect.bisect_left(prices,500),under_1000_nominal=bisect.bisect_left(prices,1000)))
            for floor in (0,1000):
                # The floor sensitivity removes very low asking prices, without asserting a mislisting.
                pool=[(p*multiplier,h,uid) for p,h,uid in cellrows if p*multiplier>=floor]
                total=len(pool)
                for group in ('lower','higher'):
                    group_rows=donors[borough,group]
                    wg=sum(w for _,_,w in group_rows)
                    degrees=Counter()
                    gain=[0.,0.,0.,0.]
                    for minbed in sorted(set(b for b,_,_ in group_rows)):
                        eligible=[p for p,h,_ in pool if h>=minbed]
                        for b,budget,w in group_rows:
                            if b!=minbed:continue
                            st=budget_steps(eligible,budget)
                            degrees[st['affordable_count']]+=w
                            for j,name in enumerate(['gap_to_first','gap_to_sixth']):
                                if st[name] is not None:
                                    gain[2*j]+=w*st[name]
                                    gain[2*j+1]+=w
                    base=dict(month=month,borough=borough,group=group,price_floor_july=floor,catalogue_L=total,
                              donor_weight=wg,full_p0=degrees[0]/wg,full_p1to5=sum(degrees[d] for d in range(1,6))/wg)
                    for n in (24,64,128):
                        out={**base,'n':n,'available':total>=n}
                        if total>=n:
                            ps=[0.]*4
                            for D,w in degrees.items():
                                for j,x in enumerate(PROBS(total,D,n)):ps[j]+=x*w/wg
                            assert abs(sum(ps[:3])-1)<1e-9 and ps[0]+1e-9>=base['full_p0']
                            out.update(draw_p0=ps[0],draw_p1to5=ps[1],draw_p6plus=ps[2],
                                       draw_p2plus=ps[3],sampling_induced_zero=ps[0]-base['full_p0'])
                        for j,label in enumerate(['first','sixth']):
                            out['mean_budget_gap_'+label]=gain[2*j]/gain[2*j+1] if gain[2*j+1] else None
                            out['threshold_available_share_'+label]=gain[2*j+1]/wg
                        aff_rows.append(out)
                # Exact probability at least one calibration lower-income donor can use a sampled unit.
                # Same rent-tie rule as sorted pool. Each band is sampled without replacement.
                p25_count=total//4
                low=donors[borough,'lower']
                maxbudget={h:max((budget for beds,budget,_ in low if beds<=h),default=-1) for h in {beds for _,beds,_ in pool}}
                usable=[p<=maxbudget[h] for p,h,_ in pool]
                for n in (24,64,128):
                    ncheap=n//2
                    ok=p25_count>=ncheap and total-p25_count>=n-ncheap
                    out=dict(month=month,borough=borough,price_floor_july=floor,n=n,eligible_for_band_draw=ok,
                             cheap_band=p25_count,usable_cheap=sum(usable[:p25_count]),usable_other=sum(usable[p25_count:]))
                    if ok:
                        D1,D2=out['usable_cheap'],out['usable_other']
                        all_ways=math.comb(p25_count,ncheap)*math.comb(total-p25_count,n-ncheap)
                        def ways(L,D,k): return math.comb(L-D,k) if L-D>=k else 0
                        empty_ways=ways(p25_count,D1,ncheap)*ways(total-p25_count,D2,n-ncheap)
                        out['p_any_eligible_lower_donor']=(all_ways-empty_ways)/all_ways
                        out['p_no_eligible_lower_donor']=empty_ways/all_ways
                    enrich_rows.append(out)
    for name,table in [('listing_cells',listing_rows),('degree_cells',aff_rows),('enrichment_cells',enrich_rows)]:
        save_csv(args.out/f'{name}.csv',table)
    (args.out/'cleaning.json').write_text(json.dumps(ledgers,indent=2)+'\n')
    print('Produced',len(listing_rows),'listing cells,',len(aff_rows),'degree rows, and',len(enrich_rows),'enrichment rows.')

if __name__=='__main__':main()
