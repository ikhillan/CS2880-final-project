"""Independent aggregate weighting and synchronous DA checks on the temporal rental results."""
import argparse
import hashlib
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd


def synchronous(renter_rank, listing_rank, display):
    """Apply in rounds. Reject all but the best candidate at each listing."""
    applicants, listings = display.shape
    prefs=[list(reversed(sorted(np.flatnonzero(row),key=lambda j:renter_rank[i,j])))
           for i,row in enumerate(display)]
    admitted = np.full(listings,-1,dtype=int)
    assignment = np.full(applicants,-1,dtype=int)
    while True:
        proposals={}
        for i in (i for i in range(applicants) if assignment[i]<0 and prefs[i]):
            proposals.setdefault(prefs[i].pop(),[]).append(i)
        if not proposals:
            return assignment
        for j, candidates in proposals.items():
            old=admitted[j]
            if old>=0:
                candidates.append(old)
                assignment[old]=-1
            admitted[j]=min(candidates,key=lambda i:listing_rank[i,j])
            assignment[admitted[j]]=j


def summarize_by_cell(markets, draws, weights):
    avail=markets[markets.applicants.notna()].copy()
    avail["uniform_placed"]=draws[avail.index,:,1].mean(1)
    avail["count1"]=avail.first_hits-avail.expected_uniform_hits
    avail["count2"]=avail.first_placed-avail.uniform_placed
    by_cell = avail.groupby(["period","design","month","borough"])[["count1","count2","applicants"]].sum()
    by_cell=by_cell.assign(delta_coverage_pp=100*by_cell.count1/by_cell.applicants,
                           delta_placement_pp=100*by_cell.count2/by_cell.applicants).reset_index()
    complete=by_cell.groupby(["period","month","borough"]).design.transform("size")==2
    for support,tab in [("own",by_cell),("common_to_both_designs",by_cell[complete])]:
        for (period,design),part in tab.groupby(["period","design"]):
            w=part.borough.map(weights).to_numpy(float)
            yield dict(support=support,period=period,design=design,bm_cells=len(part),
                       positive_coverage_cells=int((part.delta_coverage_pp>0).sum()),
                       negative_placement_cells=int((part.delta_placement_pp<0).sum()),
                       delta_coverage_pp=float(np.average(part.delta_coverage_pp,weights=w)),
                       delta_placement_pp=float(np.average(part.delta_placement_pp,weights=w)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",required=True,type=Path)
    parser.add_argument("--clean",type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    sys.path.insert(0,str(args.repo.resolve()))
    from hiddenoptions import data
    from hiddenoptions.markets import Catalog, MarketSpec, build_market
    from hiddenoptions.matching import deferred_acceptance
    out=args.output
    data.FIRSTMOVER_CLEAN=(args.clean or args.repo/"data/firstmover/clean").resolve()
    counts=pd.read_csv(out/"temporal-markets.csv.gz")
    draws=np.load(out/"temporal-draws.npz")["uniform_counts"]
    summary=pd.read_csv(out/"temporal-summary.csv").set_index(["period","design"])
    receipt=json.loads((out/"temporal-receipt.json").read_text())
    support_rows=list(summarize_by_cell(counts,draws,receipt["borough_weights"]))
    for row in support_rows:
        if row["support"]=="own":
            for key in ["delta_coverage_pp","delta_placement_pp"]:
                np.testing.assert_allclose(row[key],summary.loc[(row["period"],row["design"]),key],
                                           rtol=0,atol=1e-11)
    # First seed in every supported calendar cell and inventory design, and all 48 draws.
    # Build using the unmodified project generator and compare serial and simultaneous DA by person.
    catalog=Catalog()
    num_matches=0
    first_checks=0
    subset=counts[counts.applicants.notna()].groupby(["period","design","month","borough"]).seed.idxmin()
    for ix in subset:
        row=counts.loc[ix]
        spec=MarketSpec(row.borough,row.month,int(row.seed),design=row.design,donors="evaluation")
        assert spec.market_id==row.market_id
        inst=build_market(spec,catalog).inst
        def checked(graph):
            byround=synchronous(inst.rank_b,inst.rank_s,graph)
            serial,_=deferred_acceptance(inst,graph)
            np.testing.assert_array_equal(byround,serial)
            return byround
        full=checked(inst.E)
        eligible=np.flatnonzero(full>=0)
        first=inst.E&(inst.rank_b<5)
        first_match=checked(first)
        target_visible=first[eligible,full[eligible]]
        assert np.all(first_match[eligible[target_visible]]>=0)
        assert np.all(inst.rank_b[eligible[target_visible],first_match[eligible[target_visible]]]
                      <=inst.rank_b[eligible[target_visible],full[eligible[target_visible]]])
        assert first_match[first_match>=0].size==row.first_placed
        assert int(target_visible.sum())==row.first_hits
        degree=inst.E[eligible].sum(1)
        assert np.isclose(np.sum(np.minimum(5,degree)/degree),row.expected_uniform_hits)
        rng=np.random.default_rng([494281,zlib.crc32(spec.market_id.encode())])
        for draw in range(draws.shape[1]):
            scores=rng.random(inst.E.shape)
            # Select independently within the feasible set rather than assigning sentinel scores.
            random=np.zeros_like(inst.E)
            for i,ok in enumerate(inst.E):
                available=np.flatnonzero(ok)
                short=available[np.argsort(scores[i,available])[:5]]
                random[i,short]=True
            result=checked(random)
            np.testing.assert_array_equal(draws[ix,draw],
                [random[eligible,full[eligible]].sum(),(result>=0).sum()])
        num_matches+=(draws.shape[1]+2)*inst.n_b
        first_checks+=len(eligible[target_visible])
    pd.DataFrame(support_rows).to_csv(out/"temporal-common-support.csv",index=False)
    verified={"independent_synchronous_checks_markets":len(subset),
              "independent_synchronous_checks_assignments":num_matches,
              "independent_synchronous_checks_allocations":len(subset)*(draws.shape[1]+2),
              "independent_prefix_rank_guarantees":first_checks,
              "all_aggregate_rates_match_second_weighting_implementation":True,
              "output_sha256":{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.glob("temporal-*"))
                               if f.name!="temporal-verification.json"}}
    (out/"temporal-verification.json").write_text(json.dumps(verified,indent=2)+"\n")
    print(json.dumps(verified,indent=2))


if __name__=="__main__":
    main()
