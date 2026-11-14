"""Run the paper's common five-choice comparison on the prescribed NYC evaluation panels.

The repo, clean directory, and outputs are explicit arguments. This code reads the existing
market generator and common benchmark and writes only aggregates. See README.md.
"""
import argparse
import hashlib
import importlib.util
import json
import multiprocessing
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def existing_benchmark(repo):
    path = repo / "publication/two-studies/robustness/common_benchmark.py"
    spec = importlib.util.spec_from_file_location("common_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def init_catalog(repo, clean):
    global CATALOG, BENCHMARK
    sys.path.insert(0, str(repo))
    from hiddenoptions import data, markets
    from experiments.config import load_config, expand_labeled

    data.FIRSTMOVER_CLEAN = clean
    CATALOG = markets.Catalog()
    BENCHMARK = existing_benchmark(repo)
    specs = expand_labeled(load_config(repo / "configs/stage2_main.json"))
    # Each listing's feature score uses the already-frozen development reference only.
    # Computing that scalar once per listing is identical to recomputing on every sample.
    raw_quality = markets.listing_quality
    cached = {}
    for month in sorted({s.month for s, _ in specs}):
        frame = CATALOG.listings(month)
        assert frame.listing_uid.is_unique
        cached[month] = pd.Series(raw_quality(frame, CATALOG, "features"), index=frame.listing_uid)

    def feature_lookup(listings, catalog, kind):
        assert catalog is CATALOG and kind == "features"
        return cached[listings.month.iloc[0]].loc[listings.listing_uid].to_numpy()

    markets.listing_quality = feature_lookup
    census_calibration = set(CATALOG.renters.loc[CATALOG.renters.donors=="calibration","serialno"])
    census_evaluation = set(CATALOG.renters.loc[CATALOG.renters.donors=="evaluation","serialno"])
    assert not census_calibration.intersection(census_evaluation)
    for s, _ in specs:
        CATALOG.frame(s)
    return specs


def evaluate(task):
    from hiddenoptions.markets import build_market, InsufficientSupply

    spec, period = task
    meta = dict(period=period, borough=spec.borough, month=spec.month,
                design=spec.design, seed=spec.seed, market_id=spec.market_id)
    try:
        market = build_market(spec, CATALOG)
    except InsufficientSupply as error:
        return {**meta, "unavailable": str(error)}, None
    assert (market.renters.donors == "evaluation").all()
    first, random_hits, random = BENCHMARK.rental_market(market)
    return {**meta, "unavailable":"", "applicants":market.inst.n_b,
            "first_hits":int(first[0]), "first_placed":int(first[1]),
            "expected_uniform_hits":float(random_hits)}, random.astype(np.int16)


def summarize(counts, all_draws, weights, out):
    support = (counts.groupby(["period","design","borough","month","unavailable"], dropna=False)
               .size().reset_index(name="market_draws"))
    support.to_csv(out/"temporal-support.csv",index=False)
    result=[]
    for (period,design), plan in counts.groupby(["period","design"]):
        avail=plan[plan.unavailable==""].sort_values(["seed","month","borough"])
        assert avail.market_id.is_unique
        possible=sorted(set(zip(plan.month,plan.borough)))
        eligible=sorted(set(zip(avail.month,avail.borough)))
        # Support must be a union of whole cells, each with the original set of seeds.
        expected=pd.MultiIndex.from_tuples([(s,m,b) for s in sorted(plan.seed.unique())
                                             for m,b in eligible])
        assert expected.equals(pd.MultiIndex.from_frame(avail[["seed","month","borough"]]))
        n_seeds=avail.seed.nunique()
        nc=len(eligible)
        n=avail.applicants.to_numpy().reshape(n_seeds,nc)
        assert np.all(n==64)
        covered_weight=sum(weights[b] for m,b in eligible)/len({m for m,b in possible})
        cell_weight=np.array([weights[b] for m,b in eligible])
        cell_weight/=cell_weight.sum()
        first_hits=avail.first_hits.to_numpy().reshape(n_seeds,nc)
        first_placed=avail.first_placed.to_numpy().reshape(n_seeds,nc)
        uniform_hits=avail.expected_uniform_hits.to_numpy().reshape(n_seeds,nc)
        uniform=all_draws[avail.index].reshape(n_seeds,nc,-1,2)
        uniform_placed=uniform[:,:,:,1].mean(2)
        first_hit_rate=(first_hits/n)@cell_weight
        first_placed_rate=(first_placed/n)@cell_weight
        uniform_hit_rate=(uniform_hits/n)@cell_weight
        uniform_placed_rate=(uniform_placed/n)@cell_weight
        deltas=100*np.stack([first_hit_rate-uniform_hit_rate,first_placed_rate-uniform_placed_rate],axis=1)
        rng=np.random.default_rng(849736)
        bootstrap=deltas[rng.integers(n_seeds,size=(10000,n_seeds))].mean(axis=1)
        quantiles=np.quantile(bootstrap,[.025,.975],axis=0)
        monte_carlo=100*np.einsum("scr,c->r",uniform[:,:,:,1]/n[:,:,None],cell_weight)/n_seeds
        uniform_hit_draws=100*np.einsum("scr,c->r",uniform[:,:,:,0]/n[:,:,None],cell_weight)/n_seeds
        observed_gap=abs(uniform_hit_draws.mean()-100*uniform_hit_rate.mean())
        assert observed_gap < 5*uniform_hit_draws.std(ddof=1)/np.sqrt(len(uniform_hit_draws))
        result.append(dict(period=period, design=design, planned_markets=len(plan),
            used_markets=len(avail), n_seeds=n_seeds, supported_cells=nc,
            total_cells=len(possible), target_geographic_month_weight=covered_weight,
            applicants=int(n.sum()), draws=len(monte_carlo), first_coverage=100*first_hit_rate.mean(),
            uniform_coverage=100*uniform_hit_rate.mean(), first_placement=100*first_placed_rate.mean(),
            uniform_placement=100*uniform_placed_rate.mean(), delta_coverage_pp=deltas[:,0].mean(),
            delta_coverage_low_pp=quantiles[0,0],delta_coverage_high_pp=quantiles[1,0],
            delta_placement_pp=deltas[:,1].mean(),delta_placement_low_pp=quantiles[0,1],
            delta_placement_high_pp=quantiles[1,1],uniform_placement_mcse_pp=monte_carlo.std(ddof=1)/np.sqrt(len(monte_carlo))))
    pd.DataFrame(result).to_csv(out/"temporal-summary.csv",index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",required=True,type=Path)
    parser.add_argument("--clean",type=Path, help="Directory with eight cleaned FirstMover CSV.gz catalogs.")
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    repo=args.repo.resolve()
    clean=(args.clean or repo/"data/firstmover/clean").resolve()
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    specs=init_catalog(repo,clean)
    from hiddenoptions.data import borough_weights
    plan=[(replace(spec,design=design),period) for spec,period in specs
          for design in ("uniform","enriched")]
    assert len(plan)==11000
    if args.workers==1:
        records=list(map(evaluate,plan))
    else:
        # Fork shares the read-only catalogs and cached development references across workers.
        with multiprocessing.get_context("fork").Pool(args.workers) as pool:
            records=list(pool.imap(evaluate,plan,chunksize=20))
    counts=pd.DataFrame([row for row,_ in records])
    nd=BENCHMARK.DRAWS
    draws=np.stack([r if r is not None else np.full((nd,2),-1,dtype=np.int16)
                    for _,r in records])
    # Row order corresponds to the market key on the same line in the aggregate counts file.
    counts.to_csv(out/"temporal-markets.csv.gz",index=False,
                  compression={"method":"gzip","mtime":0})
    np.savez_compressed(out/"temporal-draws.npz",uniform_counts=draws)
    weights=borough_weights(CATALOG.renters)
    summarize(counts,draws,weights,out)
    input_files=["configs/stage2_main.json","configs/stage1_pilot.json",
                 "data/derived/renter_profiles_v2.csv.gz","data/reference/cpi_u_us_city_average.csv",
                 "data/firstmover/coverage_daily.csv","data/firstmover/heldout_dedup_counts.csv",
                 "hiddenoptions/markets.py","hiddenoptions/matching.py","hiddenoptions/data.py",
                 "publication/two-studies/robustness/common_benchmark.py"]
    expected_dedup=pd.read_csv(repo/"data/firstmover/heldout_dedup_counts.csv")
    regenerated=pd.DataFrame(CATALOG.dedup_counts.values())
    pd.testing.assert_frame_equal(regenerated[expected_dedup.columns].sort_values("month").reset_index(drop=True),
                                  expected_dedup.sort_values("month").reset_index(drop=True))
    required_months = sorted({"2026-06", "2026-07", "2026-08"} | {spec.month for spec, _ in specs})
    receipt=dict(input_sha256={f:digest(repo/f) for f in input_files},
        listing_sources={m:json.loads((repo/f"data/firstmover/raw/{m}.manifest.json").read_text())["sha256"]
                         for m in required_months},
        borough_weights=weights, dedup=CATALOG.dedup_counts,
        planned=len(counts),supported=int((counts.unavailable=="").sum()),
        random_clearings=int((counts.unavailable=="").sum()*nd),
        archived_development_donor_split_disjoint=True,
        heldout_dedup_agrees_with_original_report=True)
    (out/"temporal-receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print((out/"temporal-summary.csv").read_text())


if __name__=="__main__":
    main()
