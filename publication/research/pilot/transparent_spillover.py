"""Switch only higher-income renters to transparent shortlists in the frozen Stage 1 development markets."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np


def simultaneous_da(inst, visible):
    """Independent parallel-round DA, retaining a single favourite after each proposal round."""
    n, m = visible.shape
    prefs = [sorted(np.flatnonzero(visible[i]), key=inst.rank_b[i].__getitem__) for i in range(n)]
    held = np.full(m, -1, int)
    next_choice = np.zeros(n, int)
    bmatch = np.full(n, -1, int)
    while True:
        offers = {}
        for i in np.flatnonzero(bmatch < 0):
            if next_choice[i] < len(prefs[i]):
                j = prefs[i][next_choice[i]]
                next_choice[i] += 1
                offers.setdefault(j, []).append(i)
        if not offers:
            break
        for j, bidders in offers.items():
            if held[j] >= 0:
                bidders.append(held[j])
                bmatch[held[j]] = -1
            winner = min(bidders, key=inst.rank_s[:, j].__getitem__)
            held[j], bmatch[winner] = winner, j
    return bmatch, held


NAMES = ("lower", "higher", "fully_exposed_lower", "lost", "gained", "UU_matched_lower",
         "UT_matched_lower", "TT_matched_lower", "UU_matched_higher", "UT_matched_higher",
         "TT_matched_higher", "UU_blocked_lower", "TT_blocked_lower", "UU_blocked_higher", "TT_blocked_higher")


def outcome_measures(pooled, weights):
    vals = {name: pooled[..., i] for i, name in enumerate(NAMES)}
    lower, higher = vals["lower"], vals["higher"]
    assert np.all(lower > 0) and np.all(higher > 0)
    def rate(num, denom):
        return ((num / denom) * weights).sum(axis=-1)
    subgroup_share = rate(vals["fully_exposed_lower"], lower)
    measures = {"share_fully_exposed_lower": subgroup_share}
    for name in ("lost", "gained"):
        share = rate(vals[name], lower)
        measures[name + "_of_all_lower"] = share
        measures[name + "_given_fully_exposed_lower"] = np.divide(share, subgroup_share,
            where=subgroup_share > 0, out=np.full_like(subgroup_share, np.nan))
    measures["net_match_change_fully_exposed_of_all_lower"] = measures["gained_of_all_lower"] - measures["lost_of_all_lower"]
    for group, size in [("lower", lower), ("higher", higher)]:
        for arm in ("UU", "UT", "TT"):
            measures[f"{arm}_matched_{group}"] = rate(vals[f"{arm}_matched_{group}"], size)
        for arm in ("UU", "TT"):
            measures[f"{arm}_blocked_{group}"] = rate(vals[f"{arm}_blocked_{group}"], size)
        measures[f"UT_minus_UU_matched_{group}"] = measures[f"UT_matched_{group}"] - measures[f"UU_matched_{group}"]
    return measures


def main(repo, archive, out):
    sys.path.insert(0, str(repo.resolve()))
    from experiments.config import expand_labeled, load_config
    from hiddenoptions.data import borough_weights
    from hiddenoptions.exposure import uniform_exposure, transparent_exposure
    from hiddenoptions.markets import Catalog, InsufficientSupply, build_market

    cfg = load_config(repo / "configs/stage1_pilot.json")
    original = json.loads(archive.read_text())
    catalog = Catalog()
    shares = borough_weights(catalog.renters)
    cells = [(b,m) for b in sorted(set(shares) - {"Staten Island"}) for m in cfg["grid"]["months"]]
    weights = np.array([shares[b] / 3 for b,m in cells])
    weights /= weights.sum()
    panels = {v:np.zeros((20,len(cells),2,len(NAMES)),int) for v in ("primary", "enriched")}
    seen = set()
    for spec, variant in expand_labeled(cfg):
        assert spec.month in {"2026-06", "2026-07", "2026-08"} and spec.donors=="calibration"
        try:
            market = build_market(spec, catalog)
        except InsufficientSupply:
            assert spec.borough == "Staten Island"
            continue
        block = spec.seed - original["designs"][variant]["seeds"][0]
        cell = cells.index((spec.borough, spec.month))
        assert (variant, block, cell) not in seen
        seen.add((variant,block,cell))
        g = market.renters.lower_income.to_numpy(bool)
        inst = market.inst
        d = inst.E.sum(1)
        for budget,(k,q) in enumerate([(5,None),(None,.4)]):
            u, t = uniform_exposure(market,k,q), transparent_exposure(market,k,q)
            exposed = (d > 0) & (d == u.sum(1)) & g
            partial = np.where(g[:,None],u,t)
            assert np.array_equal(u.sum(1),t.sum(1)) and np.array_equal(u[g],partial[g])
            assert np.array_equal(u[exposed],inst.E[exposed])
            rec = {"lower":g.sum(), "higher":(~g).sum(),"fully_exposed_lower":exposed.sum()}
            assignments = {}
            for label, visible in [("UU",u),("UT",partial),("TT",t)]:
                matched, held = simultaneous_da(inst,visible)
                rb = np.where(matched<0, np.inf, inst.rank_b[np.arange(len(g)),matched])
                rs = np.where(held<0, np.inf, inst.rank_s[held,np.arange(len(held))])
                blockers = inst.E & (inst.rank_b < rb[:,None]) & (inst.rank_s < rs[None,:])
                assert not (blockers & visible).any() and not blockers[exposed].any()
                assignments[label] = matched >= 0
                for name, group in [("lower",g),("higher",~g)]:
                    rec[f"{label}_matched_{name}"] = (group & assignments[label]).sum()
                    rec[f"{label}_blocked_{name}"] = (group & blockers.any(1)).sum()
            rec["lost"] = (exposed & assignments["UU"] & ~assignments["UT"]).sum()
            rec["gained"] = (exposed & ~assignments["UU"] & assignments["UT"]).sum()
            panels[variant][block,cell,budget] = [rec[key] for key in NAMES]
    assert len(seen)==480
    results = {"market_count":480, "allocation_count":2880, "order":"first letter lower-income rule; U uniform, T transparent",
               "definition":"Group rates are four-borough standardized, pooling 20 blocks inside each cell. Conditional transition rates reweight to the fully exposed positive-degree lower-income subgroup in the standardized population. Values and intervals are percents or percentage points.",
               "checks":"Independent simultaneous DA; all 2,880 assignments stable on visible edges; all homogeneous-arm group match and blocker rates checked against stage1_results.json.",
               "bootstrap":"10,000 paired draws of 20 whole seed blocks; seed 731911", "panels":{}}
    draws = np.random.default_rng(731911).multinomial(20, np.repeat(1/20,20),size=10000)
    for variant,counts in panels.items():
        results["panels"][variant] = {}
        for qi, budget in enumerate(("k5","q40")):
            ints = counts[:,:,qi,:]
            summaries = outcome_measures(ints.sum(axis=0), weights)
            samples = outcome_measures(np.einsum("rb,bcm->rcm",draws,ints),weights)
            for arm, policy in [("UU","uniform"),("TT","transparent")]:
                for name,group in [("lower","lower_income"),("higher","higher_income")]:
                    ref = original["designs"][variant]["policies"][f"{policy}_{budget}"][group]
                    for metric,key in [("matched","match_before"),("blocked","own_blocker_rate")]:
                        assert abs(summaries[f"{arm}_{metric}_{name}"]-ref[key])<1e-12
            result = {metric:dict(estimate=100*float(v),interval_95=(100*np.quantile(samples[metric],[.025,.975])).tolist())
                      for metric,v in summaries.items()}
            totals = ints.sum(axis=(0,1))
            result["raw_counts"] = {k:int(totals[NAMES.index(k)]) for k in NAMES[:5]}
            results["panels"][variant][budget] = result
    out.write_text(json.dumps(results,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",type=Path,required=True)
    p.add_argument("--stage1-results",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args();main(args.repo,args.stage1_results,args.out)
