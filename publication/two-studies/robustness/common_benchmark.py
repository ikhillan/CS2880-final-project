"""Compare first-five vs uniform-five under the same assignment rule in both sites."""
import argparse
import sys
from pathlib import Path
import zlib

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
DRAWS = 48


def rental_market(market):
    from hiddenoptions.matching import deferred_acceptance
    inst = market.inst
    full, _ = deferred_acceptance(inst, inst.E)
    placed = full >= 0
    ids = np.flatnonzero(placed)
    def evaluate(graph):
        result, _ = deferred_acceptance(inst, graph)
        return np.array([graph[ids, full[ids]].sum(), np.count_nonzero(result >= 0)])
    first = evaluate(inst.E & (inst.rank_b < 5))
    degree = inst.E.sum(1)
    size = np.minimum(5,degree)
    expected_hits = (size[placed] / degree[placed]).sum()
    rng = np.random.default_rng([494281, zlib.crc32(market.spec.market_id.encode())])
    uniform = []
    for _ in range(DRAWS):
        score = rng.random(inst.E.shape)
        score[~inst.E] = 2.
        keep = np.argsort(score,axis=1)[:, :5]
        graph = np.zeros_like(inst.E)
        np.put_along_axis(graph, keep, True, axis=1)
        graph &= inst.E
        assert np.array_equal(graph.sum(1),size)
        uniform.append(evaluate(graph))
    return first, expected_hits, np.asarray(uniform)


def rentals(repo):
    sys.path.insert(0,str(repo.resolve()))
    from experiments.config import expand_labeled,load_config
    from experiments.analyze import cell_weights
    from hiddenoptions.markets import Catalog, InsufficientSupply, build_market
    cfg = load_config(repo/"configs/stage1_pilot.json")
    plan = expand_labeled(cfg)
    weights = cell_weights(sorted({(s.borough,s.month) for s,_ in plan}))
    catalog = Catalog()
    panels = {}
    for spec,label in plan:
        try:
            market=build_market(spec,catalog)
        except InsufficientSupply:
            assert spec.borough == "Staten Island"
            continue
        first,expected,random=rental_market(market)
        panels.setdefault(label,[]).append((weights[(spec.borough,spec.month)],market.inst.n_b,first,expected,random))
    rows=[]
    for label, entries in panels.items():
        assert len(entries) == 240
        weights,n,first,expected,random=(np.asarray(x) for x in zip(*entries))
        weights=weights/weights.sum()
        a = np.einsum("m,mrk->rk",100*weights/n,random)
        point_first = (100*weights/n)@first
        exact = (100*weights/n)@expected
        # Uniform coverage also has a closed form. Use it as the estimand in NYC.
        rows.append(dict(site="NYC",panel=label,n=int(n.sum()),draws=DRAWS,
            first_coverage=point_first[0],uniform_coverage=exact,
            first_placed=point_first[1],uniform_placed=a[:,1].mean(),
            uniform_placement_mcse_pp=a[:,1].std(ddof=1)/np.sqrt(DRAWS),
            sampled_uniform_coverage=a[:,0].mean(),
            uniform_coverage_mcse_pp=a[:,0].std(ddof=1)/np.sqrt(DRAWS)))
    return rows


def schools():
    rows=[]
    for y in (2022,2023):
        controls=pd.read_csv(R/f"evidence/school-controls-{y}.csv")
        d=controls[controls.policy=="uniform"]
        first=controls[controls.policy=="first"].iloc[0]
        assert d.draw.nunique()==DRAWS
        scale=100/first.applicants
        rows.append(dict(site="Chile",panel=str(y),n=first.applicants,draws=DRAWS,
            first_coverage=scale*first.covered_full,uniform_coverage=scale*d.covered_full.mean(),
            first_placed=scale*first.placed,uniform_placed=scale*d.placed.mean(),
            uniform_placement_mcse_pp=scale*d.placed.sem(),
            sampled_uniform_coverage=scale*d.covered_full.mean(),uniform_coverage_mcse_pp=scale*d.covered_full.sem()))
    return rows


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=pd.DataFrame(rentals(args.repo)+schools())
    result["delta_coverage_pp"]=result.first_coverage-result.uniform_coverage
    result["delta_placement_pp"]=result.first_placed-result.uniform_placed
    result.to_csv(args.output,index=False)
    print(result[["site","panel","delta_coverage_pp","delta_placement_pp","uniform_placement_mcse_pp"]].to_string(index=False))
