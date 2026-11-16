"""Reproducible synthetic audit-planning benchmark; no real housing records.

Development seeds and confirmatory evaluation seeds are disjoint. Every solver
result is independently checked, and any timeout remains explicitly censored.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import itertools
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "methodology"))
import numpy as np
import scipy
from scipy.special import expit
from audit_solver import (Market, deferred_acceptance, evaluate, exhaustive_optimum,
                          reverse_delete, solve_disclosure)


def generate_market(n=24, firms=10, components=1, preference="independent",
                    exposure=0.35, seed=0, mixed=False, weighted=False,
                    partnership_log_odds=math.log(4), access_log_odds=math.log(0.5)):
    """Explicit synthetic opportunities, rankings, portfolios, and exposure.

    Each balanced local market has complete mutual acceptability. Buyers and
    sellers cannot transact across local markets. Firm portfolios cross local
    markets. Preference scores are ordinal constructions, not estimated values.
    """
    assert n % components == 0 and firms % 2 == 0
    assert firms // 2 <= n
    rng = np.random.default_rng(seed)
    loc_b = rng.uniform(size=(n, 2))
    loc_s = rng.uniform(size=(n, 2))
    quality = rng.uniform(size=n)
    reliability = rng.uniform(size=n)
    timing_b, timing_s = rng.uniform(size=n), rng.uniform(size=n)
    noise_b, noise_s = rng.normal(size=(n, n)), rng.normal(size=(n, n))
    if preference == "independent":
        score_b, score_s = noise_b, noise_s
    elif preference == "correlated":
        dist = np.linalg.norm(loc_b[:, None, :] - loc_s[None, :, :], axis=2)
        timing = np.abs(timing_b[:, None] - timing_s[None, :])
        score_b = -dist + 0.6 * quality[None, :] - 0.3 * timing + 0.15 * noise_b
        score_s = 0.8 * reliability[:, None] - 0.4 * timing + 0.15 * noise_s
    else:
        raise ValueError(preference)
    half = firms // 2
    if mixed:
        allocation = np.tile(np.arange(firms), math.ceil(2*n/firms))[:2*n]
        rng.shuffle(allocation)
        bf, sf = allocation[:n], allocation[n:]
    else:
        bf = np.tile(np.arange(half), math.ceil(n/half))[:n]
        sf = np.tile(np.arange(half, firms), math.ceil(n/half))[:n]
        rng.shuffle(bf); rng.shuffle(sf)
    low_access = np.array([0]*(n//2) + [1]*(n-n//2))
    rng.shuffle(low_access)
    partnership = rng.random((firms, firms)) < 0.3
    block_size = n // components
    edges = [(b, s) for b in range(n) for s in range(n)
             if b//block_size == s//block_size]
    offsets = np.array([partnership_log_odds*partnership[bf[b], sf[s]] +
                        access_log_odds*low_access[b] for b, s in edges])
    lo, hi = -30.0, 30.0
    for _ in range(80):
        mid = (lo+hi)/2
        if np.mean(expit(mid+offsets)) < exposure: lo = mid
        else: hi = mid
    alpha = (lo+hi)/2
    probabilities = expit(alpha+offsets)
    initial = [e for e, u, q in zip(edges, rng.uniform(size=len(edges)), probabilities) if u < q]
    bp = [tuple(sorted((s for bb, s in edges if bb == b), key=lambda s: (-score_b[b, s], s)))
          for b in range(n)]
    sp = [tuple(sorted((b for b, ss in edges if ss == s), key=lambda b: (-score_s[b, s], b)))
          for s in range(n)]
    costs = tuple(map(int, rng.integers(1, 4, size=firms))) if weighted else (1,)*firms
    metadata = dict(n=n, firms=firms, components=components, preference=preference,
                    target_exposure=exposure, seed=seed, mixed=mixed, weighted=weighted,
                    low_access=low_access.tolist(), partnership_log_odds=partnership_log_odds,
                    access_log_odds=access_log_odds, logistic_intercept=alpha,
                    realized_exposure=len(initial)/len(edges))
    name = f"n{n}_f{firms}_c{components}_{preference}_e{exposure:g}_s{seed}"
    if mixed: name += "_mixed"
    if weighted: name += "_weighted"
    return Market(bp, sp, bf, sf, costs, initial, name=name, metadata=metadata)


def outcome_metrics(market, mask):
    before, before_blocks = evaluate(market, 0)
    out, blocks = evaluate(market, mask)
    def scores(bm, sm):
        bscore = [0.0 if s < 0 else 1-market.buyer_rank[b][s]/len(market.buyer_prefs[b])
                  for b, s in enumerate(bm)]
        sscore = [0.0 if b < 0 else 1-market.seller_rank[s][b]/len(market.seller_prefs[s])
                  for s, b in enumerate(sm)]
        return np.array(bscore), np.array(sscore)
    b0, s0 = scores(before.buyer_match, before.seller_match)
    b1, s1 = scores(out.buyer_match, out.seller_match)
    g = np.array(market.metadata.get("low_access", [0]*market.n_buyers), dtype=bool)
    result = dict(initial_blockers=len(before_blocks), final_blockers=len(blocks),
                  initial_cardinality=sum(x >= 0 for x in before.buyer_match),
                  final_cardinality=sum(x >= 0 for x in out.buyer_match),
                  buyer_rank_score=float(b1.mean()), seller_rank_score=float(s1.mean()),
                  buyer_score_change=float((b1-b0).mean()), seller_score_change=float((s1-s0).mean()),
                  buyers_harmed=int(np.sum(b1 < b0)), sellers_harmed=int(np.sum(s1 < s0)),
                  buyers_helped=int(np.sum(b1 > b0)), sellers_helped=int(np.sum(s1 > s0)),
                  initial_buyer_score=float(b0.mean()), initial_seller_score=float(s0.mean()))
    for label, ix in [("lower_access", g), ("higher_access", ~g)]:
        if ix.any():
            result[label+"_buyer_change"] = float((b1-b0)[ix].mean())
            result[label+"_buyer_harmed_fraction"] = float((b1[ix] < b0[ix]).mean())
    return result


def greedy_blockers(market):
    start = perf_counter(); mask = 0; evaluations = 1
    _, current = evaluate(market, mask)
    while current:
        candidates = []
        for f in range(market.n_firms):
            if mask & (1 << f): continue
            candidate = mask | (1 << f)
            _, blocks = evaluate(market, candidate); evaluations += 1
            gain = (len(current)-len(blocks))/market.costs[f]
            candidates.append((-gain, len(blocks), market.costs[f], f, candidate, blocks))
        _, _, _, _, mask, current = min(candidates)
    return dict(mask=mask, cost=market.cost(mask), elapsed=perf_counter()-start,
                evaluations=evaluations, optimal=False, status="feasible")


def one_instance(config, methods, limit, outdir):
    market = generate_market(**config)
    results = []
    from audit_solver import prefix_cover
    # Balance initialization/cache effects without using any observed outcomes.
    digest = int(hashlib.sha256(market.name.encode()).hexdigest()[:8], 16)
    shift = digest % len(methods)
    method_order = methods[shift:] + methods[:shift]
    if digest & 1: method_order = list(reversed(method_order))
    for method in method_order:
        start = perf_counter()
        if method == "enumeration":
            result = exhaustive_optimum(market, max_firms=14, time_limit=limit)
        elif method == "full":
            mask = (1 << market.n_firms)-1
            result = dict(mask=mask, cost=market.cost(mask), optimal=False, status="feasible")
        elif method == "reverse_delete":
            result = reverse_delete(market, time_limit=limit)
            result.update(optimal=False, status="feasible")
        elif method == "greedy":
            result = greedy_blockers(market)
        elif method == "prefix_cover":
            result = prefix_cover(market, time_limit=limit)
        else:
            support = method != "prefix_no_support"
            variant = "prefix" if method == "prefix_no_support" else method
            result = solve_disclosure(market, variant=variant, stable_support=support,
                                      time_limit=limit, warm_start="reverse_delete")
        result.setdefault("elapsed", perf_counter()-start)
        result.update(instance=market.name, method=method, config=config)
        if result.get("mask") is not None:
            metrics = outcome_metrics(market, result["mask"])
            assert metrics["final_blockers"] == 0, (market.name, method, result)
            result["metrics"] = metrics
        results.append(result)
    exact = [r["cost"] for r in results if r.get("optimal") and r.get("method") != "prefix_cover"]
    assert len(set(exact)) <= 1, (market.name, exact)
    if exact:
        opt = exact[0]
        for r in results:
            if r.get("cost") is not None:
                assert r["cost"] >= opt-1e-6
                r["reference_optimum"] = opt
                r["excess_cost"] = r["cost"]-opt
    tie_analysis = None
    if "enumeration" in methods and exact:
        start = perf_counter()
        optimal_masks = [mask for mask in range(1 << market.n_firms)
                         if abs(market.cost(mask)-exact[0]) < 1e-7 and not evaluate(market,mask)[1]]
        measurements = [outcome_metrics(market,mask) for mask in optimal_masks]
        keys = ["buyers_harmed","sellers_harmed","buyer_score_change","seller_score_change",
                "lower_access_buyer_change","higher_access_buyer_change",
                "lower_access_buyer_harmed_fraction","higher_access_buyer_harmed_fraction",
                "final_cardinality"]
        tie_analysis = dict(count=len(optimal_masks), masks=optimal_masks,
                            elapsed=perf_counter()-start,
                            ranges={k:[min(m[k] for m in measurements),max(m[k] for m in measurements)]
                                    for k in keys if all(k in m for m in measurements)})
    payload = dict(market=market.to_dict(), results=results,method_order=method_order,
                   optimal_outcome_sensitivity=tie_analysis)
    path = Path(outdir)/"instances"/(market.name+".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return market.name, len(results), sum(r["elapsed"] for r in results)


def configuration(suite):
    if suite == "pilot":
        configs = [dict(n=24, firms=10, components=c, preference=p, exposure=.35, seed=900000+j)
                   for j, (c, p) in enumerate(itertools.product((1,4), ("independent","correlated")))]
        methods = ["enumeration","assignment","component","prefix","prefix_no_support",
                   "reverse_delete","greedy","prefix_cover","full"]
        return configs, methods, 10.0
    if suite == "core":
        configs = [dict(n=24, firms=10, components=c, preference=p, exposure=e, seed=1000+j)
                   for c, p, e, j in itertools.product((1,4), ("independent","correlated"), (.25,.5), range(20))]
        methods = ["enumeration","assignment","component","prefix",
                   "reverse_delete","greedy","prefix_cover","full"]
        return configs, methods, 10.0
    if suite == "ablation":
        configs = [dict(n=24, firms=10, components=c, preference=p, exposure=e, seed=1000+j)
                   for c, p, e, j in itertools.product((1,4), ("independent","correlated"), (.25,.5), range(5))]
        return configs, ["prefix","prefix_no_support"], 5.0
    if suite == "scale":
        configs = [dict(n=64, firms=24, components=c, preference=p, exposure=.35, seed=2000+j)
                   for c, p, j in itertools.product((1,8), ("independent","correlated"), range(8))]
        methods = ["assignment","component","prefix","reverse_delete","greedy","prefix_cover","full"]
        return configs, methods, 15.0
    if suite == "sensitivity":
        configs = [dict(n=24, firms=10, components=4, preference="correlated", exposure=.35,
                        seed=3000+j, mixed=mixed, weighted=weighted)
                   for mixed, weighted, j in itertools.product((False,True), (False,True), range(8))]
        return configs, ["enumeration","prefix","reverse_delete","greedy","prefix_cover","full"], 10.0
    raise ValueError(suite)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["pilot","core","scale","sensitivity","ablation"], default="pilot")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    configs, methods, limit = configuration(args.suite)
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = dict(suite=args.suite, created_utc=datetime.now(timezone.utc).isoformat(),
                    configs=configs, methods=methods, per_method_seconds=limit, workers=args.workers,
                    python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                    platform=platform.platform(), machine=platform.machine(),
                    timing_note="Sequential methods within each market in hash-rotated order; HiGHS threads=1; markets may run in parallel.",
                    seed_note="Pilot seeds excluded. Evaluation seeds are shared across scenario cells: bootstrap seed blocks, preserving within-market method pairing.",
                    optimal_outcomes="Core/sensitivity exhaustively enumerate all cost-optimal audit sets; tie-dependent outcome ranges stored separately from solver timings.")
    target = args.output/"protocol.json"
    if target.exists():
        old = json.loads(target.read_text())
        assert old["configs"] == configs and old["methods"] == methods
    else:
        target.write_text(json.dumps(protocol, indent=2))
    pending = [c for c in configs if not (args.output/"instances"/(generate_market(**c).name+".json")).exists()]
    print(json.dumps(dict(suite=args.suite,total=len(configs),pending=len(pending),methods=methods)), flush=True)
    start = perf_counter()
    if args.workers == 1:
        completed = (one_instance(c, methods, limit, args.output) for c in pending)
        for j, result in enumerate(completed, 1):
            print(json.dumps(dict(done=j, instance=result[0], elapsed=perf_counter()-start)), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(one_instance,c,methods,limit,args.output) for c in pending]
            for j, future in enumerate(as_completed(futures),1):
                result = future.result()
                print(json.dumps(dict(done=j,instance=result[0],elapsed=perf_counter()-start)),flush=True)


if __name__ == "__main__":
    main()
