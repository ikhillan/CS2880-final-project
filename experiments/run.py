"""Run a config: for each market x exposure x method, record cost, optimality proof, timing and group outcomes.

    python -m experiments.run configs/pilot.json --workers 7

Resumable: finished (market, exposure, method) rows in results/<name>/runs.jsonl are skipped on a rerun.
Markets with an `llm` exposure but no saved recommendations skip only that exposure, and the summary says so.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(var, "1")

import numpy as np

from experiments.config import expand_labeled, load_config
from hiddenoptions import audit
from hiddenoptions.exposure import LlmRecommendations, build_exposure
from hiddenoptions.markets import Catalog, build_market, maximum_matching_size
from hiddenoptions.metrics import nonmonotone_additions, outcomes, plan_spread

ROOT = Path(__file__).resolve().parents[1]
_catalog, _llm_cache = None, {}


def _init_worker():
    global _catalog
    _catalog = Catalog()


def _llm(path):
    if path is None or not Path(path).exists():
        return None
    if path not in _llm_cache:
        _llm_cache[path] = LlmRecommendations(path)
    return _llm_cache[path]


def _row(market, label, exposure, stats, result, oracle, extra=None):
    return {
        "market_id": market.spec.market_id, "variant": label, "borough": market.spec.borough, "month": market.spec.month,
        "n": market.spec.n_renters, "seed": market.spec.seed, "share": market.spec.share, "design": market.spec.design,
        "exposure": exposure["name"], "policy": exposure["policy"], "exposure_stats": stats, "method": result.method,
        "cost": result.cost, "certified": result.certified, "lower_bound": result.lower_bound, "support_cost": result.support_cost,
        "seconds": result.seconds, "master_solves": result.master_solves, "n_cuts": result.n_cuts, "da_calls": result.da_calls,
        "timed_out": result.timed_out, "extra": {**result.extra, **(extra or {})}, "n_firms": market.inst.n_firms,
        "stable": oracle.stable(result.x), "plan": [market.inst.firm_names[i] for i in np.nonzero(result.x)[0]],
        "outcomes": outcomes(market, oracle, result.x),
    }


def run_market(task):
    spec, cfg, label, done = task
    t0 = time.perf_counter()
    market = build_market(spec, _catalog)
    inst, rows, skipped = market.inst, [], []
    llm = _llm(cfg.get("llm_results"))
    methods, cap = cfg["methods"], cfg.get("exact_time_cap", {})
    time_cap = float(cap.get(str(spec.n_renters), cap.get("default", 15.0)))
    summary = {"market_id": spec.market_id, "n_firms": inst.n_firms, "n_firms_listing": market.n_firms_listing, "edges": int(inst.E.sum()),
               "components": inst.n_components, "max_matching_share": maximum_matching_size(inst.E) / spec.n_renters
               if spec.n_renters <= cfg.get("max_matching_limit", 256) else None}
    for exposure in cfg["exposures"]:
        built = build_exposure(market, exposure, llm)
        if built is None:
            skipped.append(exposure["name"])
            continue
        e0, stats = built
        oracle = audit.Oracle(inst, e0)
        incumbent = None
        for method in methods:
            key = (spec.market_id, exposure["name"], method)
            if key in done:
                continue
            extra = {}
            if method == "no_audit":
                r = audit.no_audit(inst, oracle)
            elif method == "full_disclosure":
                r = audit.full_disclosure(inst, oracle)
            elif method == "reverse_delete":
                r = incumbent = audit.reverse_delete(inst, oracle)
            elif method == "blocker_greedy":
                r = audit.blocker_greedy(inst, oracle)
            elif method == "prefix_cover":
                r = audit.prefix_cover(inst, oracle)
            elif method == "exhaustive":
                r = audit.exhaustive(inst, oracle, cfg.get("exhaustive_max_firms", 18))
                if r is None:
                    continue
            elif method.startswith("exact_"):
                r = audit.exact(inst, oracle, cuts=method.split("_", 1)[1], time_cap=time_cap, seed=spec.seed, incumbent=incumbent)
                if method == cfg.get("analysis_method", "exact_prefix"):
                    extra["nonmonotone"] = nonmonotone_additions(inst, oracle, r.x)
                    enum = cfg.get("enumerate_plans")
                    if enum and r.certified:
                        plans, complete = audit.enumerate_optimal_plans(inst, oracle, r.cost, cap=enum["cap"], time_cap=enum["seconds"], seed=spec.seed)
                        extra["optimal_plans"] = {**plan_spread(market, oracle, plans), "complete": complete}
            else:
                raise ValueError(f"unknown method {method}")
            rows.append(_row(market, label, exposure, stats, r, oracle, extra))
    summary["seconds"] = time.perf_counter() - t0
    summary["skipped_exposures"] = skipped
    return summary, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, help="run at most this many markets, for timing")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ROOT / "results" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    runs_path, markets_path = out / "runs.jsonl", out / "markets.jsonl"
    done = set()
    if runs_path.exists():
        for line in runs_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["market_id"], r["exposure"], r["method"]))
    if cfg.get("llm_results"):
        cfg["llm_results"] = str(ROOT / cfg["llm_results"])
    labeled = expand_labeled(cfg)
    specs = [s for s, _ in labeled]
    pending = [(s, label) for s, label in labeled
               if any((s.market_id, e["name"], m) not in done for e in cfg["exposures"] for m in cfg["methods"])]
    # A fixed shuffle, so --limit gives a representative timing sample and progress estimates are fair.
    random.Random(0).shuffle(pending)
    if args.limit:
        pending = pending[: args.limit]
    print(f"{cfg['name']}: {len(specs)} markets, {len(cfg['exposures'])} exposures, {len(cfg['methods'])} methods; "
          f"{len(specs) - len(pending)} markets already complete; running {len(pending)} on {args.workers} workers", flush=True)
    tasks = [(s, cfg, label, {k for k in done if k[0] == s.market_id}) for s, label in pending]
    t0, skipped = time.perf_counter(), Counter()
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker) as pool, open(runs_path, "a") as runs, open(markets_path, "a") as ms:
        for i, (summary, rows) in enumerate(pool.imap_unordered(run_market, tasks), 1):
            for r in rows:
                runs.write(json.dumps(r) + "\n")
            ms.write(json.dumps(summary) + "\n")
            runs.flush()
            ms.flush()
            skipped.update(summary["skipped_exposures"])
            if i % max(1, len(tasks) // 20) == 0 or i == len(tasks):
                elapsed = time.perf_counter() - t0
                print(f"  {i}/{len(tasks)} markets, {elapsed:.0f}s elapsed, about {elapsed / i * (len(tasks) - i):.0f}s left", flush=True)
    print(f"wall clock {time.perf_counter() - t0:.1f}s; results in {runs_path}")
    if skipped:
        print(f"skipped (no saved recommendations): {dict(skipped)}. Run experiments/llm_recommend.py first.")


if __name__ == "__main__":
    main()
