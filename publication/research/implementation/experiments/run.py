"""Run a config: for each market x exposure x method, record cost, optimality proof, timing and group outcomes.

    python -m experiments.run configs/pilot.json --workers 7
    python -m experiments.run configs/stage1_pilot.json --shard 3 --num-shards 50     # one Slurm array task

Methods
    no_audit, full_disclosure, reverse_delete, blocker_greedy, prefix_cover, exhaustive
    exact_<cuts>                         C_DA, with cuts in {assignment, component, prefix}
    exact_<cuts>_protect_lower           C_L: also protect lower-income renters' avoidable losses
    exact_<cuts>_protect_all             C_all: protect every renter
    reverse_delete_protect_<group>, prefix_cover_protect_<group>   protected baselines (their verified costs seed the exact solver)
The exact rows carry C_support (the first master solve) in `support_cost`.

Resumable: finished (market, exposure, method) rows are skipped on a rerun. With --shard, each task writes its own file under
results/<name>/shards/, so array tasks never share a file. Cells without enough listings or donors are recorded as unestimable.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import time
from collections import Counter
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(var, "1")

import numpy as np

from experiments.config import expand_labeled, load_config
from hiddenoptions import audit
from hiddenoptions.exposure import LlmRecommendations, build_exposure
from hiddenoptions.markets import COST_SCALE, Catalog, InsufficientSupply, build_market, maximum_matching_size
from hiddenoptions.metrics import group_masks, group_switch, nonmonotone_additions, outcomes, plan_spread

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


def _summarize_exposure(market, stats: dict) -> dict:
    """Replace the per-renter fallback status list with per-group counts."""
    status = stats.pop("status_by_renter", None)
    if status is not None:
        status = np.array(status, dtype=object)
        stats["fallback_by_group"] = {g: dict(Counter(status[m].tolist())) for g, m in group_masks(market).items() if g in ("lower_income", "higher_income")}
    return stats


def _row(market, label, exposure, stats, result, oracle, extra=None):
    return {
        "market_id": market.spec.market_id, "variant": label, "borough": market.spec.borough, "month": market.spec.month,
        "n": market.spec.n_renters, "seed": market.spec.seed, "share": market.spec.share, "design": market.spec.design,
        "exposure": exposure["name"], "policy": exposure["policy"], "exposure_stats": stats, "method": result.method,
        "cost": result.cost, "certified": result.certified, "lower_bound": result.lower_bound, "support_cost": result.support_cost,
        "cost_scale": COST_SCALE[market.spec.audit_costs], "seconds": result.seconds, "master_solves": result.master_solves, "n_cuts": result.n_cuts, "da_calls": result.da_calls,
        "timed_out": result.timed_out, "extra": {**result.extra, **(extra or {})}, "n_firms": market.inst.n_firms,
        "stable": oracle.stable(result.x), "plan": [market.inst.firm_names[i] for i in np.nonzero(result.x)[0]],
        "outcomes": outcomes(market, oracle, result.x),
    }


def _parse_method(method: str) -> tuple[str, str | None]:
    """'exact_prefix_protect_lower' -> ('exact_prefix', 'lower')."""
    if "_protect_" in method:
        base, group = method.split("_protect_")
        return base, group
    return method, None


def _protected_oracle(market, base: audit.Oracle, group: str, cache: dict):
    if group not in cache:
        masks = group_masks(market)
        mask = masks["lower_income"] if group == "lower" else np.ones(market.inst.n_b, bool) if group == "all" else masks[group]
        protected, unavoidable = audit.protection_sets(market.inst, base, mask)
        cache[group] = audit.ProtectedOracle(base, protected, group)
        cache[group].unavoidable = unavoidable
    return cache[group]


def run_market(task):
    spec, cfg, label, done = task
    t0 = time.perf_counter()
    summary = {"market_id": spec.market_id, "variant": label, "borough": spec.borough, "month": spec.month, "n": spec.n_renters,
               "seed": spec.seed, "design": spec.design}
    try:
        market = build_market(spec, _catalog)
    except InsufficientSupply as e:
        summary.update(status="unestimable", reason=str(e), seconds=time.perf_counter() - t0)
        return summary, []
    inst, rows, skipped = market.inst, [], []
    llm = _llm(cfg.get("llm_results"))
    cap = cfg.get("exact_time_cap", {})
    time_cap = float(cap.get(str(spec.n_renters), cap.get("default", 15.0)))
    summary.update(status="ok", n_firms=inst.n_firms, n_firms_listing=market.n_firms_listing, edges=int(inst.E.sum()),
                   components=inst.n_components, sampling=market.sampling_log,
                   max_matching_share=maximum_matching_size(inst.E) / spec.n_renters if spec.n_renters <= cfg.get("max_matching_limit", 256) else None)
    exposures_e0 = {}
    for exposure in cfg["exposures"]:
        built = build_exposure(market, exposure, llm)
        if built is None:
            skipped.append(exposure["name"])
            continue
        e0, stats = built
        exposures_e0[exposure["name"]] = e0
        stats = _summarize_exposure(market, dict(stats))
        oracle = audit.Oracle(inst, e0)
        protected_cache, incumbents, costs = {}, {}, {}
        for method in cfg["methods"]:
            if (spec.market_id, exposure["name"], method) in done:
                continue
            base, group = _parse_method(method)
            o = _protected_oracle(market, oracle, group, protected_cache) if group else oracle
            extra = {}
            if base == "no_audit":
                r = audit.no_audit(inst, oracle)
            elif base == "full_disclosure":
                r = audit.full_disclosure(inst, oracle)
            elif base == "reverse_delete":
                r = audit.reverse_delete(inst, o)
            elif base == "blocker_greedy":
                if group:
                    raise ValueError("blocker_greedy has no protected version")
                r = audit.blocker_greedy(inst, oracle)
            elif base == "prefix_cover":
                r = audit.prefix_cover(inst, o)
            elif base == "exhaustive":
                r = audit.exhaustive(inst, o, cfg.get("exhaustive_max_firms", 18))
                if r is None:
                    continue
            elif base.startswith("exact_"):
                # Seed the upper bound with the best verified baseline for the same objective.
                seeds = [incumbents[k] for k in (f"reverse_delete{o.suffix}", f"prefix_cover{o.suffix}") if k in incumbents]
                cuts, support = base.split("_", 1)[1], True
                if cuts.endswith("_nosupport"):  # prefix cuts with the support master removed (solver ablation)
                    cuts, support = cuts.removesuffix("_nosupport"), False
                r = audit.exact(inst, o, cuts=cuts, time_cap=time_cap, seed=spec.seed, support=support,
                                incumbent=min(seeds, key=lambda p: p.cost) if seeds else None)
                if not support:
                    r.method += "_nosupport"
                if group:
                    extra["protected_renters"] = int(o.protected.sum())
                    extra["unavoidable_renters"] = int(o.unavoidable.sum())
                if method == cfg.get("analysis_method", "exact_prefix"):
                    extra["nonmonotone"] = nonmonotone_additions(inst, oracle, r.x)
                    enum = cfg.get("enumerate_plans")
                    if enum and r.certified:
                        plans, complete = audit.enumerate_optimal_plans(inst, oracle, r.cost, cap=enum["cap"], time_cap=enum["seconds"], seed=spec.seed)
                        extra["optimal_plans"] = {**plan_spread(market, oracle, plans), "complete": complete}
            else:
                raise ValueError(f"unknown method {method}")
            incumbents[r.method] = r
            costs[method] = r
            rows.append(_row(market, label, exposure, stats, r, o if group else oracle, extra))
        # The ordering C_support <= C_DA <= C_L <= C_all <= full holds for certified optima; flag any violation.
        chain = [costs[m] for m in ("exact_prefix", "exact_prefix_protect_lower", "exact_prefix_protect_all") if m in costs]
        if all(c.certified for c in chain) and len(chain) > 1:
            values = [c.cost for c in chain]
            for row in rows[-len(cfg["methods"]):]:
                if row["exposure"] == exposure["name"] and row["method"].startswith("exact_"):
                    row["extra"]["cost_chain_ok"] = values == sorted(values) and values[-1] <= inst.n_firms
    for sw in cfg.get("group_switch", []):
        if sw["a"] in exposures_e0 and sw["b"] in exposures_e0:
            summary.setdefault("group_switch", {})[sw["name"]] = {"a": sw["a"], "b": sw["b"],
                                                                  **group_switch(market, exposures_e0[sw["a"]], exposures_e0[sw["b"]])}
    summary["seconds"] = time.perf_counter() - t0
    summary["skipped_exposures"] = skipped
    return summary, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, help="run at most this many markets, for timing")
    ap.add_argument("--shard", type=int, help="this task's index (0-based), e.g. $SLURM_ARRAY_TASK_ID")
    ap.add_argument("--num-shards", type=int, help="total number of tasks")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ROOT / "results" / cfg["name"]
    if args.shard is not None:
        if not args.num_shards or not 0 <= args.shard < args.num_shards:
            ap.error("--shard needs --num-shards and 0 <= shard < num-shards")
        out = out / "shards"
        tag = f"-{args.shard:04d}"
    else:
        tag = ""
    out.mkdir(parents=True, exist_ok=True)
    runs_path, markets_path = out / f"runs{tag}.jsonl", out / f"markets{tag}.jsonl"
    done = set()
    if runs_path.exists():
        for line in runs_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["market_id"], r["exposure"], r["method"]))
    finished_markets = set()
    if markets_path.exists():
        for line in markets_path.read_text().splitlines():
            m = json.loads(line)
            if m.get("status") == "unestimable":
                finished_markets.add(m["market_id"])
    if cfg.get("llm_results"):
        cfg["llm_results"] = str(ROOT / cfg["llm_results"])
    labeled = sorted(expand_labeled(cfg), key=lambda sl: sl[0].market_id)
    if args.shard is not None:
        labeled = labeled[args.shard::args.num_shards]
    pending = [(s, label) for s, label in labeled if s.market_id not in finished_markets and
               any((s.market_id, e["name"], m) not in done for e in cfg["exposures"] for m in cfg["methods"])]
    random.Random(0).shuffle(pending)  # a representative --limit sample and fair progress estimates
    if args.limit:
        pending = pending[: args.limit]
    print(f"{cfg['name']}{tag}: {len(labeled)} markets, {len(cfg['exposures'])} exposures, {len(cfg['methods'])} methods; "
          f"running {len(pending)} on {args.workers} workers", flush=True)
    tasks = [(s, cfg, label, {k for k in done if k[0] == s.market_id}) for s, label in pending]
    t0, skipped, statuses = time.perf_counter(), Counter(), Counter()
    if args.workers == 1:
        _init_worker()
        results = map(run_market, tasks)
        pool = None
    else:
        pool = mp.get_context("spawn").Pool(args.workers, initializer=_init_worker)
        results = pool.imap_unordered(run_market, tasks)
    try:
        with open(runs_path, "a") as runs, open(markets_path, "a") as ms:
            for i, (summary, rows) in enumerate(results, 1):
                for r in rows:
                    runs.write(json.dumps(r) + "\n")
                ms.write(json.dumps(summary) + "\n")
                runs.flush()
                ms.flush()
                skipped.update(summary.get("skipped_exposures", []))
                statuses[summary["status"]] += 1
                if i % max(1, len(tasks) // 20) == 0 or i == len(tasks):
                    elapsed = time.perf_counter() - t0
                    print(f"  {i}/{len(tasks)} markets, {elapsed:.0f}s elapsed, about {elapsed / i * (len(tasks) - i):.0f}s left", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    print(f"wall clock {time.perf_counter() - t0:.1f}s; market status {dict(statuses)}; results in {runs_path}")
    if skipped:
        print(f"skipped (no saved recommendations): {dict(skipped)}. Run experiments/llm_recommend.py first.")


if __name__ == "__main__":
    main()
