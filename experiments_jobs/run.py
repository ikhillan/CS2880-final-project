"""Run a config: for each market x exposure x method, record cost, optimality proof, timing and group outcomes.

    python -m experiments_jobs.run configs/jobs_pilot.json --workers 7

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

from experiments_jobs.config import domain_modules, expand_labeled, load_config
from hiddenoptions import audit
from hiddenoptions.markets import maximum_matching_size
from hiddenoptions.metrics import nonmonotone_additions, outcomes, plan_spread

ROOT = Path(__file__).resolve().parents[1]
_catalog, _llm_cache, _mods = None, {}, None


def _init_worker(cfg):
    """Each worker loads the domain's data once. Jobs catalogs load only the months the config names, so a development
    run cannot read the frozen or held-out months."""
    global _catalog, _mods
    _mods = domain_modules(cfg)
    markets = _mods[0]
    _catalog = markets.Catalog() if cfg["domain"] == "housing" else markets.JobCatalog(months=tuple(cfg["grid"]["months"]))


def _llm(path):
    if path is None or not Path(path).exists():
        return None
    if path not in _llm_cache:
        exposure = _mods[1]
        cls = getattr(exposure, "JobLlmRecommendations", None) or exposure.LlmRecommendations
        _llm_cache[path] = cls(path)
    return _llm_cache[path]


def _row(market, label, exposure, stats, result, oracle, extra=None):
    spec = market.spec
    cell = getattr(spec, "borough", None) if hasattr(spec, "borough") else spec.cell
    n = getattr(spec, "n_renters", None) or spec.n_seekers
    return {
        "market_id": spec.market_id, "variant": label, "borough": cell, "month": spec.month,
        "n": n, "seed": spec.seed, "share": spec.share, "design": spec.design,
        "exposure": exposure["name"], "policy": exposure["policy"], "exposure_stats": stats, "method": result.method,
        "cost": result.cost, "certified": result.certified, "lower_bound": result.lower_bound, "support_cost": result.support_cost,
        "seconds": result.seconds, "master_solves": result.master_solves, "n_cuts": result.n_cuts, "da_calls": result.da_calls,
        "timed_out": result.timed_out, "extra": {**result.extra, **(extra or {})}, "n_firms": market.inst.n_firms,
        "stable": oracle.stable(result.x), "plan": [market.inst.firm_names[i] for i in np.nonzero(result.x)[0]],
        "outcomes": outcomes(market, oracle, result.x),
    }


def _group_mask(market, group: str) -> np.ndarray:
    """Renters (seekers) to protect. "lower" = lower income / lower wage; "all" = everyone."""
    if group == "all":
        return np.ones(market.inst.n_b, bool)
    if group == "lower":
        return market.renters.lower_income.to_numpy(bool)
    raise ValueError(f"unknown protect group {group}")


def run_market(task):
    spec, cfg, label, done = task
    t0 = time.perf_counter()
    markets, exposure_mod, _ = _mods
    market = markets.build_market(spec, _catalog)
    n_prop = getattr(spec, "n_renters", None) or spec.n_seekers
    inst, rows, skipped = market.inst, [], []
    llm = _llm(cfg.get("llm_results"))
    methods, cap = cfg["methods"], cfg.get("exact_time_cap", {})
    time_cap = float(cap.get(str(n_prop), cap.get("default", 15.0)))
    summary = {"market_id": spec.market_id, "n_firms": inst.n_firms, "n_firms_listing": market.n_firms_listing, "edges": int(inst.E.sum()),
               "components": inst.n_components, "max_matching_share": maximum_matching_size(inst.E) / n_prop
               if n_prop <= cfg.get("max_matching_limit", 256) else None}
    for exposure in cfg["exposures"]:
        built = exposure_mod.build_exposure(market, exposure, llm)
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
        # Protected repairs (protocol section 2): for each group in cfg["protect"], the cheapest plan that is stable AND leaves
        # every protected member at least as well off as in the first-round match. Uses audit.ProtectedOracle; method names
        # get a _protect_<group> suffix, so analyze.py tables them alongside the ordinary methods.
        for group in cfg.get("protect", []):
            mask = _group_mask(market, group)
            protected, unavoidable = audit.protection_sets(inst, oracle, mask)
            por = audit.ProtectedOracle(oracle, protected, group)
            p_extra = {"n_group": int(mask.sum()), "n_protected": int(protected.sum()), "n_unavoidable": int(unavoidable.sum())}
            p_inc = None
            for pm in cfg.get("protect_methods", ["reverse_delete", "exact_prefix"]):
                key = (spec.market_id, exposure["name"], pm + por.suffix)
                if key in done:
                    continue
                if pm == "reverse_delete":
                    r = p_inc = audit.reverse_delete(inst, por)
                elif pm == "prefix_cover":
                    r = audit.prefix_cover(inst, por)
                elif pm.startswith("exact_"):
                    r = audit.exact(inst, por, cuts=pm.split("_", 1)[1], time_cap=time_cap, seed=spec.seed, incumbent=p_inc)
                else:
                    raise ValueError(f"unknown protected method {pm}")
                if por.violations(r.x).any():
                    raise AssertionError(f"{r.method}: returned plan leaves a protected renter worse off")
                rows.append(_row(market, label, exposure, stats, r, oracle, p_extra))
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
    if cfg["domain"] == "jobs":
        from hiddenoptions.jobs.markets import MONTHS_DEVELOPMENT
        late = sorted(set(cfg["grid"]["months"]) - set(MONTHS_DEVELOPMENT))
        manifest = ROOT / cfg.get("study_manifest", "jobs_manifest.json")
        frozen = manifest.exists() and json.loads(manifest.read_text()).get("frozen") is True
        if late and not frozen:
            raise SystemExit(f"{late} are frozen or held-out months, and {manifest.name} is not frozen. Freeze the design first.")
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
    all_methods = list(cfg["methods"]) + [pm + f"_protect_{g}" for g in cfg.get("protect", [])
                                          for pm in cfg.get("protect_methods", ["reverse_delete", "exact_prefix"])]
    pending = [(s, label) for s, label in labeled
               if any((s.market_id, e["name"], m) not in done for e in cfg["exposures"] for m in all_methods)]
    # A fixed shuffle, so --limit gives a representative timing sample and progress estimates are fair.
    random.Random(0).shuffle(pending)
    if args.limit:
        pending = pending[: args.limit]
    print(f"{cfg['name']}: {len(specs)} markets, {len(cfg['exposures'])} exposures, {len(cfg['methods'])} methods; "
          f"{len(specs) - len(pending)} markets already complete; running {len(pending)} on {args.workers} workers", flush=True)
    tasks = [(s, cfg, label, {k for k in done if k[0] == s.market_id}) for s, label in pending]
    t0, skipped = time.perf_counter(), Counter()
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker, initargs=(cfg,)) as pool, open(runs_path, "a") as runs, open(markets_path, "a") as ms:
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
