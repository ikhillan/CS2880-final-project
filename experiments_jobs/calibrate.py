"""Fit the seeker wage weight w_w and taste noise sigma_r on development months (protocol section 4, jobs mirror).

    python -m experiments_jobs.calibrate configs/jobs_calibrate.json

For each (w_w, sigma_r) on the grid, build calibration markets from calibration donors only, run seeker-proposing DA with full
disclosure, and compute log(posted salary / prior wage) among matched seekers. The loss is the mean absolute error of the 25th,
50th and 75th percentiles against the targets, averaged over the two wage groups with equal weight. Ties go to the smaller w_w,
then the smaller sigma_r. Every setting sees the same seekers, postings and shocks, because the market RNG streams ignore the
preference weights.

Writes results/<name>/grid.csv and results/<name>/selected.json. Copy the selected values into jobs_manifest.json and into the
"market" block of every evaluation config before freezing.
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from experiments_jobs.config import load_config
from hiddenoptions.jobs.markets import JobCatalog, JobMarketSpec, MONTHS_DEVELOPMENT, build_market
from hiddenoptions.matching import deferred_acceptance

ROOT = Path(__file__).resolve().parents[1]
QUANTILES = (0.25, 0.50, 0.75)


def wage_changes(spec: JobMarketSpec, catalog: JobCatalog) -> tuple[np.ndarray, np.ndarray]:
    """log(salary / prior wage) of matched seekers under full disclosure, split by wage group."""
    m = build_market(spec, catalog)
    match_b, _ = deferred_acceptance(m.inst, m.inst.E)
    matched = np.nonzero(match_b >= 0)[0]
    change = np.log(m.listings.salary.to_numpy(float)[match_b[matched]] / m.renters.wage_month.to_numpy(float)[matched])
    lower = m.renters.lower_wage.to_numpy(bool)[matched]
    return change[lower], change[~lower]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    months = cfg["grid"]["months"]
    if not set(months) <= set(MONTHS_DEVELOPMENT):
        raise SystemExit(f"calibration may only use development months {MONTHS_DEVELOPMENT}; got {months}")
    target = cfg["targets"]["log_wage_change_quantiles"]
    if any(v is None for v in target):
        raise SystemExit("fill targets.log_wage_change_quantiles in the config first (see its _notes)")
    target = np.asarray(target, float)

    catalog = JobCatalog(months=tuple(months))
    base = dict(cfg.get("market", {}), donors="calibration")
    seeds = range(cfg["grid"]["seed_start"], cfg["grid"]["seed_start"] + cfg["grid"]["seeds"])
    specs = [JobMarketSpec(cell=c, month=mo, seed=sd, n_seekers=cfg["grid"]["size"], n_postings=cfg["grid"]["size"], **base)
             for c, mo, sd in itertools.product(cfg["grid"]["cells"], months, seeds)]

    rows = []
    for w_w, sigma in itertools.product(cfg["search"]["w_wage"], cfg["search"]["seeker_noise"]):
        low, high = [], []
        for spec in specs:
            a, b = wage_changes(replace(spec, w_wage=w_w, seeker_noise=sigma), catalog)
            low.append(a)
            high.append(b)
        low, high = np.concatenate(low), np.concatenate(high)
        q_low, q_high = np.quantile(low, QUANTILES), np.quantile(high, QUANTILES)
        loss = 0.5 * np.abs(q_low - target).mean() + 0.5 * np.abs(q_high - target).mean()
        q_all = np.quantile(np.concatenate([low, high]), QUANTILES)
        rows.append({"w_wage": w_w, "seeker_noise": sigma, "loss": loss, "matched_lower": len(low), "matched_higher": len(high),
                     **{f"lower_p{int(q*100)}": v for q, v in zip(QUANTILES, q_low)},
                     **{f"higher_p{int(q*100)}": v for q, v in zip(QUANTILES, q_high)},
                     **{f"all_p{int(q*100)}": v for q, v in zip(QUANTILES, q_all)}})
        print(f"w_wage={w_w:<4} sigma={sigma:<5} loss={loss:.4f}  lower p50={q_low[1]:+.3f}  higher p50={q_high[1]:+.3f}", flush=True)

    grid = pd.DataFrame(rows).sort_values(["loss", "w_wage", "seeker_noise"]).reset_index(drop=True)
    out = ROOT / "results" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out / "grid.csv", index=False)
    best = grid.iloc[0]
    selected = {"w_wage": float(best.w_wage), "seeker_noise": float(best.seeker_noise), "loss": float(best.loss),
                "targets": target.tolist(), "months": months, "cells": cfg["grid"]["cells"], "seeds": list(seeds),
                "size": cfg["grid"]["size"], "donors": "calibration", "n_markets_per_setting": len(specs),
                "citywide_fit_p25_p50_p75": [float(best.all_p25), float(best.all_p50), float(best.all_p75)]}
    (out / "selected.json").write_text(json.dumps(selected, indent=2))
    print(f"\nselected w_wage={selected['w_wage']}, seeker_noise={selected['seeker_noise']} (loss {selected['loss']:.4f})")
    print(f"wrote {out / 'grid.csv'} and {out / 'selected.json'}")


if __name__ == "__main__":
    main()
