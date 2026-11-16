"""Calibrate the renter price weight and taste noise (protocol section 4).

    python -m experiments.calibrate_preferences --workers 7

Grid: w_p in {1, 2, 4} x sigma_r in {0.5, 0.75, 1.25}. Each setting runs on the SAME markets and random draws (market streams
ignore preference weights): 20 calibration seed blocks x the 15 development cells (5 boroughs x Jun-Aug 2026), 64 per side,
uniform design, calibration donors, disjoint from the pilot's seeds.

Fit: full-disclosure DA, then the 25th, 50th and 75th percentiles of rent burden (12 x monthly asking rent / annual income,
both July 2026 dollars) among matched renters, by income group. Target: survey-weighted contract-rent burden of positive-income
recent-mover ACS households (calibration donors) whose burden fits under the same cap. Loss: mean absolute error over the
three quantiles, groups weighted equally. Ties go to the smaller w_p, then the smaller sigma_r.

Writes results/calibration/calibration.json and calibration.md. All nine fits are kept for the sensitivity panel.
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
from dataclasses import replace
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(var, "1")

import numpy as np

from hiddenoptions.data import BOROUGHS, DEV_MONTHS, weighted_quantile
from hiddenoptions.markets import Catalog, InsufficientSupply, MarketSpec, build_market
from hiddenoptions.matching import deferred_acceptance

ROOT = Path(__file__).resolve().parents[1]
W_P, SIGMA_R = (1.0, 2.0, 4.0), (0.5, 0.75, 1.25)
QUANTILES = (0.25, 0.5, 0.75)
SHARE = 0.40
_catalog = None


def _init():
    global _catalog
    _catalog = Catalog()


def burdens(spec: MarketSpec):
    """Matched renters' burdens under full-disclosure DA for every grid setting, by income group."""
    out = {}
    for w_p, sigma_r in itertools.product(W_P, SIGMA_R):
        try:
            m = build_market(replace(spec, w_p=w_p, sigma_r=sigma_r), _catalog)
        except InsufficientSupply:
            return None
        mb, _ = deferred_acceptance(m.inst, m.inst.E)
        matched = mb >= 0
        burden = 12 * m.listings.rent.to_numpy(float)[mb[matched]] / m.renters.income.to_numpy(float)[matched]
        lower = m.renters.lower_income.to_numpy(bool)[matched]
        out[(w_p, sigma_r)] = {"lower": burden[lower].tolist(), "higher": burden[~lower].tolist()}
    return out


def acs_targets(renters) -> dict:
    m = renters[renters.recent_mover & (renters.donors == "calibration") & (renters.income_2024 > 0) & renters.acs_contract_burden.notna()]
    out = {"excluded_by_cap_share": float(m.weight[m.acs_contract_burden > SHARE].sum() / m.weight.sum())}
    for group, mask in (("lower", m.lower_income), ("higher", ~m.lower_income)):
        g = m[mask]
        out[f"unconstrained_{group}"] = [weighted_quantile(g.acs_contract_burden.to_numpy(), g.weight.to_numpy(), q) for q in QUANTILES]
        feasible = g[g.acs_contract_burden <= SHARE]
        out[group] = [weighted_quantile(feasible.acs_contract_burden.to_numpy(), feasible.weight.to_numpy(), q) for q in QUANTILES]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    specs = [MarketSpec(b, m, seed, 64, 64, design="uniform", donors="calibration", share=SHARE)
             for b in BOROUGHS for m in DEV_MONTHS for seed in range(args.seeds)]
    with mp.get_context("spawn").Pool(args.workers, initializer=_init) as pool:
        results = pool.map(burdens, specs)
    usable = [r for r in results if r is not None]
    pooled = {k: {"lower": [], "higher": []} for k in itertools.product(W_P, SIGMA_R)}
    for r in usable:
        for k, v in r.items():
            pooled[k]["lower"] += v["lower"]
            pooled[k]["higher"] += v["higher"]
    targets = acs_targets(Catalog().renters)
    fits = []
    for (w_p, sigma_r), groups in pooled.items():
        sim = {g: [float(np.quantile(v, q)) for q in QUANTILES] if v else [np.nan] * 3 for g, v in groups.items()}
        loss = float(np.mean([np.mean(np.abs(np.array(sim[g]) - np.array(targets[g]))) for g in ("lower", "higher")]))
        fits.append({"w_p": w_p, "sigma_r": sigma_r, "loss": loss, "simulated": sim,
                     "matched_lower": len(groups["lower"]), "matched_higher": len(groups["higher"])})
    fits.sort(key=lambda f: (round(f["loss"], 12), f["w_p"], f["sigma_r"]))
    selected = fits[0]
    out = ROOT / "results" / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    payload = {"selected": {"w_p": selected["w_p"], "sigma_r": selected["sigma_r"]}, "targets": targets, "fits": fits,
               "markets_used": len(usable), "markets_unestimable": len(results) - len(usable), "share": SHARE}
    (out / "calibration.json").write_text(json.dumps(payload, indent=1))
    lines = ["# Preference calibration\n", f"Markets used: {len(usable)} (unestimable: {len(results) - len(usable)}).\n",
             f"ACS targets (25/50/75th pct burden, within the {SHARE:.0%} cap): lower {np.round(targets['lower'], 3).tolist()}, "
             f"higher {np.round(targets['higher'], 3).tolist()}. Share of recent movers above the cap: {targets['excluded_by_cap_share']:.1%}.\n",
             f"Unconstrained ACS: lower {np.round(targets['unconstrained_lower'], 3).tolist()}, higher {np.round(targets['unconstrained_higher'], 3).tolist()}.\n",
             "| w_p | sigma_r | loss | lower sim (25/50/75) | higher sim (25/50/75) |", "|---|---|---|---|---|"]
    for f in fits:
        lines.append(f"| {f['w_p']} | {f['sigma_r']} | {f['loss']:.4f} | {np.round(f['simulated']['lower'], 3).tolist()} | {np.round(f['simulated']['higher'], 3).tolist()} |")
    lines.append(f"\n**Selected:** w_p = {selected['w_p']}, sigma_r = {selected['sigma_r']}.")
    (out / "calibration.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
