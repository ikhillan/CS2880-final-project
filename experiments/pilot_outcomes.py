"""Reconstruct paired renter outcomes for the archived stage-1 Haiku pilot.

Run from the repository root with the stage-1 development catalogs prepared:
    python -m experiments.pilot_outcomes --out results/haiku-followup
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.analyze import cell_weights, interval, joint_rates
from experiments.config import expand_labeled, load_config
from hiddenoptions.exposure import LlmRecommendations, uniform_exposure
from hiddenoptions.markets import Catalog, InsufficientSupply, build_market
from hiddenoptions.matching import blocking_mask, deferred_acceptance

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "findings" / "claude-haiku-pilot"
DEFAULT_CACHE = BASE / "claude_recs.jsonl.gz"
CONFIG = ROOT / "configs" / "stage1_pilot.json"
ARM = "claude_k5"
GROUPS = ("lower_income", "higher_income")
METRICS = ("hidden_blocker", "placed", "partner_shown")


def load_pilot_cache(path: Path) -> LlmRecommendations:
    """Read the original pilot under its documented no-failure, single-reply design."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    cache = LlmRecommendations()
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            row = json.loads(line)
            if (row["exposure"] != ARM or row["rep"] != 0 or row["count"] != 5
                    or row["model"] != "claude-haiku-5-5" or row["status"] != "ok"
                    or row["raw_valid_ids"] != 5 or any(row[k] for k in
                        ("invalid_ids", "duplicate_ids", "extra_ids"))):
                raise ValueError(f"Unexpected response settings or status on line {line_no}")
            picks = row["listings"]
            if (not isinstance(picks, list) or len(picks) != 5
                    or not all(type(j) is int and j >= 0 for j in picks)
                    or len(set(picks)) != 5 or type(row["renter"]) is not int):
                raise ValueError(f"Invalid pilot shortlist or renter index on line {line_no}")
            rows = cache.table.setdefault(row["market_id"], {}).setdefault(ARM, {})
            key = (row["renter"], 0)
            if key in rows:
                raise ValueError(f"Duplicate pilot reply on line {line_no}")
            rows[key] = row
    return cache


def pilot_displays(market, cache: LlmRecommendations) -> tuple[np.ndarray, np.ndarray]:
    """Return the two realized displays only if every needed reply matches feasibility."""
    mid = market.spec.market_id
    feasible = market.inst.E
    rows = cache.table.get(mid, {}).get(ARM, {})
    required = {(int(i), 0) for i in np.flatnonzero(feasible.sum(1) > 5)}
    if set(rows) != required:
        raise ValueError(f"Cache and required renter rows differ in {mid}")
    for (i, _), row in rows.items():
        ids = row["listings"]
        if max(ids) >= feasible.shape[1] or not feasible[i, ids].all():
            raise ValueError(f"Cache has a pick outside the candidate set in {mid}, renter {i}")
    haiku, _ = cache.exposure(market, ARM, k=5)
    return uniform_exposure(market, k=5), haiku


def policy_events(inst, display, full_partners):
    assigned, holders = deferred_acceptance(inst, display)
    blockers = blocking_mask(inst, assigned, holders)
    if (blockers & display).any():
        raise AssertionError("A displayed pair blocks the market's own DA outcome")
    partner_shown = np.zeros(inst.n_b, dtype=bool)
    placed_full = np.flatnonzero(full_partners >= 0)
    partner_shown[placed_full] = display[placed_full, full_partners[placed_full]]
    return {"placed": assigned >= 0, "hidden_blocker": blockers.any(axis=1),
            "partner_shown": partner_shown}


def paired_counts(cache):
    specs = expand_labeled(load_config(CONFIG))
    catalog = Catalog()
    records = []
    seen = set()
    skipped = 0
    for spec, design in specs:
        try:
            market = build_market(spec, catalog)
        except InsufficientSupply:
            skipped += 1
            continue
        seen.add(spec.market_id)
        uniform, haiku = pilot_displays(market, cache)
        full, _ = deferred_acceptance(market.inst, market.inst.E)
        old = policy_events(market.inst, uniform, full)
        new = policy_events(market.inst, haiku, full)
        lower = market.renters.lower_income.to_numpy(bool)
        for group, members in zip(GROUPS, (lower, ~lower)):
            for metric in METRICS:
                before, after = old[metric][members], new[metric][members]
                records.append({"variant": design, "seed": spec.seed, "borough": spec.borough,
                                "month": spec.month, "group": group, "metric": metric,
                                "n": len(before), "uniform": before.sum(), "haiku": after.sum(),
                                "gained_event": (~before & after).sum(),
                                "lost_event": (before & ~after).sum()})
    unused = set(cache.table) - seen
    if unused:
        raise ValueError(f"{len(unused)} unexpected pilot markets in the cache")
    if len(seen) != 480 or skipped != 120:
        raise ValueError(f"Expected 480 estimable and 120 unestimable markets, got {len(seen)}, {skipped}")
    planned = cell_weights(sorted({(s.borough, s.month) for s, _ in specs}))
    return pd.DataFrame(records), planned


def summarise(frame, weights, reps):
    summaries = []
    arms = ["uniform", "haiku"]
    original_t = pd.read_csv(BASE / "contrast_T.csv")
    for (design, metric), data in frame.groupby(["variant", "metric"]):
        counts = data.melt(id_vars=["seed", "borough", "month", "group", "n"],
                           value_vars=arms, var_name="arm", value_name="count")
        rates, draws, covered, cells = joint_rates(counts, weights, arms, reps)
        delta = rates[1] - rates[0]
        changes = draws[:, 1] - draws[:, 0]
        if metric == "hidden_blocker":
            reference = original_t.loc[(original_t.variant == design) & (original_t.renters == "all")].iloc[0]
            expected = [[reference[f"B_{g}_{a}"] for g in GROUPS] for a in ("uniform_k5", ARM)]
            np.testing.assert_allclose(rates, expected, rtol=0, atol=1e-12)
            if reps == 10000:
                np.testing.assert_allclose(100 * np.array(interval(changes[:, 0] - changes[:, 1])),
                                           [reference.T_lo_pp, reference.T_hi_pp], rtol=0, atol=1e-10)
        for g, group in enumerate(GROUPS):
            rows = data[data.group == group]
            lo, hi = interval(changes[:, g])
            totals = rows[["n", "uniform", "haiku", "gained_event", "lost_event"]].sum()
            if totals.haiku - totals.uniform != totals.gained_event - totals.lost_event:
                raise AssertionError("Inconsistent switch counts")
            summaries.append({
                "variant": design, "group": group, "metric": metric,
                "n_renter_instances": int(totals.n), "uniform_events": int(totals.uniform),
                "haiku_events": int(totals.haiku), "gained_event": int(totals.gained_event),
                "lost_event": int(totals.lost_event), "uniform_pct": 100 * rates[0, g],
                "haiku_pct": 100 * rates[1, g], "difference_pp": 100 * delta[g],
                "lo_pp": 100 * lo, "hi_pp": 100 * hi, "draws": reps,
                "valid_draws": int(np.isfinite(changes[:, g]).sum()), "seed_blocks": data.seed.nunique(),
                "markets": len(rows), "supported_cells": cells, "covered_weight": covered})
    return pd.DataFrame(summaries)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "haiku-followup")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--reps", type=int, default=10000)
    args = ap.parse_args()
    if args.reps < 1:
        ap.error("--reps must be positive")
    counts, weights = paired_counts(load_pilot_cache(args.cache))
    summary = summarise(counts, weights, args.reps)
    args.out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out / "paired-outcomes.csv", index=False)
    counts.to_csv(args.out / "paired-outcomes-counts.csv", index=False)
    print(f"Paired results and cell/seed counts written to {args.out}")


if __name__ == "__main__":
    main()
