"""Who can use a vacancy after the archived five-listing pilot? One seat per listing."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from experiments.config import expand_labeled, load_config
from hiddenoptions.markets import Catalog, InsufficientSupply, build_market, maximum_matching_size
from hiddenoptions.matching import deferred_acceptance
from hiddenoptions.metrics import full_list_rank

ROOT = Path(__file__).resolve().parents[1]


def extreme_extension(edges, lower_income, favor_lower):
    """Among largest extensions, maximize the number from the favored income group."""
    n = edges.shape[0]
    favored = lower_income if favor_lower else ~lower_income
    # A cardinality bonus of n+1 dominates every possible change in the group count.
    # A zero-weight nonedge pads a partial matching to an assignment and is discarded.
    score = edges * (n + 1 + favored[:, None])
    rows, cols = linear_sum_assignment(score, maximize=True)
    valid = edges[rows, cols]
    return rows[valid], cols[valid]


def assess(market, display, rank_limits=(5, None)):
    inst = market.inst
    lower = market.renters.lower_income.to_numpy(bool)
    before, held = deferred_acceptance(inst, display)
    on_market = before >= 0
    base_rank = full_list_rank(inst, before)
    vacant_edges = inst.E & ~on_market[:, None] & (held < 0)[None, :]
    if (vacant_edges & display).any():
        raise ValueError("An unmatched renter and an available listing form a visible blocking pair.")
    for limit in rank_limits:
        edges = vacant_edges if limit is None else vacant_edges & (inst.rank_b < limit)
        maximum = maximum_matching_size(edges)
        low_counts = []
        for favor_lower in (False, True):
            renters, units = extreme_extension(edges, lower, favor_lower)
            if len(renters) != maximum:
                raise AssertionError("Extension size differs from independent maximum matching.")
            revised = display.copy()
            if len(renters):
                # All selected renters have a hidden option and a full, positive quota.
                remove = np.argmax(np.where(display[renters], inst.rank_b[renters], -1), axis=1)
                revised[renters, remove] = False
                revised[renters, units] = True
            result, _ = deferred_acceptance(inst, revised)
            expected = on_market.copy()
            expected[renters] = True
            if not (np.array_equal(revised.sum(1), display.sum(1))
                    and np.array_equal(result >= 0, expected)
                    and np.array_equal(result[renters], units)
                    and np.all(full_list_rank(inst, result)[on_market] <= base_rank[on_market])):
                raise AssertionError("Named referral, quota, recipient set or incumbent rank check failed.")
            low_counts.append(int(lower[renters].sum()))
        yield {
            "rank_limit": "any" if limit is None else str(limit),
            "markets": 1, "renters": len(lower), "lower_income": int(lower.sum()),
            "baseline_placed": int(on_market.sum()), "baseline_low_placed": int((on_market & lower).sum()),
            "vacant": int((held < 0).sum()), "eligible": int(edges.any(1).sum()),
            "eligible_lower": int((edges.any(1) & lower).sum()), "maximum": maximum,
            "min_lower_at_max": low_counts[0], "max_lower_at_max": low_counts[1],
        }


def main(output):
    from experiments.pilot_outcomes import DEFAULT_CACHE, load_pilot_cache, pilot_displays

    cfg = load_config(ROOT / "configs/stage1_pilot.json")
    cache = load_pilot_cache(DEFAULT_CACHE)
    catalog = Catalog()
    rows = []
    skipped = Counter()
    for spec, design in expand_labeled(cfg):
        try:
            market = build_market(spec, catalog)
        except InsufficientSupply:
            skipped[design, spec.borough] += 1
            continue
        uniform, haiku = pilot_displays(market, cache)
        for policy, display in [("uniform", uniform), ("haiku", haiku)]:
            for row in assess(market, display):
                rows.append({"design": design, "policy": policy, **row})
    summary = pd.DataFrame(rows).groupby(["design", "policy", "rank_limit"], as_index=False).sum()
    if not (len(summary) == 8 and (summary.markets == 240).all()
            and skipped == {("primary", "Staten Island"): 60, ("enriched", "Staten Island"): 60}):
        raise ValueError(f"Unexpected support for the archived pilot: {skipped}")
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "findings/claude-haiku-pilot/referrals.csv")
    main(parser.parse_args().output)
