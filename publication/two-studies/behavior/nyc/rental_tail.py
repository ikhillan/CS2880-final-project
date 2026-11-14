"""Append the same recovered listing after a household's existing shortlist."""

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from rental_advice import plus_one, prefix_hits


def suffixes(inst, original, extended):
    ordered = [
        [j for j in ranks if original[i, j]] for i, ranks in enumerate(inst.prefs_b)
    ]
    new = extended & ~original
    for i, ranks in enumerate(inst.prefs_b):
        ordered[i].extend(j for j in ranks if new[i, j])
    return ordered


def continue_da(inst, ordered, original, old, previous_holders):
    """Resume serial DA with all old unmatched applicants free, same source of priorities."""
    partner = old.copy()
    owners = previous_holders.copy()
    position = original.sum(1).copy()
    for i in np.flatnonzero(old >= 0):
        position[i] = ordered[i].index(old[i]) + 1
    active = list(np.flatnonzero(partner < 0))
    while active:
        i = active.pop()
        while position[i] < len(ordered[i]):
            j = ordered[i][position[i]]
            position[i] += 1
            incumbent = owners[j]
            if incumbent >= 0 and inst.rank_s[i, j] > inst.rank_s[incumbent, j]:
                continue
            if incumbent >= 0:
                partner[incumbent] = -1
                active.append(incumbent)
            partner[i], owners[j] = j, i
            break
    return partner, owners


def outcomes(inst, original, extended, full, synchronous):
    from hiddenoptions.matching import blocking_mask, deferred_acceptance

    old, old_holders = deferred_acceptance(inst, original)
    ordered = suffixes(inst, original, extended)
    match, holders = continue_da(inst, ordered, original, old, old_holders)
    effective_rank = np.full_like(inst.rank_b, 1000000)
    for i, prefs in enumerate(ordered):
        effective_rank[i, prefs] = np.arange(len(prefs), dtype=int)
    # Completely fresh synchronous DA, starting with every applicant free.
    np.testing.assert_array_equal(
        match, synchronous(effective_rank, inst.rank_s, extended)
    )
    placed = match >= 0
    assert (holders[old_holders >= 0] >= 0).all()
    entering = np.flatnonzero((old < 0) & placed)
    assert np.all((extended & ~original)[entering, match[entering]])
    exiting = np.flatnonzero((old >= 0) & ~placed)
    assert len(entering) - len(exiting) == ((old_holders < 0) & (holders >= 0)).sum()
    return (
        dict(
            covered=int(extended[np.flatnonzero(full >= 0), full[full >= 0]].sum()),
            first=prefix_hits(inst, match),
            placed=int(placed.sum()),
            enters=len(entering),
            exits=len(exiting),
            any_original_blocker=int(blocking_mask(inst, match, holders).any(1).sum()),
        ),
        match,
        effective_rank,
    )


def witness_paths(start, end, order_start, order_end, listing_rank):
    """Trace each original recipient who exits to a distinct matched applicant who reverses the two partners."""
    occupants = np.full(listing_rank.shape[1], -1, dtype=int)
    placed = np.flatnonzero(end >= 0)
    occupants[end[placed]] = placed
    terminals = set()
    path_lengths = []
    for origin in np.flatnonzero((start >= 0) & (end < 0)):
        applicant = origin
        visited = set()
        while True:
            assert applicant not in visited
            visited.add(applicant)
            former = start[applicant]
            if end[applicant] >= 0:
                assert (
                    order_end[applicant, former] < order_end[applicant, end[applicant]]
                )
            other = occupants[former]
            assert other >= 0 and other != applicant
            assert listing_rank[other, former] < listing_rank[applicant, former]
            assert (
                start[other] >= 0
                and order_start[other, start[other]] < order_start[other, former]
            )
            if order_end[other, former] < order_end[other, start[other]]:
                assert other not in terminals
                terminals.add(other)
                path_lengths.append(len(visited))
                break
            applicant = other
    return terminals, path_lengths


def main(repo, out):
    sys.path.insert(0, str(repo))
    from experiments.analyze import cell_weights
    from experiments.config import expand_labeled, load_config
    from experiments.pilot_outcomes import (
        CONFIG,
        DEFAULT_CACHE,
        load_pilot_cache,
        pilot_displays,
    )
    from hiddenoptions.markets import build_market, InsufficientSupply, Catalog
    from hiddenoptions.matching import deferred_acceptance

    path = repo / "publication/two-studies/audit/nyc-temporal/verify_temporal.py"
    spec = importlib.util.spec_from_file_location("independent_da", path)
    aux = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(aux)
    config = expand_labeled(load_config(CONFIG))
    cache = load_pilot_cache(DEFAULT_CACHE)
    catalog = Catalog()
    weights = cell_weights(sorted({(s.borough, s.month) for s, _ in config}))
    rows = []
    witness_rows = []
    for s, design in config:
        try:
            market = build_market(s, catalog)
        except InsufficientSupply:
            assert s.borough == "Staten Island"
            continue
        inst = market.inst
        full, _ = deferred_acceptance(inst, inst.E)
        a, b = pilot_displays(market, cache)
        for scenario, ext_a, ext_b in [
            ("one", plus_one(inst, a), plus_one(inst, b)),
            ("all", inst.E, inst.E),
        ]:
            stats_a, match_a, ranks_a = outcomes(inst, a, ext_a, full, aux.synchronous)
            stats_b, match_b, ranks_b = outcomes(inst, b, ext_b, full, aux.synchronous)
            same = (ranks_a == ranks_b).all(1)
            same_sets = (ext_a == ext_b).all(1)
            assert np.all(~same | same_sets)
            gain, loss = (match_a < 0) & (match_b >= 0), (match_a >= 0) & (match_b < 0)
            if scenario == "all":
                both = np.flatnonzero(
                    (match_a >= 0) & (match_b >= 0) & (match_a != match_b)
                )
                reversal = both[
                    (ranks_a[both, match_a[both]] < ranks_a[both, match_b[both]])
                    & (ranks_b[both, match_b[both]] < ranks_b[both, match_a[both]])
                ]
                old_w, old_lengths = witness_paths(
                    match_a, match_b, ranks_a, ranks_b, inst.rank_s
                )
                new_w, new_lengths = witness_paths(
                    match_b, match_a, ranks_b, ranks_a, inst.rank_s
                )
                assert old_w.issubset(reversal.tolist()) and new_w.issubset(
                    reversal.tolist()
                )
                assert len(old_w) == loss.sum() and len(new_w) == gain.sum()
                witness_rows.append(
                    dict(
                        design=design,
                        seed=s.seed,
                        borough=s.borough,
                        month=s.month,
                        people=inst.n_b,
                        source_losses=len(old_w),
                        source_gains=len(new_w),
                        reversals=len(reversal),
                        shared_witnesses=len(old_w & new_w),
                        loss_path_steps=sum(old_lengths),
                        gain_path_steps=sum(new_lengths),
                        max_loss_path=max(old_lengths, default=0),
                        max_gain_path=max(new_lengths, default=0),
                    )
                )
            rec = dict(
                design=design,
                scenario=scenario,
                seed=s.seed,
                borough=s.borough,
                month=s.month,
                cell_weight=weights[s.borough, s.month],
                people=inst.n_b,
                gain=int(gain.sum()),
                loss=int(loss.sum()),
                same_menu=int(same.sum()),
                same_set=int(same_sets.sum()),
                same_gain=int((same & gain).sum()),
                same_loss=int((same & loss).sum()),
            )
            for key in stats_a:
                rec[key + "_a"], rec[key + "_b"] = stats_a[key], stats_b[key]
                rec["delta_" + key] = stats_b[key] - stats_a[key]
            rows.append(rec)
    assert len(rows) == 960
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(witness_rows).to_csv(out / "nyc-full-tail-witnesses.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out / "nyc-tail-counts.csv", index=False)
    metrics = [
        c
        for c in df
        if c
        not in (
            "design",
            "scenario",
            "seed",
            "borough",
            "month",
            "cell_weight",
            "people",
        )
    ]
    summary = []
    for (design, scenario), panel in df.groupby(["design", "scenario"]):
        byseed = []
        for _, g in panel.groupby("seed"):
            byseed.append(
                np.average(g[metrics], weights=g.cell_weight, axis=0) * 100 / 64
            )
        byseed = np.asarray(byseed)
        bootstrap = np.random.default_rng(0).integers(
            0, len(byseed), size=(10000, len(byseed))
        )
        ci = np.quantile(byseed[bootstrap].mean(1), [0.025, 0.975], axis=0)
        for k, key in enumerate(metrics):
            summary.append(
                dict(
                    design=design,
                    scenario=scenario,
                    metric=key,
                    weighted_per100=byseed[:, k].mean(),
                    block_lo=ci[0, k],
                    block_hi=ci[1, k],
                    pooled_count=int(panel[key].sum()),
                    people=int(panel.people.sum()),
                )
            )
    results = pd.DataFrame(summary)
    ranked = (
        pd.read_csv(out / "nyc-advice-results.csv")
        .query('scenario=="retrieve_best_omitted"')
        .set_index(["design", "metric"])
    )
    for row in results.itertuples():
        if row.scenario == "all" and row.metric in ("same_set", "delta_covered"):
            expected = 100 if row.metric == "same_set" else 0
            np.testing.assert_allclose(
                [row.weighted_per100, row.block_lo, row.block_hi],
                expected,
                rtol=0,
                atol=1e-10,
            )
        if row.scenario == "one" and row.metric in (
            "delta_covered",
            "covered_a",
            "covered_b",
        ):
            ref = ranked.loc[
                row.design,
                "delta_coverage" if row.metric == "delta_covered" else row.metric,
            ]
            np.testing.assert_allclose(
                [row.weighted_per100, row.block_lo, row.block_hi, row.pooled_count],
                [ref.weighted_per100, ref.block_lo, ref.block_hi, ref.pooled_count],
                rtol=0,
                atol=1e-10,
            )
    results.to_csv(out / "nyc-tail-results.csv", index=False)
    print(
        results[
            results.metric.isin(
                [
                    "delta_covered",
                    "delta_first",
                    "delta_placed",
                    "same_loss",
                    "exits_a",
                    "exits_b",
                    "enters_a",
                    "enters_b",
                ]
            )
        ].to_string(index=False)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    main(args.repo.resolve(), args.out.resolve())
