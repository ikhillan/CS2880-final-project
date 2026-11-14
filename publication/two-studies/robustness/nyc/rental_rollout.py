"""Partial market-wide adoption of cached housing recommendations on original stage-1 draws.

Use the complete development catalogs, renter profiles and archived recommendation cache
described in the upstream pilot. Outputs contain no individual-level information.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROBABILITIES = np.array([0, 0.25, 0.5, 0.75, 1])
STATS = [
    "placed",
    "covered",
    "gains",
    "losses",
    "fixed_placed",
    "fixed_gains",
    "fixed_losses",
]
GROUPS = ("all", "lower_income", "higher_income")
SCENARIOS = {
    "calibrated": {},
    "income_blind_listing_lottery": {"w_income": 0.0},
    "common_income_priority": {"common_priority": True},
    "weak_price_penalty": {"w_p": 1.0},
    "larger_taste_shock": {"sigma_r": 1.25},
}


def coverage(full, display):
    covered = np.zeros(len(full), dtype=bool)
    i = np.nonzero(full >= 0)[0]
    covered[i] = display[i, full[i]]
    return covered


def outcomes(match, hit, previous, fixed):
    placed = match >= 0
    new = placed & ~previous
    lost = previous & ~placed
    return np.array(
        [
            placed.sum(),
            hit.sum(),
            new.sum(),
            lost.sum(),
            placed[fixed].sum(),
            new[fixed].sum(),
            lost[fixed].sum(),
        ],
        dtype=int,
    )


def market_seed(identifier):
    payload = ("NYC-partial-rollout-v1|" + identifier).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def simulate(market, a, b, pairs):
    from hiddenoptions.matching import deferred_acceptance

    inst = market.inst
    full, _ = deferred_acceptance(inst, inst.E)
    assign_a, _ = deferred_acceptance(inst, a)
    assign_b, _ = deferred_acceptance(inst, b)
    covered_a, covered_b = coverage(full, a), coverage(full, b)
    same = np.all(a == b, axis=1)
    original = assign_a >= 0
    results = np.zeros((pairs, 2, len(PROBABILITIES), len(STATS)), dtype=np.int16)
    results[:, :, 0, :] = outcomes(assign_a, covered_a, original, same)
    results[:, :, -1, :] = outcomes(assign_b, covered_b, original, same)
    rng = np.random.default_rng(market_seed(market.spec.market_id))
    midpoint_own = np.empty(pairs, dtype=int)
    for r, u in enumerate(rng.random((pairs, inst.n_b))):
        for ip, p in enumerate(PROBABILITIES[1:-1], 1):
            for orientation, treated in enumerate((u < p, u >= 1 - p)):
                graph = np.where(treated[:, None], b, a)
                match, _ = deferred_acceptance(inst, graph)
                hit = np.where(treated, covered_b, covered_a)
                results[r, orientation, ip] = outcomes(match, hit, original, same)
                if p == 0.5:
                    if orientation == 0:
                        first_placed = match >= 0
                        first_treated = treated
                    else:
                        signed_change = first_placed.astype(int) - (match >= 0)
                        midpoint_own[r] = (
                            (2 * first_treated[~same] - 1) * signed_change[~same]
                        ).sum()
    assert np.array_equal(
        results[..., 0] - results[:, :, 0:1, 0], results[..., 2] - results[..., 3]
    )
    assert np.array_equal(
        results[..., 4] - results[:, :, 0:1, 4], results[..., 5] - results[..., 6]
    )
    # Complementary masks at midpoint give exactly affine coverage in each pair.
    assert np.all(results[:, :, 2, 1].sum(1) == results[0, 0, (0, -1), 1].sum())
    return results, int(same.sum()), midpoint_own


def sensitivity(market, a, b, catalog):
    from hiddenoptions.markets import build_market
    from hiddenoptions.matching import deferred_acceptance

    low = market.renters.lower_income.to_numpy(bool)
    groups = [np.ones(len(low), bool), low, ~low]
    rows = []
    for label, parameters in SCENARIOS.items():
        altered = (
            market
            if not parameters
            else build_market(replace(market.spec, **parameters), catalog)
        )
        # These are the original menu masks, not new draws under new scenario IDs.
        np.testing.assert_array_equal(altered.renters.serialno, market.renters.serialno)
        np.testing.assert_array_equal(
            altered.listings.listing_uid, market.listings.listing_uid
        )
        np.testing.assert_array_equal(altered.inst.E, market.inst.E)
        if label in ("common_income_priority", "income_blind_listing_lottery"):
            np.testing.assert_array_equal(altered.inst.rank_b, market.inst.rank_b)
        if label in ("weak_price_penalty", "larger_taste_shock"):
            np.testing.assert_array_equal(altered.inst.rank_s, market.inst.rank_s)
        full, _ = deferred_acceptance(altered.inst, altered.inst.E)
        base, _ = deferred_acceptance(altered.inst, a)
        haiku, _ = deferred_acceptance(altered.inst, b)
        hits_a, hits_b = coverage(full, a), coverage(full, b)
        for group, mask in zip(GROUPS, groups):
            rows.append(
                dict(
                    scenario=label,
                    group=group,
                    n=int(mask.sum()),
                    covered_a=int(hits_a[mask].sum()),
                    covered_b=int(hits_b[mask].sum()),
                    placed_a=int((base[mask] >= 0).sum()),
                    placed_b=int((haiku[mask] >= 0).sum()),
                )
            )
    return rows


def weighted_rollout(market_pairs, meta, weights, own_effect):
    """Conditional MC standard errors and a separate whole-seed-block bootstrap."""
    stats = np.asarray(market_pairs).mean(
        axis=2
    )  # market x antithetic pair x p x count
    rows = []
    for design, panel in meta.groupby("design"):
        panel = panel.sort_values(["seed", "borough", "month"])
        seeds = panel.seed.unique()
        cells = sorted(set(zip(panel.borough, panel.month)))
        np.testing.assert_array_equal(
            panel[["seed", "borough", "month"]].to_records(index=False),
            pd.DataFrame(
                [(s, b, m) for s in seeds for b, m in cells],
                columns=["seed", "borough", "month"],
            ).to_records(index=False),
        )
        w = np.array([weights[cell] for cell in cells])
        w /= w.sum()
        data = stats[panel.index].reshape(len(seeds), len(cells), *stats.shape[1:])
        # All markets have 64 applicants so these population rates are linear in market counts.
        w_by_pair = np.einsum("c,scrpj->rpj", w, data) * (100 / (64 * len(seeds)))
        w_by_seed = np.einsum("c,scrpj->spj", w, data) * (100 / (64 * data.shape[2]))
        bootstrap = np.random.default_rng(0).integers(
            0, len(seeds), size=(10000, len(seeds))
        )
        own = np.asarray(own_effect)[panel.index].reshape(len(seeds), len(cells), -1)
        own_per_pair = 100 * np.einsum("c,scr->r", w, own) / (len(seeds) * 64)
        own_per_block = 100 * np.einsum("c,scr->s", w, own) / (own.shape[2] * 64)
        lo, hi = np.quantile(own_per_block[bootstrap].mean(axis=1), [0.025, 0.975])
        rows.append(
            dict(
                design=design,
                p=0.5,
                metric="own_list_effect",
                rate_pct=own_per_pair.mean(),
                delta_pp=own_per_pair.mean(),
                mean_market_count=own.mean(),
                pair_mc_se_pp=own_per_pair.std(ddof=1) / np.sqrt(len(own_per_pair)),
                interpolation_residual_pp=None,
                residual_mc_se_pp=None,
                block_delta_lo_pp=lo,
                block_delta_hi_pp=hi,
                block_residual_lo_pp=None,
                block_residual_hi_pp=None,
                markets=len(panel),
                people=panel.n.sum(),
                same_menu_people=panel.same.sum(),
                pairs=len(own_per_pair),
            )
        )
        for name, label in [
            ("placed", "placed"),
            ("covered", "covered"),
            ("fixed_placed", "fixed_placed"),
            ("gains", "gains"),
            ("losses", "losses"),
            ("fixed_gains", "fixed_gains"),
            ("fixed_losses", "fixed_losses"),
        ]:
            j = STATS.index(name)
            base = w_by_pair[:, 0, j]
            end = w_by_pair[:, -1, j]
            seed_values = w_by_seed[:, :, j]
            for ip, p in enumerate(PROBABILITIES):
                draws = w_by_pair[:, ip, j]
                delta = draws - base
                excess = draws - ((1 - p) * base + p * end)
                sd = seed_values[:, ip] - seed_values[:, 0]
                se = seed_values[:, ip] - (
                    (1 - p) * seed_values[:, 0] + p * seed_values[:, -1]
                )
                lo, hi = np.quantile(sd[bootstrap].mean(axis=1), [0.025, 0.975])
                exlo, exhi = np.quantile(se[bootstrap].mean(axis=1), [0.025, 0.975])
                rows.append(
                    dict(
                        design=design,
                        p=p,
                        metric=label,
                        rate_pct=draws.mean(),
                        delta_pp=delta.mean(),
                        mean_market_count=data[:, :, :, ip, j].mean(),
                        pair_mc_se_pp=draws.std(ddof=1) / np.sqrt(len(draws)),
                        interpolation_residual_pp=excess.mean(),
                        residual_mc_se_pp=excess.std(ddof=1) / np.sqrt(len(draws)),
                        block_delta_lo_pp=lo,
                        block_delta_hi_pp=hi,
                        block_residual_lo_pp=exlo,
                        block_residual_hi_pp=exhi,
                        markets=len(panel),
                        people=panel.n.sum(),
                        same_menu_people=panel.same.sum(),
                        pairs=len(draws),
                    )
                )
    return pd.DataFrame(rows)


def sensitivity_summary(values, weights):
    results = []
    for (design, scenario, group), panel in values.groupby(
        ["design", "scenario", "group"]
    ):
        cols = ["n", "covered_a", "covered_b", "placed_a", "placed_b"]
        seeds = np.sort(panel.seed.unique())
        cells = sorted(set(zip(panel.borough, panel.month)))
        ordered = panel.set_index(["seed", "borough", "month"]).loc[
            [(s, b, m) for s in seeds for b, m in cells], cols
        ]
        cube = ordered.to_numpy().reshape(len(seeds), len(cells), len(cols))
        w = np.array([weights[c] for c in cells])
        w /= w.sum()
        rng = np.random.default_rng(0)
        draws = rng.integers(0, len(seeds), size=(10000, len(seeds)))
        frequencies = np.zeros((len(draws), len(seeds)), int)
        np.add.at(frequencies, (np.arange(len(draws))[:, None], draws), 1)
        frequencies = np.vstack([np.ones(len(seeds), int), frequencies])
        agg = (frequencies @ cube.reshape(len(seeds), -1)).reshape(
            len(frequencies), len(cells), len(cols)
        )
        assert (agg[:, :, 0] > 0).all()
        rates = np.einsum("c,qcj->qj", w, agg[:, :, 1:] / agg[:, :, 0, None]) * 100
        for j, name in [(0, "covered"), (2, "placed")]:
            change = rates[:, j + 1] - rates[:, j]
            lo, hi = np.quantile(change[1:], [0.025, 0.975])
            results.append(
                dict(
                    design=design,
                    scenario=scenario,
                    group=group,
                    metric=name,
                    n=int(panel.n.sum()),
                    a_pct=rates[0, j],
                    b_pct=rates[0, j + 1],
                    delta_pp=change[0],
                    block_lo_pp=lo,
                    block_hi_pp=hi,
                )
            )
    return pd.DataFrame(results)


def main(repo, out, pairs):
    sys.path.insert(0, str(repo.resolve()))
    from experiments.config import expand_labeled, load_config
    from experiments.analyze import cell_weights
    from experiments.pilot_outcomes import (
        CONFIG,
        DEFAULT_CACHE,
        load_pilot_cache,
        pilot_displays,
    )
    from hiddenoptions.markets import Catalog, InsufficientSupply, build_market

    catalog = Catalog()
    config = expand_labeled(load_config(CONFIG))
    cache = load_pilot_cache(DEFAULT_CACHE)
    weights = cell_weights(sorted({(s.borough, s.month) for s, _ in config}))
    out.mkdir(parents=True, exist_ok=True)
    meta = []
    pair_data = []
    own_effect = []
    scenario_data = []
    omitted = Counter()
    for spec, design in config:
        try:
            market = build_market(spec, catalog)
        except InsufficientSupply:
            omitted[(design, spec.borough)] += 1
            continue
        a, b = pilot_displays(market, cache)
        counts, same, own = simulate(market, a, b, pairs)
        own_effect.append(own)
        identity = dict(
            design=design, seed=spec.seed, borough=spec.borough, month=spec.month
        )
        meta.append(dict(**identity, n=market.inst.n_b, same=same))
        pair_data.append(counts)
        scenario_data.extend(
            dict(**identity, **row) for row in sensitivity(market, a, b, catalog)
        )
    assert (
        omitted == {("primary", "Staten Island"): 60, ("enriched", "Staten Island"): 60}
        and len(meta) == 480
    )
    meta = pd.DataFrame(meta)
    rollout = weighted_rollout(pair_data, meta, weights, own_effect)
    alternative = pd.DataFrame(scenario_data)
    sensitivities = sensitivity_summary(alternative, weights)
    archived = pd.read_csv(repo / "publication/two-studies/evidence/policy-rates.csv")
    archived = archived[archived.comparison == "haiku_vs_uniform"].set_index(
        ["design", "group", "metric"]
    )
    for record in sensitivities[sensitivities.scenario == "calibrated"].itertuples():
        orig = archived.loc[record.design, record.group, record.metric]
        np.testing.assert_allclose(
            [
                record.a_pct,
                record.b_pct,
                record.delta_pp,
                record.block_lo_pp,
                record.block_hi_pp,
            ],
            [orig.a_pct, orig.b_pct, orig.delta_pp, orig.lo_pp, orig.hi_pp],
            atol=1e-10,
            rtol=0,
        )
    for row in rollout[
        (rollout.p == 1) & (rollout.metric.isin(["covered", "placed"]))
    ].itertuples():
        orig = archived.loc[row.design, "all", row.metric]
        np.testing.assert_allclose(
            [row.rate_pct, row.delta_pp],
            [orig.b_pct, orig.delta_pp],
            atol=1e-11,
            rtol=0,
        )
    identity = meta.loc[meta.index.repeat(pairs * 2 * len(PROBABILITIES))].reset_index(
        drop=True
    )
    identity["pair"] = np.tile(
        np.repeat(np.arange(pairs), 2 * len(PROBABILITIES)), len(meta)
    )
    identity["orientation"] = np.tile(
        np.repeat([0, 1], len(PROBABILITIES)), len(meta) * pairs
    )
    identity["p"] = np.tile(PROBABILITIES, len(meta) * pairs * 2)
    hierarchy = pd.concat(
        [
            identity,
            pd.DataFrame(np.asarray(pair_data).reshape(-1, len(STATS)), columns=STATS),
        ],
        axis=1,
    )
    midpoint = meta.loc[
        meta.index.repeat(pairs), ["design", "seed", "borough", "month", "n", "same"]
    ].reset_index(drop=True)
    midpoint["pair"] = np.tile(np.arange(pairs), len(meta))
    midpoint["own_placement_contrast"] = np.asarray(own_effect).ravel()
    midpoint.to_csv(
        out / "rental-midpoint-own.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    hierarchy.to_csv(
        out / "rental-rollout-counts.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    rollout.to_csv(out / "rental-rollout.csv", index=False)
    alternative.to_csv(
        out / "rental-sensitivity-counts.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    sensitivities.to_csv(out / "rental-sensitivity.csv", index=False)
    print(
        rollout[rollout.metric == "placed"][
            [
                "design",
                "p",
                "delta_pp",
                "interpolation_residual_pp",
                "pair_mc_se_pp",
                "block_delta_lo_pp",
                "block_delta_hi_pp",
            ]
        ].to_string(index=False)
    )
    print(
        sensitivities[sensitivities.group == "all"][
            ["design", "scenario", "metric", "delta_pp", "block_lo_pp", "block_hi_pp"]
        ].to_string(index=False)
    )
    print(
        "Validated",
        len(meta),
        "markets and",
        len(hierarchy),
        "replicate-level aggregate rows.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=64)
    args = parser.parse_args()
    if args.pairs < 2:
        parser.error(
            "--pairs needs at least two independent pairs to estimate MC error"
        )
    main(args.repo, args.out, args.pairs)
