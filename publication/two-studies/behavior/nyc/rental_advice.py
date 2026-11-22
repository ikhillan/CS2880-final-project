"""Paired NYC replays when renters may pursue options outside their recommendations."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROBABILITIES = (0, 0.25, 0.5, 0.75, 1)
SCENARIOS = [
    "adhere_0",
    "adhere_25",
    "adhere_50",
    "adhere_75",
    "adhere_100",
    "retrieve_best_omitted",
]
COUNTS = [
    "covered_a",
    "covered_b",
    "first_a",
    "first_b",
    "placed_a",
    "placed_b",
    "gain",
    "loss",
    "same_menu",
    "same_gain",
    "same_loss",
]
CONTRASTS = [
    ("delta_coverage", "covered_a", "covered_b"),
    ("delta_first", "first_a", "first_b"),
    ("delta_placed", "placed_a", "placed_b"),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plus_one(inst, menu):
    expanded = menu.copy()
    for person, preferences in enumerate(inst.prefs_b):
        for option in preferences:
            if not menu[person, option]:
                expanded[person, option] = True
                break
    np.testing.assert_array_equal(
        expanded.sum(1), np.minimum(menu.sum(1) + 1, inst.E.sum(1))
    )
    return expanded


def prefix_hits(inst, offer):
    indices = np.flatnonzero(offer >= 0)
    return int((inst.rank_b[indices, offer[indices]] == 0).sum())


def compare(inst, full, a, b, audit):
    from hiddenoptions.matching import deferred_acceptance, blocking_mask

    match_a, inverse_a = deferred_acceptance(inst, a)
    match_b, inverse_b = deferred_acceptance(inst, b)
    if audit:
        for menu, assignment, inverse in (
            (a, match_a, inverse_a),
            (b, match_b, inverse_b),
        ):
            np.testing.assert_array_equal(
                assignment, SYNC(inst.rank_b, inst.rank_s, menu)
            )
            assert not np.any(menu & blocking_mask(inst, assignment, inverse))
    eligible = np.flatnonzero(full >= 0)
    hit_a = int(a[eligible, full[eligible]].sum())
    hit_b = int(b[eligible, full[eligible]].sum())
    placed_a, placed_b = match_a >= 0, match_b >= 0
    gain, loss = ~placed_a & placed_b, placed_a & ~placed_b
    same = (a == b).all(1)
    return [
        hit_a,
        hit_b,
        prefix_hits(inst, match_a),
        prefix_hits(inst, match_b),
        int(placed_a.sum()),
        int(placed_b.sum()),
        int(gain.sum()),
        int(loss.sum()),
        int(same.sum()),
        int((same & gain).sum()),
        int((same & loss).sum()),
    ]


def process_market(entry):
    from experiments.pilot_outcomes import pilot_displays
    from hiddenoptions.markets import build_market, InsufficientSupply
    from hiddenoptions.matching import deferred_acceptance

    spec, design = entry
    try:
        market = build_market(spec, CATALOG)
    except InsufficientSupply:
        assert spec.borough == "Staten Island"
        return None
    inst = market.inst
    a, b = pilot_displays(market, CACHE)
    full, _ = deferred_acceptance(inst, inst.E)
    digest_bytes = hashlib.sha256(
        ("nyc-advice-adherence-v1|" + spec.market_id).encode()
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest_bytes[:16], "big"))
    data = np.empty((PAIRS, 2, len(SCENARIOS), len(COUNTS)), dtype="int16")
    audit = spec.seed == FIRST_SEED[design]
    identity_checks = 0
    for idx, left, right in (
        (0, inst.E, inst.E),
        (4, a, b),
        (5, plus_one(inst, a), plus_one(inst, b)),
    ):
        data[:, :, idx] = compare(inst, full, left, right, audit)
        identity_checks += 2 * int(audit)
    for pair, uniforms in enumerate(rng.random((PAIRS, inst.n_b))):
        for index, p in enumerate(PROBABILITIES[1:-1], 1):
            for side, adherence in enumerate((uniforms < p, uniforms >= 1 - p)):
                graph_a = np.where(adherence[:, None], a, inst.E)
                graph_b = np.where(adherence[:, None], b, inst.E)
                verify = audit and pair == 0
                data[pair, side, index] = compare(inst, full, graph_a, graph_b, verify)
                identity_checks += 2 * int(verify)
    assert (data[:, :, 0, 0] == (full >= 0).sum()).all()
    # At p=.5 the two orientations give full and shortlist exactly once per renter and arm.
    for col in (0, 1):
        assert (
            data[:, :, 2, col].sum(1) == data[0, 0, 0, col] + data[0, 0, 4, col]
        ).all()
    assert (data[..., 5] - data[..., 4] == data[..., 6] - data[..., 7]).all()
    assert (data[..., 9] <= data[..., 6]).all() and (
        data[..., 10] <= data[..., 7]
    ).all()
    return (
        dict(
            design=design,
            seed=spec.seed,
            month=spec.month,
            borough=spec.borough,
            people=inst.n_b,
            cell_weight=WEIGHTS[spec.borough, spec.month],
        ),
        data,
        identity_checks,
    )


def report(meta, data):
    output = []
    for design, panel in meta.groupby("design"):
        ordered = panel.sort_values(["seed", "month", "borough"])
        weights = ordered.cell_weight.to_numpy(float) / ordered.cell_weight.sum()
        # Every market has 64 participants. Sample the entire 12-cell block for each seed.
        n = ordered.people.unique()
        assert n.tolist() == [64] and len(panel) == 240
        factor = 100 / n[0]
        raw = data[ordered.index].mean(2)
        keys = list(COUNTS)
        arrays = [raw]
        for key, left, right in CONTRASTS:
            arrays.append(
                (raw[..., COUNTS.index(right)] - raw[..., COUNTS.index(left)])[
                    ..., None
                ]
            )
            keys.append(key)
        values = np.concatenate(arrays, axis=-1)
        conditional = factor * np.einsum("m,mrsv->rsv", weights, values)
        mean = conditional.mean(0)
        mcse = conditional.std(axis=0, ddof=1) / np.sqrt(len(conditional))
        seeds = ordered.seed.unique()
        by_block = []
        for seed in seeds:
            take = ordered.seed.to_numpy() == seed
            by_block.append(
                factor
                * np.einsum(
                    "m,msv->sv",
                    weights[take] / weights[take].sum(),
                    values[take].mean(1),
                )
            )
        np.testing.assert_allclose(np.mean(by_block, axis=0), mean, atol=1e-10)
        bootstrap = np.random.default_rng(0).integers(
            0, len(seeds), size=(10000, len(seeds))
        )
        frequency = np.zeros((len(bootstrap), len(seeds)), dtype=int)
        np.add.at(frequency, (np.arange(len(frequency))[:, None], bootstrap), 1)
        boot = np.einsum("dr,rsv->dsv", frequency / len(seeds), by_block)
        low, high = np.quantile(boot, [0.025, 0.975], axis=0)
        raw_total = values.mean(axis=1).sum(axis=0)
        for s, scenario in enumerate(SCENARIOS):
            for j, metric in enumerate(keys):
                output.append(
                    dict(
                        design=design,
                        scenario=scenario,
                        metric=metric,
                        weighted_per100=mean[s, j],
                        pair_mcse_pp=mcse[s, j],
                        block_lo=low[s, j],
                        block_hi=high[s, j],
                        pooled_count=raw_total[s, j],
                        people=int(ordered.people.sum()),
                    )
                )
    return pd.DataFrame(output)


def main(args):
    global CACHE, CATALOG, SYNC, WEIGHTS, PAIRS, FIRST_SEED
    repo = args.repo.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo))
    from experiments.pilot_outcomes import CONFIG, DEFAULT_CACHE, load_pilot_cache
    from experiments.config import expand_labeled, load_config
    from experiments.analyze import cell_weights
    from hiddenoptions.markets import Catalog

    path = repo / "publication/two-studies/audit/nyc-temporal/verify_temporal.py"
    spec = importlib.util.spec_from_file_location("rental_independent_da", path)
    aux = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(aux)
    SYNC = aux.synchronous
    entries = expand_labeled(load_config(CONFIG))
    PAIRS = args.pairs
    FIRST_SEED = {
        d: min(s.seed for s, other in entries if other == d)
        for d in set(d for _, d in entries)
    }
    WEIGHTS = cell_weights(sorted({(s.borough, s.month) for s, _ in entries}))
    CACHE = load_pilot_cache(DEFAULT_CACHE)
    CATALOG = Catalog()
    if args.workers == 1:
        results = list(map(process_market, entries))
    else:
        with multiprocessing.get_context("fork").Pool(args.workers) as pool:
            results = pool.map(process_market, entries)
    records = [r for r in results if r is not None]
    assert len(records) == 480 and len(entries) == 600
    meta = pd.DataFrame([r[0] for r in records])
    data = np.stack([r[1] for r in records])
    counts = sum(r[2] for r in records)
    assert counts == 24 * 18
    meta.to_csv(out / "nyc-markets.csv", index=False)
    np.savez_compressed(
        out / "nyc-advice-counts.npz",
        counts=data.astype("uint8"),
        scenarios=SCENARIOS,
        metrics=COUNTS,
    )
    summary = report(meta, data)
    original = (
        pd.read_csv(repo / "publication/two-studies/evidence/policy-rates.csv")
        .query("group=='all' and comparison=='haiku_vs_uniform'")
        .set_index(["design", "metric"])
    )
    ranks = (
        pd.read_csv(
            repo / "publication/two-studies/audit/ordinal/rental-rank-summary.csv"
        )
        .query("cutoff==1")
        .set_index("design")
    )
    for design, panel in summary[summary.scenario == "adhere_100"].groupby("design"):
        for metric, ref in [
            ("delta_coverage", original.loc[(design, "covered")]),
            ("delta_first", ranks.loc[design]),
            ("delta_placed", original.loc[(design, "placed")]),
        ]:
            row = panel.set_index("metric").loc[metric]
            np.testing.assert_allclose(
                row[["weighted_per100", "block_lo", "block_hi"]].to_numpy(float),
                ref[["delta_pp", "lo_pp", "hi_pp"]].to_numpy(float),
                atol=1e-10,
                rtol=0,
            )
    summary.to_csv(out / "nyc-advice-results.csv", index=False)
    outputs = ["nyc-markets.csv", "nyc-advice-counts.npz", "nyc-advice-results.csv"]
    receipt = dict(
        pairs=PAIRS,
        n_markets=len(meta),
        n_people=int(meta.people.sum()),
        independent_allocation_checks=counts,
        independent_applicant_checks=64 * counts,
        interior_clears=480 * PAIRS * 2 * 3 * 2,
        inputs={"cache": digest(DEFAULT_CACHE), "market_config": digest(CONFIG)},
        files={n: digest(out / n) for n in outputs},
    )
    (out / "nyc-advice-checks.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        summary[
            summary.metric.isin(
                ["delta_placed", "delta_first", "delta_coverage", "same_loss"]
            )
        ].to_string(index=False)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--pairs", type=int, default=32)
    args = parser.parse_args()
    if args.pairs < 2 or args.workers < 1:
        parser.error("require at least two pairs and one worker")
    main(args)
