"""Replay the development pilot. Store aggregate counts only."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rate_summary(market_counts, weights, reps):
    """Standardize cell-level ratios, resampling the whole 12-cell seed block."""
    columns = ["n", "covered_a", "covered_b", "placed_a", "placed_b", "max_a", "max_b"]
    summaries = []
    for (design, comparison, group), rows in market_counts.groupby(
        ["design", "comparison", "group"]
    ):
        if group != "all":
            columns_used = columns[:5]
        else:
            columns_used = columns
        piv = rows.pivot_table(
            index="seed",
            columns=["borough", "month"],
            values=columns_used,
            aggfunc="sum",
        )
        cells = sorted(set(zip(rows.borough, rows.month)))
        w = np.array([weights[cell] for cell in cells])
        covered = w.sum()
        w /= covered
        cube = np.stack(
            [piv.loc[:, key].loc[:, cells].to_numpy() for key in columns_used], axis=-1
        )
        # Frequencies keep every borough and month in a resampled seed block together.
        rng = np.random.default_rng(0)
        samples = rng.integers(0, len(cube), size=(reps, len(cube)))
        freq = np.zeros((reps, len(cube)), int)
        np.add.at(freq, (np.arange(reps)[:, None], samples), 1)
        freq = np.vstack([np.ones(len(cube), int), freq])
        aggregate = (freq @ cube.reshape(len(cube), -1)).reshape(
            len(freq), len(cells), len(columns_used)
        )
        if (aggregate[:, :, 0] <= 0).any():
            raise ValueError("Some resample lost a required supported denominator")
        pct = 100 * np.einsum(
            "c,dcj->dj", w, aggregate[:, :, 1:] / aggregate[:, :, 0, None]
        )
        for i, measure in enumerate(
            ["covered", "placed"] + (["feasible_capacity"] if group == "all" else [])
        ):
            a, b = pct[:, 2 * i], pct[:, 2 * i + 1]
            diff = b - a
            lower, upper = np.quantile(diff[1:], [0.025, 0.975])
            summaries.append(
                dict(
                    design=design,
                    comparison=comparison,
                    group=group,
                    metric=measure,
                    n=int(rows.n.sum()),
                    blocks=len(cube),
                    cells=len(cells),
                    weight=covered,
                    a_pct=a[0],
                    b_pct=b[0],
                    delta_pp=diff[0],
                    lo_pp=lower,
                    hi_pp=upper,
                )
            )
        if group == "all":
            diff = (pct[:, 5] - pct[:, 3]) - (pct[:, 4] - pct[:, 2])
            lower, upper = np.quantile(diff[1:], [0.025, 0.975])
            summaries.append(
                dict(
                    design=design,
                    comparison=comparison,
                    group=group,
                    metric="capacity_minus_placed",
                    n=int(rows.n.sum()),
                    blocks=len(cube),
                    cells=len(cells),
                    weight=covered,
                    a_pct=pct[0, 4] - pct[0, 2],
                    b_pct=pct[0, 5] - pct[0, 3],
                    delta_pp=diff[0],
                    lo_pp=lower,
                    hi_pp=upper,
                )
            )
    return pd.DataFrame(summaries)


def replay(repo, out, reps):
    sys.path.insert(0, str(repo))
    from experiments.analyze import cell_weights
    from experiments.config import expand_labeled, load_config
    from experiments.pilot_outcomes import (
        BASE,
        CONFIG,
        DEFAULT_CACHE,
        load_pilot_cache,
        pilot_displays,
        policy_events,
    )
    from hiddenoptions.exposure import transparent_exposure
    from hiddenoptions.markets import Catalog, InsufficientSupply, build_market
    from hiddenoptions.matching import deferred_acceptance

    cache = load_pilot_cache(DEFAULT_CACHE)
    catalog = Catalog()
    plan = expand_labeled(load_config(CONFIG))
    weights = cell_weights(sorted({(s.borough, s.month) for s, _ in plan}))
    count_path = BASE / "paired-outcomes-counts.csv"
    ref = pd.read_csv(count_path).set_index(
        ["variant", "seed", "borough", "month", "group", "metric"]
    )
    if not ref.index.is_unique:
        raise ValueError("Ambiguous reference counts")
    seen = set()
    omitted = Counter()
    checks = 0
    market_rows, joint_rows, status_rows, named_rows = [], [], [], []
    for spec, design in plan:
        try:
            market = build_market(spec, catalog)
        except InsufficientSupply:
            omitted[(design, spec.borough)] += 1
            continue
        if spec.market_id in seen:
            raise ValueError("Duplicate market")
        seen.add(spec.market_id)
        base, haiku = pilot_displays(market, cache)
        menus = {
            "uniform": base,
            "haiku": haiku,
            "transparent": transparent_exposure(market, k=5),
        }
        full, _ = deferred_acceptance(market.inst, market.inst.E)
        events = {key: policy_events(market.inst, d, full) for key, d in menus.items()}
        named = {}
        for key in ("uniform", "haiku"):
            assignment, _ = deferred_acceptance(market.inst, menus[key])
            named[key] = (full >= 0) & (assignment == full)
            if (named[key] & ~events[key]["partner_shown"]).any():
                raise AssertionError(
                    "A named full-list assignment must have been visible"
                )
            np.testing.assert_array_equal(assignment >= 0, events[key]["placed"])
        capacities = {
            key: int(
                (
                    maximum_bipartite_matching(csr_matrix(d), perm_type="column") >= 0
                ).sum()
            )
            for key, d in menus.items()
        }
        low = market.renters.lower_income.to_numpy(bool)
        metadata = dict(
            design=design, seed=spec.seed, borough=spec.borough, month=spec.month
        )
        for old, new in [
            ("uniform", "haiku"),
            ("transparent", "haiku"),
            ("uniform", "transparent"),
        ]:
            a, b = events[old], events[new]
            same = (menus[old] == menus[new]).all(axis=1)
            d_cov = b["partner_shown"].astype(int) - a["partner_shown"]
            d_place = b["placed"].astype(int) - a["placed"]
            if (same & (d_cov != 0)).any():
                raise AssertionError("Partner coverage changed on an identical menu")
            for group, mask in [
                ("all", np.ones(len(low), bool)),
                ("lower_income", low),
                ("higher_income", ~low),
            ]:
                record = dict(
                    **metadata,
                    comparison=f"{new}_vs_{old}",
                    group=group,
                    n=int(mask.sum()),
                    covered_a=int(a["partner_shown"][mask].sum()),
                    covered_b=int(b["partner_shown"][mask].sum()),
                    placed_a=int(a["placed"][mask].sum()),
                    placed_b=int(b["placed"][mask].sum()),
                    max_a=capacities[old] if group == "all" else None,
                    max_b=capacities[new] if group == "all" else None,
                )
                market_rows.append(record)
                if (old, new) != ("uniform", "haiku"):
                    continue
                new_target = mask & (d_cov == 1)
                n_new = int(new_target.sum())
                new_named = int((new_target & named["haiku"]).sum())
                new_other = int((new_target & ~named["haiku"] & b["placed"]).sum())
                new_unplaced = int((new_target & ~b["placed"]).sum())
                assert new_named + new_other + new_unplaced == n_new
                named_rows.append(
                    dict(
                        design=design,
                        group=group,
                        n=int(mask.sum()),
                        newly_shown=n_new,
                        newly_shown_named=new_named,
                        newly_shown_other=new_other,
                        newly_shown_unplaced=new_unplaced,
                        named_uniform=int((mask & named["uniform"]).sum()),
                        named_haiku=int((mask & named["haiku"]).sum()),
                        named_gain=int(
                            (mask & ~named["uniform"] & named["haiku"]).sum()
                        ),
                        named_loss=int(
                            (mask & named["uniform"] & ~named["haiku"]).sum()
                        ),
                    )
                )
                if group != "all":
                    for metric in ("placed", "partner_shown", "hidden_blocker"):
                        refrow = ref.loc[
                            design, spec.seed, spec.borough, spec.month, group, metric
                        ]
                        before, after = a[metric][mask], b[metric][mask]
                        observed = [
                            len(before),
                            before.sum(),
                            after.sum(),
                            ((~before) & after).sum(),
                            (before & (~after)).sum(),
                        ]
                        if not np.array_equal(
                            observed,
                            refrow[
                                ["n", "uniform", "haiku", "gained_event", "lost_event"]
                            ].values,
                        ):
                            raise AssertionError((metadata, group, metric))
                        checks += 1
                # Split a zero coverage change into staying covered, staying hidden,
                # or having no partner in full-list DA. All three imply d_cov=0.
                for label, mask_f in [
                    ("no_full_partner", full < 0),
                    (
                        "always_covered",
                        (full >= 0) & a["partner_shown"] & b["partner_shown"],
                    ),
                    (
                        "always_hidden",
                        (full >= 0) & ~a["partner_shown"] & ~b["partner_shown"],
                    ),
                    ("newly_covered", d_cov > 0),
                    ("newly_hidden", d_cov < 0),
                ]:
                    for menu_flag in (False, True):
                        members = mask & mask_f & (same == menu_flag)
                        status_rows.append(
                            dict(
                                design=design,
                                group=group,
                                same_menu=int(menu_flag),
                                coverage_status=label,
                                n=int(members.sum()),
                                with_feasible=int(
                                    (members & market.inst.E.any(axis=1)).sum()
                                ),
                                lost=int((members & (d_place < 0)).sum()),
                                gained=int((members & (d_place > 0)).sum()),
                            )
                        )
                encoded = same.astype(int) * 9 + (d_cov + 1) * 3 + (d_place + 1)
                counts = np.bincount(encoded[mask], minlength=18)
                for i in counts.nonzero()[0]:
                    menu, delta_pair = divmod(i, 9)
                    dc, dy = divmod(delta_pair, 3)
                    joint_rows.append(
                        dict(
                            **metadata,
                            group=group,
                            same_menu=menu,
                            delta_coverage=dc - 1,
                            delta_placement=dy - 1,
                            count=int(counts[i]),
                        )
                    )
    assert set(cache.table).issubset(seen) and len(seen) == 480
    assert omitted == {
        ("primary", "Staten Island"): 60,
        ("enriched", "Staten Island"): 60,
    }
    assert checks == len(ref) == 2 * 3 * 480
    out.mkdir(parents=True, exist_ok=True)
    markets = pd.DataFrame(market_rows)
    joints = pd.DataFrame(joint_rows)
    summary = rate_summary(markets, weights, reps)
    old_rates = pd.read_csv(BASE / "paired-outcomes.csv")
    for row in summary.query(
        "comparison == 'haiku_vs_uniform' and group != 'all'"
    ).itertuples():
        label = {"placed": "placed", "covered": "partner_shown"}[row.metric]
        old = old_rates[
            (old_rates.variant == row.design)
            & (old_rates.group == row.group)
            & (old_rates.metric == label)
        ].iloc[0]
        np.testing.assert_allclose(
            [row.a_pct, row.b_pct, row.delta_pp],
            [old.uniform_pct, old.haiku_pct, old.difference_pp],
            atol=1e-10,
        )
        if reps == 10000:
            np.testing.assert_allclose(
                [row.lo_pp, row.hi_pp], [old.lo_pp, old.hi_pp], atol=1e-10
            )
    joint = joints.groupby(
        ["design", "group", "same_menu", "delta_coverage", "delta_placement"]
    ).sum(numeric_only=True)
    idx = pd.MultiIndex.from_product(
        [
            ["primary", "enriched"],
            ["all", "lower_income", "higher_income"],
            [0, 1],
            [-1, 0, 1],
            [-1, 0, 1],
        ],
        names=joint.index.names,
    )
    joint["count"].reindex(idx, fill_value=0).to_csv(out / "joint.csv")
    # A public seed/block output is sufficient to recompute raw transition uncertainty.
    keys = ["design", "seed", "group", "same_menu", "delta_coverage", "delta_placement"]
    joints.groupby(keys)["count"].sum().to_csv(out / "joint-by-seed.csv")
    detail = (
        pd.DataFrame(status_rows)
        .groupby(["design", "group", "same_menu", "coverage_status"], as_index=False)
        .sum()
    )
    check = detail.groupby(["design", "group"])[["n", "lost", "gained"]].sum()
    tallies = (
        joints.groupby(["design", "group", "delta_placement"])["count"]
        .sum()
        .unstack(fill_value=0)
    )
    np.testing.assert_array_equal(check.n, tallies.sum(axis=1))
    np.testing.assert_array_equal(check.lost, tallies[-1])
    np.testing.assert_array_equal(check.gained, tallies[1])
    detail.to_csv(out / "coverage-history.csv", index=False)
    named_results = (
        pd.DataFrame(named_rows).groupby(["design", "group"], as_index=False).sum()
    )
    np.testing.assert_array_equal(
        named_results.named_haiku - named_results.named_uniform,
        named_results.named_gain - named_results.named_loss,
    )
    named_results.to_csv(out / "named-partners.csv", index=False)
    summary.to_csv(out / "policy-rates.csv", index=False)
    receipt = dict(
        config_sha256=hash_file(CONFIG),
        cache_sha256=hash_file(DEFAULT_CACHE),
        archived_counts_sha256=hash_file(count_path),
        designs=["primary", "enriched"],
        months=sorted({s.month for s, _ in plan}),
        market_sizes=[64, 64],
        markets=480,
        planned_markets=600,
        skipped_staten_island_per_design=60,
        validated_reference_rows=checks,
        public_group_rows_per_market=6,
        market_block_bootstrap_reps=reps,
        bootstrap_seed=0,
    )
    (out / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        required=True,
        help="Ishan follow-up checkout at f0c4f65, with development catalogs",
    )
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--reps", type=int, default=10000)
    args = parser.parse_args()
    replay(args.repo.resolve(), args.out, args.reps)
