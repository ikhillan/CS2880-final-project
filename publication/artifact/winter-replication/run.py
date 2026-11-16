"""Run the frozen Jan–Mar temporal replication with only calibration donor profiles."""

import argparse
from collections import Counter
import csv
from hashlib import file_digest
import importlib.util
import json
from pathlib import Path
import platform
import sys
import numpy as np


HERE = Path(__file__).resolve().parent
MONTHS = ("2026-01", "2026-02", "2026-03")
BOROUGHS = ("Bronx", "Brooklyn", "Manhattan", "Queens")
POLICIES = ("UU", "CU", "UC", "CC", "balanced")
GROUPS = ("lower", "higher")
SIZE = 64
BLOCKS = 40
MISSING = {"", "NA", "NONE", "UNKNOWN", "NULL"}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


replay = load("spillover_replay", HERE.parent / "spillovers/replay.py")
anchored = load("anchored", HERE.parent / "anchored/run.py")


def digest(path):
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def build_pools(data, winter, loader, cpi):
    def unit_key(row):
        street, unit = loader.normalized(row["street"]), loader.normalized(row["unit"])
        if street not in MISSING and unit not in MISSING:
            return (row["borough"].strip(), street, unit)
        return None

    summer_ids, summer_keys = set(), set()
    for m in ("2026-06", "2026-07", "2026-08"):
        with (data / "firstmover" / f"{m}.csv").open(
            encoding="utf-8-sig", newline=""
        ) as f:
            for row in csv.DictReader(f):
                if row["borough"].strip() in BOROUGHS and row["state"].strip() == "NY":
                    if row["id"].strip():
                        summer_ids.add(row["id"])
                    key = unit_key(row)
                    if key:
                        summer_keys.add(key)
    winter_ids, winter_keys = set(), set()
    pools = {}
    ledgers = {}
    for m in MONTHS:
        path = winter / f"{m}.csv"
        cleaned, pre_ledger = loader.load_catalogue(path)
        # Join unit identity to the first valid (timestamp, ID)-sorted source row retained by the loader.
        sought = {
            (listing_id, b, price, beds)
            for b, price, beds, listing_id in cleaned
            if b in BOROUGHS
        }
        keys = {}
        with path.open(encoding="utf-8-sig", newline="") as f:
            for row in sorted(
                csv.DictReader(f), key=lambda r: (r["created_at_utc"], r["id"])
            ):
                if row["state"].strip() != "NY":
                    continue
                try:
                    join = (
                        row["id"],
                        row["borough"].strip(),
                        float(row["price"]),
                        float(row["bedrooms"]),
                    )
                except ValueError:
                    continue
                if join in sought and join not in keys:
                    keys[join] = unit_key(row)
        assert set(keys) == sought
        counters = {b: Counter() for b in BOROUGHS}
        kept = {b: [] for b in BOROUGHS}
        for b, price, beds, listing_id in cleaned:
            if b not in BOROUGHS:
                continue
            count = counters[b]
            count["cleaned"] += 1
            key = keys[listing_id, b, price, beds]
            summer_id = listing_id in summer_ids
            summer_key = key is not None and key in summer_keys
            count["summer_id_matches_any"] += summer_id
            count["summer_unit_matches_any"] += summer_key
            if key is None:
                count["ambiguous_excluded"] += 1
            elif summer_id or summer_key:
                count["summer_overlap_excluded"] += 1
            elif listing_id in winter_ids or key in winter_keys:
                count["winter_duplicate_excluded"] += 1
            else:
                winter_ids.add(listing_id)
                winter_keys.add(key)
                kept[b].append((price * cpi["2026-07"] / cpi[m], beds))
                count["kept"] += 1
        for b, cell in kept.items():
            count = counters[b]
            assert count["cleaned"] == sum(
                count[k]
                for k in (
                    "ambiguous_excluded",
                    "summer_overlap_excluded",
                    "winter_duplicate_excluded",
                    "kept",
                )
            )
            pools[m, b] = np.asarray(cell)
        ledgers[m] = {
            "archived_within_month_cleaning": pre_ledger,
            "four_boroughs": counters,
        }
    assert summer_ids.isdisjoint(winter_ids) and summer_keys.isdisjoint(winter_keys)
    return pools, ledgers


def calibration(data):
    meta = json.loads((data / "acs/summary.json").read_text())
    records = {b: [] for b in BOROUGHS}
    with (data / "acs/recent_donors.csv").open() as f:
        for row in csv.DictReader(f):
            if row["donor_split"] != "calibration":
                continue
            b = row["borough"]
            if b in records:
                records[b].append(
                    (
                        float(row["income_july2026"]),
                        max(0, (int(row["household_size"]) - 1) // 2),
                        float(row["income_2024"]) <= meta["median_income_2024"],
                        float(row["weight"]),
                    )
                )
    donors = {b: np.array(rows) for b, rows in records.items()}
    weights = [meta["by_borough"][b]["recent_weight"] for b in BOROUGHS]
    return donors, np.array(weights), meta


def market(rows, pool, seed):
    rng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(9)]
    draw = rows[
        rng[0].choice(len(rows), SIZE, replace=True, p=rows[:, 3] / rows[:, 3].sum())
    ]
    prices, beds = pool[rng[1].choice(len(pool), SIZE, replace=False)].T
    income, need, lower = draw[:, :3].T
    low = lower.astype(bool)
    budget = np.maximum(0, 0.4 * income / 12)
    feasible = (prices <= budget[:, None]) & (beds >= need[:, None])
    degree = feasible.sum(axis=1)
    utility = (
        -2 * prices / np.maximum(budget[:, None], 1)
        + 0.5 * np.minimum(2, beds - need[:, None])
        + 0.5 * rng[2].standard_normal(SIZE)
        + 0.75 * rng[3].standard_normal((SIZE, SIZE))
    )
    pref = np.argsort(-utility, axis=1, kind="stable")
    u = replay.visible_edges(
        np.argsort(rng[4].random((SIZE, SIZE)), axis=1, kind="stable"), feasible
    )
    prices_order = np.lexsort((rng[5].random(SIZE), prices))
    c = replay.visible_edges(np.broadcast_to(prices_order, feasible.shape), feasible)
    listing_draws = rng[7].random((SIZE, SIZE))
    lr = np.argsort(
        np.argsort(listing_draws, axis=0, kind="stable"), axis=0, kind="stable"
    )
    price_lists = [prices_order[eligible[prices_order]] for eligible in feasible]
    balanced = anchored.balanced_mask(price_lists, rng[8], SIZE)
    masks = (u, np.where(low[:, None], c, u), np.where(low[:, None], u, c), c, balanced)
    assert np.array_equal(masks[0][low], masks[2][low])
    matched = []
    stats = np.zeros((len(POLICIES), len(GROUPS), 4))
    for a, mask in enumerate(masks):
        length = mask.sum(axis=1)
        assert np.array_equal(length, np.minimum(5, degree))
        assert not (mask & ~feasible).any()
        assert np.array_equal(mask[degree <= 5], feasible[degree <= 5])
        assignments, holders = replay.simultaneous_da(mask, pref, lr)
        blockers = replay.blockers(feasible, pref, lr, assignments, holders)
        assert not (mask & blockers).any()
        mean_rent = (mask @ prices) / np.maximum(1, length)
        matched.append(assignments >= 0)
        for g, group in enumerate((low, ~low)):
            stats[a, g] = [
                group.sum(),
                (group & (degree > 0)).sum(),
                (group & (assignments >= 0)).sum(),
                mean_rent[group].sum(),
            ]
    band = low & (degree > 0) & (degree <= 5)
    lost = matched[0] & ~matched[2] & band
    gained = ~matched[0] & matched[2] & band
    return stats, np.array([band.sum(), lost.sum(), gained.sum()])


def run(data, winter, catalogue_path, out):
    manifest = json.loads((HERE / "sources.json").read_text())
    reference = json.loads((HERE.parent / "nyc-matching-dev/results.json").read_text())
    files = {
        **{m + ".csv": winter / (m + ".csv") for m in MONTHS},
        "BLS-snapshot.json": HERE / "BLS-snapshot.json",
    }
    for name, path in files.items():
        assert digest(path) == manifest[name]["sha256"], name
    for name, path in [
        (s + ".csv", data / "firstmover" / f"{s}.csv")
        for s in ("2026-06", "2026-07", "2026-08")
    ] + [
        ("recent_donors.csv", data / "acs/recent_donors.csv"),
        ("summary.json", data / "acs/summary.json"),
        ("scarcity.py", catalogue_path),
    ]:
        assert digest(path) == reference["raw_hashes"][name], name
    assert (
        digest(HERE / "PROTOCOL.md")
        == (HERE / "PROTOCOL.sha256").read_text().split()[0]
    )
    sys.path.insert(0, str(catalogue_path.resolve().parent))
    loader = load("catalogue_loader", catalogue_path)
    official = json.loads((HERE / "BLS-snapshot.json").read_text())
    series = official["Results"]["series"][0]
    assert (
        official["status"] == "REQUEST_SUCCEEDED"
        and series["seriesID"] == "CUUR0000SA0"
    )
    allowed = set(MONTHS) | {"2026-07"}
    cpi = {
        k: float(r["value"])
        for r in series["data"]
        if (k := r["year"] + "-" + r["period"][1:]) in allowed
    }
    donors, weights, meta = calibration(data)
    assert set(cpi) == allowed and cpi["2026-07"] == meta["cpi_u"]["2026-07"]
    pools, ledgers = build_pools(data, winter, loader, cpi)
    qualified = np.array([[len(pools[m, b]) >= SIZE for b in BOROUGHS] for m in MONTHS])
    w = qualified * weights[None, :]
    active = [(i, j) for i, j in zip(*np.where(qualified))]
    counts = np.zeros((BLOCKS, len(active), len(POLICIES), len(GROUPS), 4))
    transitions = np.zeros((BLOCKS, len(active), 3), int)
    for block in range(BLOCKS):
        for cell, (i, j) in enumerate(active):
            counts[block, cell], transitions[block, cell] = market(
                donors[BOROUGHS[j]],
                pools[MONTHS[i], BOROUGHS[j]],
                [2026111301, block, int(i), int(j)],
            )
    doc = {
        "protocol_sha256": digest(HERE / "PROTOCOL.md"),
        "source_hashes": {
            **reference["raw_hashes"],
            **{n: digest(p) for n, p in files.items()},
        },
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "cpi": cpi,
        "input_ledger": ledgers,
        "markets": len(active) * BLOCKS,
        "assignments_verified": len(active) * BLOCKS * len(POLICIES),
        "seed": 2026111301,
        "bootstrap_seed": 2026111302,
        "bootstrap_draws": 5000,
        "excluded_cells": [
            f"{MONTHS[i]} {BOROUGHS[j]}" for i, j in zip(*np.where(~qualified))
        ],
    }
    if not w.sum(axis=1).all():
        doc["status"] = (
            "Unavailable: at least one entire month did not meet the support condition."
        )
        out.write_text(json.dumps(doc, indent=2) + "\n")
        return
    w = (w / w.sum(axis=1)[:, None] / len(MONTHS))[qualified]

    def standard(block_weights):
        totals = np.tensordot(block_weights, counts, axes=(0, 0))
        denom, positive, matched, rents = np.moveaxis(totals, -1, 0)

        def ratio(num, den):
            return np.divide(
                num, den, out=np.full_like(num, np.nan, dtype=float), where=den > 0
            )

        match = np.einsum("c,cag->ag", w, ratio(matched, denom)) * 100
        rent = np.einsum("c,cag->ag", w, ratio(rents, positive))
        very_small = np.tensordot(block_weights, transitions, axes=(0, 0)).T
        band, lost, gained = (
            np.einsum("c,mc->m", w, ratio(very_small, denom[:, 0, 0])) * 100
        )
        return (
            match,
            rent,
            np.r_[
                match[2, 0] - match[0, 0],
                lost,
                gained,
                gained - lost,
                match[4] - match[3],
                rent[4] - rent[3],
            ],
        )

    observed_match, observed_rent, values = standard(np.ones(BLOCKS, int))
    rng = np.random.default_rng(2026111302)
    draws = [
        standard(np.bincount(rng.integers(BLOCKS, size=BLOCKS), minlength=BLOCKS))[2]
        for _ in range(5000)
    ]
    bounds = np.quantile(draws, [0.025, 0.975], axis=0)

    def output_number(value):
        return float(value) if np.isfinite(value) else None

    names = (
        "primary_UC_minus_UU_lower_matched_pp",
        "fully_exposed_lower_losses_pp_all_lower",
        "fully_exposed_lower_gains_pp_all_lower",
        "fully_exposed_lower_net_pp_all_lower",
        "balanced_minus_CC_lower_matched_pp",
        "balanced_minus_CC_higher_matched_pp",
        "balanced_minus_CC_lower_mean_displayed_rent_dollars",
        "balanced_minus_CC_higher_mean_displayed_rent_dollars",
    )
    doc["contrasts"] = {
        n: {
            "estimate": output_number(values[i]),
            "conditional_mc_95": [output_number(x) for x in bounds[:, i]],
        }
        for i, n in enumerate(names)
    }
    doc["arm_means"] = {
        a: {
            g: {
                "percent_matched": output_number(observed_match[i, j]),
                "conditional_displayed_rent_dollars": output_number(
                    observed_rent[i, j]
                ),
            }
            for j, g in enumerate(GROUPS)
        }
        for i, a in enumerate(POLICIES)
    }
    doc["fully_exposed_lower_raw_counts"] = dict(
        zip(
            ("eligible_1_to_5", "lost", "gained"),
            map(int, transitions.sum(axis=(0, 1))),
        )
    )
    doc["renter_instance_counts"] = dict(
        zip(GROUPS, map(int, counts[:, :, 0, :, 0].sum(axis=(0, 1))))
    )
    doc["status"] = (
        "Completed. Null point estimates or intervals denote unsupported conditional ratios."
    )
    out.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--winter", required=True, type=Path)
    p.add_argument("--catalogue-module", required=True, type=Path)
    p.add_argument("--out", type=Path, default=HERE / "results.json")
    a = p.parse_args()
    run(a.data, a.winter, a.catalogue_module, a.out)
