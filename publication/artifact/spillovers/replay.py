"""Independent DA replay and within-market policy-switch analysis on development data."""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


MONTHS = ("2026-06", "2026-07", "2026-08")
BOROUGHS = ("Bronx", "Brooklyn", "Manhattan", "Queens")
ARMS = ("UU", "CU", "UC", "CC")  # lower group's policy first
GROUPS = ("lower", "higher")
PRIORITIES = ("common", "heterogeneous")
METRICS = ("matched", "blocked", "choices_gt5", "max_alone", "distinct_listings_alone")
CONTRASTS = (
    "CC_minus_UU",
    "switch_lower_average",
    "switch_higher_average",
    "switch_lower_when_higher_U",
    "switch_lower_when_higher_C",
    "switch_higher_when_lower_U",
    "switch_higher_when_lower_C",
)


def load_development(data_path, catalogue_module):
    spec = importlib.util.spec_from_file_location(
        "development_catalogue_loader", catalogue_module
    )
    sys.path.insert(0, str(catalogue_module.resolve().parent))
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    meta = json.loads((data_path / "acs/summary.json").read_text())
    rows = {b: [] for b in BOROUGHS}
    with (data_path / "acs/recent_donors.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["donor_split"] == "calibration" and row["borough"] in rows:
                rows[row["borough"]].append(
                    (
                        float(row["income_july2026"]),
                        (int(row["household_size"]) - 1) // 2,
                        int(float(row["income_2024"]) <= meta["median_income_2024"]),
                        float(row["weight"]),
                    )
                )
    donors = {b: np.asarray(r) for b, r in rows.items()}
    catalogues = {}
    for month in MONTHS:
        listings, _ = loader.load_catalogue(data_path / "firstmover" / f"{month}.csv")
        for borough in BOROUGHS:
            catalogues[month, borough] = np.asarray(
                [
                    (price * meta["cpi_u"]["2026-07"] / meta["cpi_u"][month], beds)
                    for b, price, beds, _ in listings
                    if b == borough
                ]
            )
    weights = np.array(
        [meta["by_borough"][b]["recent_weight"] for b in BOROUGHS], float
    )
    return donors, catalogues, weights / weights.sum()


def visible_edges(order, eligible):
    mask = np.zeros_like(eligible)
    for b in range(len(mask)):
        feasible_order = order[b, eligible[b, order[b]]]
        mask[b, feasible_order[:5]] = True
    assert np.array_equal(mask.sum(axis=1), np.minimum(eligible.sum(axis=1), 5))
    return mask


def instance(donors, catalogue, block, month_id, borough_id):
    size = 64
    children = np.random.SeedSequence([20261112, block, month_id, borough_id]).spawn(9)
    rng = [np.random.default_rng(child) for child in children]
    sampled = donors[
        rng[0].choice(
            len(donors), size, replace=True, p=donors[:, 3] / donors[:, 3].sum()
        )
    ]
    prices, beds = catalogue[rng[1].choice(len(catalogue), size, replace=False)].T
    income, needed, lower = sampled[:, :3].T
    budget = np.maximum(0, 0.4 * income / 12)
    eligible = (prices <= budget[:, None]) & (beds >= needed[:, None])
    utility = (
        -2 * prices / np.maximum(budget[:, None], 1)
        + 0.5 * np.minimum(2, beds - needed[:, None])
        + 0.5 * rng[2].standard_normal(size)
        + 0.75 * rng[3].standard_normal((size, size))
    )
    preference = np.argsort(-utility, axis=1, kind="stable")
    uniform_order = np.argsort(rng[4].random((size, size)), axis=1, kind="stable")
    cheapest_order = np.broadcast_to(
        np.lexsort((rng[5].random(size), prices)), (size, size)
    )
    uniform = visible_edges(uniform_order, eligible)
    cheapest = visible_edges(cheapest_order, eligible)
    low = lower.astype(bool)
    masks = (
        uniform,
        np.where(low[:, None], cheapest, uniform),
        np.where(low[:, None], uniform, cheapest),
        cheapest,
    )
    common = rng[6].permutation(size)
    general = rng[7].random((size, size))
    # In each listing's ordering, the archived generator interprets the common permutation as IDs.
    common_rank = np.broadcast_to(np.argsort(common)[:, None], (size, size))
    independent_rank = np.argsort(
        np.argsort(general, axis=0, kind="stable"), axis=0, kind="stable"
    )
    return eligible, preference, (common_rank, independent_rank), masks, low


def simultaneous_da(visible, preference, rank):
    """In every round each free applicant with options makes one fresh proposal."""
    n, m = visible.shape
    lists = [row[visible[b, row]] for b, row in enumerate(preference)]
    remaining = np.array([len(row) for row in lists])
    next_entry = np.zeros(n, int)
    renters = np.full(n, -1, int)
    holders = np.full(m, -1, int)
    while True:
        active = np.flatnonzero((renters < 0) & (next_entry < remaining))
        if not len(active):
            break
        proposals = {}
        for b in active:
            s = lists[b][next_entry[b]]
            proposals.setdefault(s, []).append(b)
            next_entry[b] += 1
        for s, people in proposals.items():
            old = holders[s]
            if old >= 0:
                people.append(old)
            winner = min(people, key=lambda i: rank[i, s])
            if old >= 0:
                renters[old] = -1
            renters[winner] = s
            holders[s] = winner
    matched = np.flatnonzero(renters >= 0)
    assert np.all(visible[matched, renters[matched]])
    assert np.array_equal(holders[renters[matched]], matched)
    return renters, holders


def blockers(eligible, pref, rank, renters, holders):
    n, m = eligible.shape
    rr = np.argsort(pref, axis=1, kind="stable")
    curr = np.where(renters < 0, m, rr[np.arange(n), renters])
    partner = np.where(holders < 0, n, rank[holders, np.arange(m)])
    return eligible & (rr < curr[:, None]) & (rank < partner)


def matching_cap(graph):
    if not graph.any():
        return 0
    assignment = maximum_bipartite_matching(csr_matrix(graph), perm_type="column")
    return int((assignment >= 0).sum())


def transforms(rates):
    uu, cu, uc, cc = np.moveaxis(rates, 1, 0)
    return np.stack(
        [
            cc - uu,
            0.5 * ((cu - uu) + (cc - uc)),
            0.5 * ((uc - uu) + (cc - cu)),
            cu - uu,
            cc - uc,
            uc - uu,
            cc - cu,
        ],
        axis=1,
    )


def main(data, module, reference_path, output):
    ref = json.loads(reference_path.read_text())
    B = ref["independent_seed_blocks"]
    assert B == 40
    for path in [
        data / "acs/summary.json",
        data / "acs/recent_donors.csv",
        *(data / "firstmover" / f"{month}.csv" for month in MONTHS),
        module,
    ]:
        with path.open("rb") as f:
            assert (
                hashlib.file_digest(f, "sha256").hexdigest()
                == ref["raw_hashes"][path.name]
            ), path
    donors, catalogues, borough_weights = load_development(data, module)
    weights = np.broadcast_to(
        borough_weights[None, :] / len(MONTHS), (len(MONTHS), len(BOROUGHS))
    )
    counts = np.zeros((B, 3, 4, 2, 4, 2, len(METRICS)), int)
    denominators = np.zeros((B, 3, 4, 2), int)
    whole = np.zeros((B, 3, 4, 4, 2), int)
    # Within lower-income renters compare the same individual before and after only
    # higher-income shortlists switch. Record losing/gaining a match separately.
    transitions = np.zeros(
        (B, 3, 4, 2, 2, 2, 3), int
    )  # priority, lower policy, degree band, denominator/loss/gain
    for b in range(B):
        for t, month in enumerate(MONTHS):
            for c, borough in enumerate(BOROUGHS):
                eligible, order, priorities, masks, low = instance(
                    donors[borough], catalogues[month, borough], b, t, c
                )
                degree = eligible.sum(axis=1)
                high_degree = degree > 5
                assignments = np.zeros((2, 4, len(low)), bool)
                for g, group in enumerate((low, ~low)):
                    denominators[b, t, c, g] = group.sum()
                for a, mask in enumerate(masks):
                    assert np.array_equal(
                        mask.sum(axis=1), np.minimum(eligible.sum(axis=1), 5)
                    )
                    assert not (mask & ~eligible).any()
                    whole[b, t, c, a] = [matching_cap(mask), mask.any(axis=0).sum()]
                    caps = [matching_cap(mask[group]) for group in (low, ~low)]
                    reachable = [mask[group].any(axis=0).sum() for group in (low, ~low)]
                    for h, rank in enumerate(priorities):
                        renters, holders = simultaneous_da(mask, order, rank)
                        assignments[h, a] = renters >= 0
                        blocks = blockers(eligible, order, rank, renters, holders)
                        assert not (mask & blocks).any()
                        assert not (blocks.any(axis=1) & ~high_degree).any()
                        for g, group in enumerate((low, ~low)):
                            assigned = int(((renters >= 0) & group).sum())
                            assert assigned <= caps[g] <= reachable[g]
                            counts[b, t, c, h, a, g] = [
                                assigned,
                                (blocks.any(axis=1) & group).sum(),
                                (high_degree & group).sum(),
                                caps[g],
                                reachable[g],
                            ]
                for h in range(len(PRIORITIES)):
                    for own_policy, (a0, a1) in enumerate([(0, 2), (1, 3)]):
                        before, after = assignments[h, [a0, a1]]
                        for band, group in enumerate(
                            (low & (degree > 0) & ~high_degree, low & high_degree)
                        ):
                            transitions[b, t, c, h, own_policy, band] = [
                                group.sum(),
                                (before & ~after & group).sum(),
                                (~before & after & group).sum(),
                            ]
                assert not assignments[:, :, degree == 0].any()

    def rates(block_weights):
        totals = np.tensordot(block_weights, counts, axes=(0, 0))
        den = np.tensordot(block_weights, denominators, axes=(0, 0))
        assert np.all(den > 0)
        return np.einsum(
            "tc,tchagm->hagm", weights, totals / den[:, :, None, None, :, None]
        )

    def transition_rates(block_weights):
        totals = np.tensordot(block_weights, transitions, axes=(0, 0))
        lower_counts = np.tensordot(
            block_weights, denominators[:, :, :, 0], axes=(0, 0)
        )
        overall = np.einsum(
            "tc,tchobm->hobm",
            weights,
            totals / lower_counts[:, :, None, None, None, None],
        )
        # Among each individual band, also report pooled raw counts in the result.
        return overall

    all_weights = np.ones(B, int)
    estimates = rates(all_weights)
    diff = transforms(estimates)
    raw = counts.sum(axis=(0, 1, 2))
    den_total = denominators.sum(axis=(0, 1, 2))
    for h, prior in enumerate(PRIORITIES):
        for a, policy in [(0, "uniform"), (3, "cheapest")]:
            for g, group in enumerate(GROUPS):
                record = ref["group_results"][prior][policy][group]
                assert record["renters"] == den_total[g]
                for k, key in [(0, "matched"), (1, "blockers"), (2, "choices_gt5")]:
                    assert record[key] == raw[h, a, g, k]
                    assert np.isclose(
                        100 * record["standardized_" + key],
                        100 * estimates[h, a, g, k],
                        atol=1e-10,
                    )
    assert np.allclose(diff[:, 0], diff[:, 1] + diff[:, 2])
    rng = np.random.default_rng(19112026)
    n_boot = 5000
    boot = np.empty((n_boot, *diff.shape))
    observed_transitions = transition_rates(all_weights)
    boot_transitions = np.empty((n_boot, *observed_transitions.shape))
    for j in range(n_boot):
        draw = rng.integers(B, size=B)
        multiplicity = np.bincount(draw, minlength=B)
        boot[j] = transforms(rates(multiplicity))
        boot_transitions[j] = transition_rates(multiplicity)

    def interval(values):
        return (100 * np.quantile(values, [0.025, 0.975])).tolist()

    arms = {}
    decomp = {}
    for h, prior in enumerate(PRIORITIES):
        arms[prior] = {}
        for a, arm in enumerate(ARMS):
            arms[prior][arm] = {
                g: {
                    k: float(100 * estimates[h, a, i, j]) for j, k in enumerate(METRICS)
                }
                for i, g in enumerate(GROUPS)
            }
        decomp[prior] = {}
        for j, measure in enumerate(METRICS[:2]):
            decomp[prior][measure] = {}
            for i, group in enumerate(GROUPS):
                decomp[prior][measure][group] = {
                    key: {
                        "pp": float(100 * diff[h, c, i, j]),
                        "ci95": interval(boot[:, h, c, i, j]),
                    }
                    for c, key in enumerate(CONTRASTS)
                }
            decomp[prior][measure]["lower_minus_higher"] = {
                key: {
                    "pp": float(100 * (diff[h, c, 0, j] - diff[h, c, 1, j])),
                    "ci95": interval(boot[:, h, c, 0, j] - boot[:, h, c, 1, j]),
                }
                for c, key in enumerate(CONTRASTS)
            }
    transition_results = {}
    raw_transitions = transitions.sum(axis=(0, 1, 2))
    for h, prior in enumerate(PRIORITIES):
        transition_results[prior] = {}
        for o, own in enumerate(("uniform", "cheapest")):
            table = {}
            for q, band in enumerate(("one_to_five", "six_or_more")):
                share, lost, gained = observed_transitions[h, o, q]
                draws = boot_transitions[:, h, o, q]
                table[band] = {
                    "raw_renters": int(raw_transitions[h, o, q, 0]),
                    "raw_lost": int(raw_transitions[h, o, q, 1]),
                    "raw_gained": int(raw_transitions[h, o, q, 2]),
                    "percent_of_all_lower": 100 * share,
                    "lost_pp_of_all_lower": 100 * lost,
                    "gained_pp_of_all_lower": 100 * gained,
                    "net_gain_pp_of_all_lower": 100 * (gained - lost),
                    "net_gain_ci95": interval(draws[:, 2] - draws[:, 1]),
                }
            total_net = sum(v["net_gain_pp_of_all_lower"] for v in table.values())
            ref_key = "switch_higher_when_lower_" + ("U", "C")[o]
            assert np.isclose(
                total_net, decomp[prior]["matched"]["lower"][ref_key]["pp"]
            )
            transition_results[prior][own] = table
    # Whole-market capacity is averaged within cells and standardized with common borough/month weights.
    capacity = np.einsum("tc,tcam->am", weights, whole.mean(axis=0)) / 64 * 100
    doc = {
        "design": "Paired mixed policies on 480 archived development markets; independent simultaneous DA and explicit full-graph blocker check.",
        "blocks": B,
        "market_seed": 20261112,
        "bootstrap_replicates": n_boot,
        "bootstrap_seed": 19112026,
        "input_hashes": ref["raw_hashes"],
        "arm_order": "first letter lower-income, second higher-income; U uniform feasible shortlist, C cheapest feasible shortlist",
        "units": "arm rates percent; standardized group numerator/denominator ratios. distinct_listings_alone divides distinct listings by renters and can exceed 100%.",
        "raw_renters": dict(zip(GROUPS, map(int, den_total))),
        "arm_rates": arms,
        "contrasts": decomp,
        "lower_transitions_when_only_higher_switch": transition_results,
        "whole_market_graph_capacity": {
            arm: {
                "percent_maximum_matched": float(capacity[a, 0]),
                "percent_distinct_listings": float(capacity[a, 1]),
            }
            for a, arm in enumerate(ARMS)
        },
        "validation": "Both priorities and both same-policy arms exactly matched all archived raw group match/block/choice counts and standardized rates. No visible blockers in any of 3,840 DA runs. All assignments visible and feasible.",
    }
    output.write_text(json.dumps(doc, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--catalogue-module", required=True, type=Path)
    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    main(args.data, args.catalogue_module, args.reference, args.out)
