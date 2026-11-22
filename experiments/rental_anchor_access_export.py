"""Export the public cached-GPT rental experiments for anchor/access replay.

No model calls. Rebuild every planned market from the original configs, verify
every saved candidate identity, reproduce archived placement/anchor means, and
write an anonymous NumPy packet. Raw inputs remain local and git-ignored.

    python -m experiments.rental_anchor_access_export --out results/anchor-access/rental-panel.npz
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.config import expand_labeled, load_config
from experiments import rental_capacity_anchor as archived
from experiments_jobs.anchor_access import FORMAT_VERSION, save_packet, sha256, versions
from experiments_jobs.capacity_anchor import apply_anchor
from hiddenoptions import data, markets as rental_markets
from hiddenoptions.exposure import LlmRecommendations, build_exposure, candidate_digest
from hiddenoptions.matching import deferred_acceptance

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ("stage1_pilot", "heldout_primary", "heldout_replication")
MONTHS = ("2026-06", "2026-07", "2026-08", "2026-09", "2026-10", "2025-06", "2025-07", "2025-08")
EXPECTED = {"stage1_pilot": 480, "heldout_primary": 320, "heldout_replication": 480}
EXPOSURES = {"random": "uniform_k5", "gpt": "gpt_k5", "score": "transparent_k5"}
RULES = {"random": "Random (5)", "gpt": "GPT (5)", "score": "Honest ranker (5)"}


def input_paths():
    return [data.PUMS_HOUSEHOLDS, data.CPI_CSV,
            *[data.FIRSTMOVER_CLEAN / f"firstmover_listings_{month}.csv.gz" for month in MONTHS],
            *[ROOT / "findings/gpt-luna-rentals" / window / "gpt_recs.jsonl.gz" for window in WINDOWS]]


def read_saved(path):
    """Load the frozen public replies without manufacturing any fallback row."""
    saved = LlmRecommendations()
    count = 0
    with gzip.open(path, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if (row.get("exposure") != "gpt_k5" or row.get("status") != "ok"
                    or row.get("count") != 5 or row.get("rep", 0) != 0
                    or len(row.get("listings", [])) != 5
                    or len(set(row["listings"])) != 5
                    or any(row.get(key, 0) for key in ("invalid_ids", "duplicate_ids", "extra_ids"))
                    or not row.get("candidates_sha256")):
                raise ValueError("The public GPT cache contains a non-original or invalid response")
            table = saved.table.setdefault(row["market_id"], {}).setdefault("gpt_k5", {})
            key = (int(row["renter"]), 0)
            if key in table:
                raise ValueError(f"Duplicate cached response: {row['market_id']}: {key}")
            table[key] = row
            count += 1
    return saved, count


def exact_score_ranks(E, scores):
    """Dense ascending integer scores; preserve every original ordering/tie."""
    if not np.isfinite(scores[E]).all():
        raise ValueError("Nonfinite feasible observable score")
    ranks = np.zeros(E.shape, dtype=np.int64)
    for i in range(len(E)):
        feasible = scores[i, E[i]]
        ranks[i, E[i]] = np.searchsorted(np.unique(feasible), feasible)
    return ranks


def validated_block(market, saved, exposures):
    inst, E = market.inst, np.asarray(market.inst.E, dtype=bool)
    degree = E.sum(1)
    replies = saved.table.get(market.spec.market_id, {}).get("gpt_k5", {})
    required = {(int(i), 0) for i in np.flatnonzero(degree > 5)}
    if set(replies) != required:
        raise ValueError(f"Cached applicant coverage mismatch: {market.spec.market_id}")
    gpt_order = np.full(E.shape, -1, dtype=np.int64)
    for i in range(len(E)):
        if degree[i] > 5:
            row = replies[(i, 0)]
            if row["candidates_sha256"] != candidate_digest(market, i):
                raise ValueError(f"Candidate identity mismatch: {market.spec.market_id}: {i}")
            choices = row["listings"]
            if any(not isinstance(j, int) or not 0 <= j < E.shape[1] or not E[i, j] for j in choices):
                raise ValueError(f"Invalid feasible choice: {market.spec.market_id}: {i}")
        else:
            # No model call was made: this whole feasible set was displayed.
            # Fixed quotas leave this row unchanged under menu optimization.
            choices = np.flatnonzero(E[i]).tolist()
        gpt_order[i, :len(choices)] = choices
    block = {"E": E, "rank_b": inst.rank_b, "rank_s": inst.rank_s,
             "s2p": np.arange(E.shape[1], dtype=np.int64),
             "lower_group": market.renters.lower_income.to_numpy(bool),
             "score_ranks": exact_score_ranks(E, market.u_observable), "gpt_order": gpt_order}
    for arm, name in EXPOSURES.items():
        built = build_exposure(market, exposures[name], saved)
        if built is None:
            raise ValueError(f"Missing {arm} menu: {market.spec.market_id}")
        shown, stats = built
        if arm == "gpt" and any(stats.get(key, 0) for key in
                                ("missing", "refused", "unparseable", "invalid_ids", "dropped_outside_E", "partial_fill", "full_fallback")):
            raise ValueError(f"GPT fallback is forbidden: {market.spec.market_id}: {stats}")
        if (shown & ~E).any() or not np.array_equal(shown.sum(1), np.minimum(5, degree)):
            raise ValueError(f"Incorrect menu budget: {market.spec.market_id}: {arm}")
        block["shown_" + arm] = shown
    return block, len(required)


def reference_rows(market, label, window, block, saved, exposures):
    """Use the archived rental deletion rule, including its original RNG salt."""
    archived._llm = saved
    rows = []
    for arm, name in EXPOSURES.items():
        shown = block["shown_" + arm]
        anchored, changed = apply_anchor(market.inst.E, shown, block["s2p"],
                                         archived.drop_order(market, exposures[name]))
        before = deferred_acceptance(market.inst, shown)[0]
        after = deferred_acceptance(market.inst, anchored)[0]
        rows.append({"window": window, "variant": label, "arm": arm,
                     "market": market.spec.market_id, "placed": 100 * (before >= 0).mean(),
                     "anchor_placed": 100 * (after >= 0).mean(),
                     "lists_changed": 100 * changed / market.inst.n_b})
    return rows


def verify_reproduction(rows):
    table, checks = pd.DataFrame(rows), []
    for window in WINDOWS:
        ref = pd.read_csv(ROOT / "findings/rental-capacity-gpt" / window / "anchor.csv")
        levels = pd.read_csv(ROOT / "findings/rental-capacity-gpt" / window / "capacity_levels.csv")
        for (variant, arm), part in table[table.window == window].groupby(["variant", "arm"]):
            expected = ref[(ref.variant == variant) & (ref.rule == RULES[arm])]
            expected_levels = levels[(levels.variant == variant) & (levels.rule == RULES[arm])]
            if len(expected) != 1 or len(expected_levels) != 1 or len(part) != int(expected_levels.iloc[0].markets):
                raise ValueError(f"Archived market panel mismatch: {window}, {variant}, {arm}")
            for metric in ("placed", "anchor_placed", "lists_changed"):
                observed, target = float(part[metric].mean()), float(expected.iloc[0][metric])
                if not np.isclose(observed, target, rtol=0, atol=1e-10):
                    raise ValueError(f"Archived {metric} differs: {window}, {variant}, {arm}: {observed} vs {target}")
                checks.append({"window": window, "variant": variant, "arm": arm, "metric": metric,
                               "observed_percent": observed, "archived_percent": target, "matches": True})
    return checks


def export(out):
    paths = input_paths()
    if missing := [str(path) for path in paths if not path.is_file()]:
        raise FileNotFoundError("Required frozen rental inputs are missing: " + ", ".join(missing))
    provenance = {str(path.relative_to(ROOT)): sha256(path) for path in paths}
    # Rebuild from the hashed household source, not an unversioned derived cache.
    renters = data.build_renter_profiles()
    renters["income"] = renters.income_2024 * data.cpi_value(data.BASE_MONTH)[0] / data.cpi_2024_annual()
    original_loader = rental_markets.load_renters
    try:
        rental_markets.load_renters = lambda: renters.copy()
        catalog = rental_markets.Catalog()
    finally:
        rental_markets.load_renters = original_loader
    arrays, records, planned, skipped, audits, cache_counts = {}, [], [], [], [], {}
    for window in WINDOWS:
        cfg = load_config(ROOT / "configs" / f"{window}.json")
        saved, cached = read_saved(ROOT / "findings/gpt-luna-rentals" / window / "gpt_recs.jsonl.gz")
        exposures = {ex["name"]: ex for ex in cfg["exposures"]}
        used, seen = 0, set()
        for spec, label in expand_labeled(cfg):
            record = {"window": window, "variant": label, "market": spec.market_id,
                      "seed": spec.seed, "cell": spec.borough, "month": spec.month,
                      "domain": "housing", "exposure_names": EXPOSURES, "spec": asdict(spec)}
            planned.append(record)
            try:
                market = rental_markets.build_market(spec, catalog)
            except rental_markets.InsufficientSupply as error:
                skipped.append({**record, "reason": str(error), "exception": "InsufficientSupply"})
                continue
            block, n_calls = validated_block(market, saved, exposures)
            used += n_calls
            seen.add(spec.market_id)
            arrays.update({f"m{len(records):05d}_" + key: value for key, value in block.items()})
            records.append(record)
            audits.extend(reference_rows(market, label, window, block, saved, exposures))
            if len(records) % 40 == 0:
                print(f"Validated/exported {len(records)} rental markets", flush=True)
        if len(seen) != EXPECTED[window] or seen != set(saved.table) or used != cached:
            raise ValueError(f"Original GPT panel/cache not reproduced: {window}: {len(seen)} markets, {used}/{cached} calls")
        cache_counts[window] = {"markets": len(seen), "cached_calls": cached, "candidate_hashes_verified": used}
    if len(records) != 1280 or len(planned) != 1600 or len(skipped) != 320:
        raise ValueError("The three original rental panels do not have their archived counts")
    checks = verify_reproduction(audits)
    source_paths = ["experiments/rental_anchor_access_export.py", "experiments/rental_capacity_anchor.py",
                    "experiments/config.py", "hiddenoptions/data.py", "hiddenoptions/markets.py",
                    "hiddenoptions/exposure.py", "hiddenoptions/matching.py", "experiments_jobs/capacity_anchor.py",
                    *[f"configs/{window}.json" for window in WINDOWS],
                    *[f"findings/rental-capacity-gpt/{window}/{name}.csv"
                      for window in WINDOWS for name in ("anchor", "capacity_levels")]]
    metadata = {"format_version": FORMAT_VERSION, "data_kind": "frozen_rental_replay", "domain": "housing",
                "lower_group_label": "lower_income", "versions": versions(), "input_hashes": provenance,
                "source_hashes": {path: sha256(ROOT / path) for path in source_paths},
                "markets": records, "planned_markets": planned, "skipped_markets": skipped,
                "counts": {"planned": len(planned), "exported": len(records), "skipped": len(skipped)},
                "skip_boroughs": dict(Counter(row["cell"] for row in skipped)), "cache_validation": cache_counts,
                "archived_reproduction_checks": checks, "api_calls": 0,
                "preferences": "original modeled rental preferences; not observed household choices",
                "score_convention": "exact integer dense rank of u_observable; no quantization",
                "gpt_whole_set_order": "ascending local indices when degree<=5; no model preference observed",
                "contains_raw_identifiers_or_salaries": False}
    save_packet(out, metadata, arrays)
    print(json.dumps({"out": str(out), "sha256": sha256(out), "counts": metadata["counts"],
                      "verified_candidate_hashes": sum(x["candidate_hashes_verified"] for x in cache_counts.values()),
                      "archived_reproduction_checks_passed": len(checks)}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/anchor-access/rental-panel.npz")
    export(parser.parse_args().out)


if __name__ == "__main__":
    main()
