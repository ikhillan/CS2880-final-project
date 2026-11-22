"""Validate complete replay shards and summarize paired jobs anchor outcomes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments_jobs.anchor_access import ROOT, CONFIGS, CONTRASTS, sha256
from experiments_jobs.anchor_access_metrics import aggregate_levels, paired_contrasts
from experiments_jobs.config import expand_labeled, load_config

KEYS = ["window", "variant", "market", "seed", "cell", "month", "arm", "baseline", "group", "metric"]
PRIMARY = CONTRASTS[:4]


def load_runs(folder, allow_synthetic=False):
    folder = Path(folder)
    manifests = [json.loads(p.read_text()) for p in sorted(folder.glob("run-*.json"))]
    if not manifests:
        raise ValueError("No complete run manifests found")
    reference = manifests[0]
    for manifest in manifests:
        for key in ("data_kind", "packet_sha256", "shards", "tie_draws", "source_hashes", "panel_manifest"):
            if manifest[key] != reference[key]:
                raise ValueError(f"Incompatible shards: {key}")
    ids = [m["shard"] for m in manifests]
    if sorted(ids) != list(range(reference["shards"])):
        raise ValueError("Missing or duplicate shards; no partial-panel results will be reported")
    if reference["data_kind"] not in ("frozen_jobs_replay", "frozen_rental_replay") and not allow_synthetic:
        raise ValueError("Synthetic runs cannot be summarized as empirical jobs evidence")
    metrics, checks = [], []
    for m in manifests:
        stem = f"{m['shard']:02d}"
        raw_part = pd.read_csv(folder / f"metrics-{stem}.csv.gz", keep_default_na=False)
        checks.append(pd.read_csv(folder / f"checks-{stem}.csv.gz", keep_default_na=False))
        for table in (raw_part, checks[-1]):
            if not ((table.seed % reference["shards"]) == m["shard"]).all():
                raise ValueError("Record in the wrong seed shard")
            if table.market.nunique() != m["markets"]:
                raise ValueError("Market count differs from run manifest")
        if raw_part.duplicated(KEYS + ["tie"]).any():
            raise ValueError("Duplicate metric records")
        counts = raw_part.groupby(KEYS, sort=False).tie.agg(["min", "max", "size", "nunique"])
        varied = counts.index.get_level_values("arm").str.endswith("_ties")
        required_count = np.where(varied, reference["tie_draws"], 1)
        required_min = np.where(varied, 0, -1)
        required_max = np.where(varied, reference["tie_draws"] - 1, -1)
        if not ((counts["size"] == required_count) & (counts["nunique"] == required_count)
                & (counts["min"] == required_min) & (counts["max"] == required_max)).all():
            raise ValueError("Incomplete or duplicated policy randomization")
        # Collapse each shard before reading the next to bound peak RAM.
        metrics.append(raw_part.groupby(KEYS, as_index=False, sort=False)[["numerator", "denominator"]].mean())
        del raw_part
    raw, audit = pd.concat(metrics, ignore_index=True), pd.concat(checks, ignore_index=True)
    if raw.duplicated(KEYS).any():
        raise ValueError("Duplicate metric records")
    if not audit.menu_budget_kept.all():
        raise ValueError("A menu-budget check failed")
    if reference["data_kind"] == "frozen_jobs_replay":
        variants = set(raw.variant)
        if not variants <= {"primary", "natural"} or "primary" not in variants:
            raise ValueError("Unsupported or missing primary variant")
        expected = set()
        for path in CONFIGS:
            cfg = load_config(ROOT / path)
            window = "frozen" if cfg["name"] == "jobs_frozen" else "replication"
            expected.update((window, label, spec.market_id, spec.seed)
                            for spec, label in expand_labeled(cfg) if label in variants)
        actual = set(raw[["window", "variant", "market", "seed"]].itertuples(index=False, name=None))
        if actual != expected:
            raise ValueError("Frozen market identities do not match the complete prespecified panel")
    elif reference["data_kind"] == "frozen_rental_replay":
        from experiments.config import expand_labeled as rental_specs
        variants = set(raw.variant)
        if variants != {"primary", "enriched"}:
            raise ValueError("Rental follow-up must include both sampling designs")
        expected = set()
        for window in ("stage1_pilot", "heldout_primary", "heldout_replication"):
            expected.update((window, label, spec.market_id, spec.seed)
                            for spec, label in rental_specs(load_config(ROOT / f"configs/{window}.json"))
                            if label in variants)
        fields = ("window", "variant", "market", "seed")
        planned = {tuple(r[k] for k in fields) for r in reference["panel_manifest"]["planned_markets"]}
        skipped = {tuple(r[k] for k in fields) for r in reference["panel_manifest"]["skipped_markets"]}
        actual = set(raw[list(fields)].itertuples(index=False, name=None))
        if planned != expected or actual & skipped or actual | skipped != planned:
            raise ValueError("Rental panel coverage/declared omissions differ from frozen configurations")
    # Mean counts retain equal market weight while integrating algorithm ties.
    # Conditional transition rates are ratios of expected counts, not averages
    # of conditional rates over individual randomizations.
    return raw, audit, reference, manifests


def reproduce_original(audit, data_kind="frozen_jobs_replay"):
    records = []
    if data_kind == "frozen_rental_replay":
        for window in ("stage1_pilot", "heldout_primary", "heldout_replication"):
            source = ROOT / f"findings/rental-capacity-gpt/{window}/anchor.csv"
            table = pd.read_csv(source)
            for variant in ("primary", "enriched"):
                for arm, label in (("gpt", "GPT (5)"), ("random", "Random (5)"), ("score", "Honest ranker (5)")):
                    reference = table[(table.variant == variant) & (table.rule == label)].iloc[0]
                    for name, column in ((arm, "placed"), ("anchor_" + arm, "anchor_placed")):
                        part = audit[(audit.window == window) & (audit.variant == variant) & (audit.arm == name)]
                        if part.empty or set(part.tie) != {-1}:
                            raise ValueError("Original rental arm is absent or has extra randomizations")
                        value = float(100 * (part.placed / part.n).mean())
                        expected = float(reference[column])
                        records.append({"window": window, "variant": variant, "arm": name,
                                        "observed_pp": value, "published_pp": expected,
                                        "difference_pp": value - expected, "source_sha256": sha256(source)})
        check = pd.DataFrame(records)
        if not np.allclose(check.observed_pp, check.published_pp, atol=1e-9, rtol=0):
            raise ValueError("Published rental placement reproduction failed:\n" + check.to_string(index=False))
        return check
    for window, suffix in (("frozen", ""), ("replication", "replication/")):
        source = ROOT / "findings/jobs-gpt-frozen" / suffix / "anchor.csv"
        table = pd.read_csv(source)
        for arm, label in (("gpt", "GPT (5 jobs)"), ("random", "Random (5 jobs)"),
                           ("score", "Honest ranker (5 jobs)")):
            reference = table[(table.variant == "primary") & (table.rule == label)].iloc[0]
            for name, column in ((arm, "placed"), ("anchor_" + arm, "anchor_placed")):
                part = audit[(audit.window == window) & (audit.variant == "primary") & (audit.arm == name)]
                if part.empty or set(part.tie) != {-1}:
                    raise ValueError("Original arm is absent or has extra randomizations")
                value = float(100 * (part.placed / part.n).mean())
                expected = float(reference[column])
                records.append({"window": window, "arm": name, "observed_pp": value,
                                "published_pp": expected, "difference_pp": value - expected,
                                "source_sha256": sha256(source)})
    check = pd.DataFrame(records)
    if not np.allclose(check.observed_pp, check.published_pp, atol=1e-9, rtol=0):
        raise ValueError("Published placement reproduction failed:\n" + check.to_string(index=False))
    return check


def analyze(runs, out, allow_synthetic=False, reps=10000):
    rows, checks, reference, manifests = load_runs(runs, allow_synthetic)
    empirical = reference["data_kind"] in ("frozen_jobs_replay", "frozen_rental_replay")
    reproduction = reproduce_original(checks, reference["data_kind"]) if empirical else pd.DataFrame()
    levels = rows[rows.baseline == ""].drop(columns="baseline")
    transitions = rows[rows.baseline != ""]
    primary_coverage = "coverage_conditional"
    if reference["data_kind"] == "frozen_rental_replay":
        # Preserve the released rental accuracy estimand (equal market means).
        # Keep pooled-recipient coverage as an additional, separately named endpoint.
        market_coverage = levels[levels.metric == "coverage_conditional"].copy()
        positive = market_coverage.denominator > 0
        market_coverage.numerator = np.divide(market_coverage.numerator, market_coverage.denominator,
                                             out=np.zeros(len(market_coverage)), where=positive)
        market_coverage.denominator = positive.astype(int)
        market_coverage.metric = "coverage_market_mean"
        levels = pd.concat([levels, market_coverage], ignore_index=True)
        primary_coverage = "coverage_market_mean"
    available = set(levels.arm)
    contrasts = [p for p in CONTRASTS if p[0] in available and p[1] in available]
    kwargs = {"reps": reps, "expected_seeds": 20 if empirical else None}
    level_summary = aggregate_levels(levels, **kwargs)
    transition_summary = aggregate_levels(transitions, **kwargs)
    all_contrasts = paired_contrasts(levels, contrasts, simultaneous=False, **kwargs)
    primary = paired_contrasts(levels, PRIMARY, metrics=[primary_coverage, "placed"],
                               groups=["all"], simultaneous=True, **kwargs)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, table in (("levels", level_summary), ("transitions", transition_summary),
                        ("contrasts", all_contrasts), ("primary", primary), ("reproduction", reproduction)):
        table.to_csv(out / f"{name}.csv", index=False)
    manifest = {"data_kind": reference["data_kind"], "packet_sha256": reference["packet_sha256"],
                "complete_shards": len(manifests), "markets": rows.market.nunique(),
                "bootstrap_draws": reps, "inference_seed": 0, "ties_per_market": reference["tie_draws"],
                "tie_estimand": "rates_of_mean_counts_within_market_then_mean_seed_rates",
                "post_hoc_followup": True, "primary_family_size_per_window_variant": 8,
                "primary_coverage_metric": primary_coverage,
                "original_placement_reproduced": True if empirical else None,
                "source_hashes": reference["source_hashes"],
                "analysis_source_sha256": sha256(Path(__file__)),
                "counts_are_not_field_outcomes": True}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(primary[["window", "variant", "arm", "comparator", "metric", "mean", "lo", "hi", "status"]].to_string(index=False))
    print(f"Wrote {out}; data kind: {reference['data_kind']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--allow-synthetic", action="store_true")
    args = ap.parse_args()
    analyze(args.runs, args.out, args.allow_synthetic, args.reps)


if __name__ == "__main__":
    main()
