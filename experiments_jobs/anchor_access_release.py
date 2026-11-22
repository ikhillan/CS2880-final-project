"""Release sufficient seed counts and rebuild rental counterfactual intervals.

Release creation validates the complete private replay logs. Rebuilding needs
only seed_counts.csv.gz, the released tables/manifests, and the analysis code;
it never opens a replay packet, raw data, or individual/market outcome logs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from experiments_jobs.anchor_access import CONTRASTS, sha256
from experiments_jobs.anchor_access_analysis import PRIMARY, load_runs
from experiments_jobs.anchor_access_metrics import aggregate_levels, paired_contrasts

SEED_KEYS = ["window", "variant", "seed", "arm", "baseline", "group", "metric"]
TABLES = ("levels", "transitions", "contrasts", "primary", "reproduction")
MARKET_COLUMNS = {"markets", "markets_expected", "missing_markets", "paired_markets", "unpaired_markets"}
ROOT = Path(__file__).resolve().parents[1]


def sufficient_seed_counts(rows):
    """Sum already tie-averaged counts; transform market coverage BEFORE pooling."""
    rows = rows.copy()
    if "tie" in rows:
        raise ValueError("Average and validate policy randomization before releasing counts")
    if "coverage_market_mean" in set(rows.metric):
        raise ValueError("Market coverage must be derived once, before seed aggregation")
    coverage = rows[(rows.baseline == "") & (rows.metric == "coverage_conditional")].copy()
    if coverage.empty:
        raise ValueError("Conditional coverage is required to preserve the rental estimand")
    positive = coverage.denominator > 0
    coverage.numerator = np.divide(coverage.numerator, coverage.denominator,
                                   out=np.zeros(len(coverage)), where=positive)
    coverage.denominator = positive.astype(int)
    coverage.metric = "coverage_market_mean"
    combined = pd.concat([rows, coverage], ignore_index=True)
    counts = combined.groupby(SEED_KEYS, as_index=False, sort=True)[["numerator", "denominator"]].sum()
    return counts[SEED_KEYS + ["numerator", "denominator"]]


def rebuild_tables(counts, manifest):
    """The only numerical outcome input is the sufficient seed-count table."""
    required = set(SEED_KEYS + ["numerator", "denominator"])
    if set(counts.columns) != required or counts.duplicated(SEED_KEYS).any():
        raise ValueError("Seed counts have unexpected columns or duplicate records")
    if manifest["data_kind"] != "frozen_rental_replay":
        raise ValueError("This release is restricted to reconstructed rental counterfactuals")
    if manifest.get("primary_coverage_metric") != "coverage_market_mean":
        raise ValueError("Rental primary coverage must retain the market-mean estimand")
    rows = counts.copy()
    # A dummy identifier permits the unchanged paired analysis API to treat
    # each seed as one sufficient block. It is not an observed market ID.
    rows["market"] = "seed-block-" + rows.seed.astype(str)
    levels = rows[rows.baseline == ""].drop(columns="baseline")
    transitions = rows[rows.baseline != ""]
    available = set(levels.arm)
    comparisons = [pair for pair in CONTRASTS if pair[0] in available and pair[1] in available]
    kwargs = {"reps": int(manifest["bootstrap_draws"]), "rng_seed": int(manifest["inference_seed"]),
              "expected_seeds": 20}
    return {
        "levels": aggregate_levels(levels, **kwargs),
        "transitions": aggregate_levels(transitions, **kwargs),
        "contrasts": paired_contrasts(levels, comparisons, simultaneous=False, **kwargs),
        "primary": paired_contrasts(levels, PRIMARY, metrics=["coverage_market_mean", "placed"],
                                    groups=["all"], simultaneous=True, **kwargs),
    }


def compare_table(rebuilt, released, name, tolerance=1e-10):
    """Compare all estimands, intervals and support flags except market counts."""
    if set(rebuilt.columns) != set(released.columns):
        raise ValueError(f"{name}: output columns differ")
    keys = [key for key in ("window", "variant", "arm", "comparator", "baseline", "group", "metric")
            if key in rebuilt]
    if rebuilt.duplicated(keys).any() or released.duplicated(keys).any():
        raise ValueError(f"{name}: duplicate result identifiers")
    a = rebuilt.sort_values(keys).reset_index(drop=True)
    b = released.sort_values(keys).reset_index(drop=True)
    if len(a) != len(b) or not a[keys].equals(b[keys]):
        raise ValueError(f"{name}: output identities differ")
    max_difference, numeric = 0.0, []
    for column in a.columns:
        if column in MARKET_COLUMNS or column in keys:
            continue
        if pd.api.types.is_numeric_dtype(a[column]) and not pd.api.types.is_bool_dtype(a[column]):
            x = a[column].to_numpy(float)
            y = pd.to_numeric(b[column], errors="raise").to_numpy(float)
            if not np.allclose(x, y, rtol=0, atol=tolerance, equal_nan=True):
                raise ValueError(f"{name}: {column} does not rebuild within {tolerance:g}")
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.any():
                max_difference = max(max_difference, float(np.max(np.abs(x[valid] - y[valid]))))
            numeric.append(column)
        else:
            # CSV empty cells represent unavailable annotations, not zeroes.
            x, y = a[column].fillna("").astype(str), b[column].fillna("").astype(str)
            if not x.equals(y):
                raise ValueError(f"{name}: {column} annotations differ")
    return {"table": name, "rows": len(a), "numeric_columns_checked": numeric,
            "max_absolute_difference": max_difference, "matches": True}


def compare_reproduction(levels, released, tolerance=1e-10):
    """Recover original placement means from seed counts; retain published targets."""
    placed = levels[(levels.group == "all") & (levels.metric == "placed")]
    merged = released.merge(placed[["window", "variant", "arm", "mean"]],
                            on=["window", "variant", "arm"], how="left", validate="one_to_one")
    observed = 100 * merged["mean"].to_numpy(float)
    if (not np.isfinite(observed).all()
            or not np.allclose(observed, merged.observed_pp, rtol=0, atol=tolerance)
            or not np.allclose(observed - merged.published_pp, merged.difference_pp, rtol=0, atol=tolerance)):
        raise ValueError("Archived placement reproduction cannot be recovered from seed counts")
    return {"table": "reproduction", "rows": len(merged), "matches": True,
            "max_absolute_difference": float(np.max(np.abs(observed - merged.observed_pp)))}


def _read_counts(path):
    return pd.read_csv(path, keep_default_na=False,
                       dtype={key: str for key in SEED_KEYS if key != "seed"}, float_precision="round_trip")


def verify_release(folder, *, write_report=True):
    """Rebuild without opening runs, packets, household data or listing data."""
    folder = Path(folder)
    release = json.loads((folder / "release_manifest.json").read_text())
    if release.get("source_kind") != "reconstructed_rental_counterfactual":
        raise ValueError("The release must be labeled as rental counterfactual evidence")
    for filename, digest in release["released_file_sha256"].items():
        if Path(filename).name != filename or sha256(folder / filename) != digest:
            raise ValueError(f"Released file hash differs: {filename}")
    manifest = json.loads((folder / "manifest.json").read_text())
    counts = _read_counts(folder / "seed_counts.csv.gz")
    rebuilt = rebuild_tables(counts, manifest)
    checks = [compare_table(table, pd.read_csv(folder / f"{name}.csv", float_precision="round_trip"), name)
              for name, table in rebuilt.items()]
    checks.append(compare_reproduction(rebuilt["levels"], pd.read_csv(folder / "reproduction.csv", float_precision="round_trip")))
    result = {"status": "all_released_estimates_and_intervals_reproduced", "absolute_tolerance": 1e-10,
              "numerical_outcome_input": "seed_counts.csv.gz", "packet_or_raw_inputs_used": False,
              "ignored_original_market_count_columns": sorted(MARKET_COLUMNS), "checks": checks,
              "release_helper_sha256": sha256(Path(__file__))}
    if write_report:
        (folder / "rebuild_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def create_release(runs, analysis, out):
    rows, audit, reference, manifests = load_runs(runs)
    if reference["data_kind"] != "frozen_rental_replay" or rows.market.nunique() != 1280:
        raise ValueError("Release requires all 1,280 original reconstructed rental markets")
    analysis, out = Path(analysis), Path(out)
    manifest = json.loads((analysis / "manifest.json").read_text())
    for key in ("data_kind", "packet_sha256", "source_hashes"):
        if manifest[key] != reference[key]:
            raise ValueError(f"Analysis and validated replay differ: {key}")
    if (manifest["markets"] != 1280 or manifest["complete_shards"] != len(manifests)
            or manifest["ties_per_market"] != reference["tie_draws"]
            or manifest["original_placement_reproduced"] is not True):
        raise ValueError("Analysis did not validate the complete original rental panel")
    if manifest["analysis_source_sha256"] != sha256(ROOT / "experiments_jobs/anchor_access_analysis.py"):
        raise ValueError("Use the frozen analysis source recorded by this run")
    metrics_path = "experiments_jobs/anchor_access_metrics.py"
    if reference["source_hashes"][metrics_path] != sha256(ROOT / metrics_path):
        raise ValueError("Use the frozen bootstrap/metrics source recorded by this run")
    counts = sufficient_seed_counts(rows)
    out.mkdir(parents=True, exist_ok=True)
    counts.to_csv(out / "seed_counts.csv.gz", index=False, float_format="%.17g",
                  compression={"method": "gzip", "mtime": 0})
    filenames = ["seed_counts.csv.gz", "manifest.json", *[f"{name}.csv" for name in TABLES]]
    for name in filenames[1:]:
        source, destination = analysis / name, out / name
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
    panel = reference["panel_manifest"]
    panel_counts = rows[["window", "variant", "market", "seed"]].drop_duplicates().groupby(
        ["window", "variant"]).agg(markets=("market", "nunique"), seeds=("seed", "nunique")).reset_index()
    release = {
        "release_format_version": 1, "source_kind": "reconstructed_rental_counterfactual",
        "counts_are_not_field_outcomes": True, "post_hoc_followup": True,
        "packet_sha256": reference["packet_sha256"], "packet_included": False,
        "individual_or_market_outcome_records_included": False,
        "seed_count_columns": list(counts.columns), "seed_count_rows": len(counts),
        "panel_counts": panel_counts.to_dict("records"),
        "planned_markets": len(panel["planned_markets"]), "skipped_markets": len(panel["skipped_markets"]),
        "input_sha256": panel["input_hashes"], "export_source_sha256": panel["source_hashes"],
        "evaluation_source_sha256": reference["source_hashes"],
        "analysis_source_sha256": manifest["analysis_source_sha256"],
        "release_helper_sha256": sha256(Path(__file__)),
        "run_manifests": [{key: run[key] for key in ("shard", "shards", "markets", "tie_draws", "versions")}
                          for run in manifests],
        "aggregation": "Validate and average policy ties within market; transform market coverage; sum counts within seed.",
        "uncertainty_scope": "Shared paired seed-block bootstrap of modeled rental counterfactuals, not uncertainty about field effects.",
        "coverage_market_mean": "Within each seed: sum defined market coverage fractions / number of defined markets. Zero denominators remain undefined.",
        "rebuild_command": "python -m experiments_jobs.anchor_access_release rebuild --release " + str(out),
        "released_file_sha256": {name: sha256(out / name) for name in filenames},
    }
    (out / "release_manifest.json").write_text(json.dumps(release, indent=2) + "\n")
    result = verify_release(out)
    print(json.dumps({"release": str(out), "seed_rows": len(counts), "status": result["status"]}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("release")
    make.add_argument("--runs", type=Path, required=True)
    make.add_argument("--analysis", type=Path, required=True)
    make.add_argument("--out", type=Path, default=ROOT / "findings/rental-anchor-access")
    rebuild = sub.add_parser("rebuild")
    rebuild.add_argument("--release", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "release":
        create_release(args.runs, args.analysis, args.out)
    else:
        print(json.dumps(verify_release(args.release), indent=2))


if __name__ == "__main__":
    main()
