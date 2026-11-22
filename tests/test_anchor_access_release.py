"""Sufficient aggregate counts reproduce paired inference without a packet."""
import json

import numpy as np
import pandas as pd
import pytest

from experiments_jobs.anchor_access import CONTRASTS, sha256
from experiments_jobs.anchor_access_analysis import PRIMARY
from experiments_jobs.anchor_access_metrics import aggregate_levels, paired_contrasts
from experiments_jobs.anchor_access_release import (
    compare_table, rebuild_tables, sufficient_seed_counts, verify_release)


def sample_rows():
    records = []
    arms = {"random": 0, "gpt": -1, "anchor_gpt": 3, "anchor_random": 1, "balanced_gpt": 2}
    for seed in range(20):
        for market, full in (("a", 10), ("b", 20)):
            for arm, offset in arms.items():
                base = dict(window="heldout_primary", variant="enriched", market=f"{seed}-{market}",
                            seed=seed, cell="sample", month="sample", arm=arm, baseline="", group="all")
                for metric, numerator, denominator in (
                    ("coverage_conditional", 3 + seed % 3 + offset, full),
                    ("placed", 10 + seed % 5 + offset, 24),
                    ("normalized_rank", 24 / (2 + seed % 4), 24),
                ):
                    records.append({**base, "metric": metric, "numerator": numerator, "denominator": denominator})
                if arm == "anchor_gpt":
                    records.append({**base, "baseline": "gpt", "metric": "gross_exits",
                                    "numerator": (seed % 3) / 20, "denominator": 24})
    return pd.DataFrame(records)


def manifest():
    return {"data_kind": "frozen_rental_replay", "primary_coverage_metric": "coverage_market_mean",
            "bootstrap_draws": 300, "inference_seed": 0}


def original_tables(rows):
    # Build the unaggregated comparison independently, keeping both markets.
    levels = rows[rows.baseline == ""].drop(columns="baseline")
    coverage = levels[levels.metric == "coverage_conditional"].copy()
    coverage.numerator = coverage.numerator / coverage.denominator
    coverage.denominator = 1
    coverage.metric = "coverage_market_mean"
    levels = pd.concat([levels, coverage], ignore_index=True)
    available = set(levels.arm)
    comparisons = [pair for pair in CONTRASTS if set(pair) <= available]
    kwargs = {"reps": 300, "expected_seeds": 20}
    return {"levels": aggregate_levels(levels, **kwargs),
            "transitions": aggregate_levels(rows[rows.baseline != ""], **kwargs),
            "contrasts": paired_contrasts(levels, comparisons, simultaneous=False, **kwargs),
            "primary": paired_contrasts(levels, PRIMARY, metrics=["coverage_market_mean", "placed"],
                                        groups=["all"], simultaneous=True, **kwargs)}


def test_market_coverage_transform_precedes_seed_pooling():
    counts = sufficient_seed_counts(sample_rows())
    part = counts[(counts.seed == 0) & (counts.arm == "random") & (counts.baseline == "")]
    conditional = part[part.metric == "coverage_conditional"].iloc[0]
    market_mean = part[part.metric == "coverage_market_mean"].iloc[0]
    assert conditional.numerator == 6 and conditional.denominator == 30
    assert market_mean.numerator == pytest.approx(.3 + .15) and market_mean.denominator == 2
    assert conditional.numerator / conditional.denominator != pytest.approx(market_mean.numerator / market_mean.denominator)
    assert not {"market", "cell", "month", "tie"} & set(counts.columns)


def test_seed_counts_preserve_every_estimate_and_interval_after_roundtrip(tmp_path):
    original = original_tables(sample_rows())
    path = tmp_path / "seed_counts.csv.gz"
    sufficient_seed_counts(sample_rows()).to_csv(path, index=False, float_format="%.17g")
    counts = pd.read_csv(path, keep_default_na=False, float_precision="round_trip")
    rebuilt = rebuild_tables(counts, manifest())
    for name in original:
        check = compare_table(rebuilt[name], original[name], name)
        assert check["matches"] and check["max_absolute_difference"] < 1e-10
    assert set(rebuilt["primary"].family_size) == {8}
    assert rebuilt["levels"].markets.max() == 20
    assert original["levels"].markets.max() == 40


def test_undefined_market_coverage_keeps_zero_support():
    rows = sample_rows()
    rows.loc[(rows.seed == 0) & (rows.metric == "coverage_conditional"), ["numerator", "denominator"]] = 0
    counts = sufficient_seed_counts(rows)
    coverage = counts[(counts.seed == 0) & (counts.metric == "coverage_market_mean")]
    assert (coverage.numerator == 0).all() and (coverage.denominator == 0).all()
    result = rebuild_tables(counts, manifest())["primary"]
    assert result[result.metric == "coverage_market_mean"]["mean"].isna().all()
    assert result.simultaneous_lo.isna().all()
    assert result[result.metric == "placed"]["mean"].notna().all()


def test_ties_duplicates_and_counterfactual_identity_are_checked():
    with pytest.raises(ValueError, match="randomization"):
        sufficient_seed_counts(sample_rows().assign(tie=0))
    counts = sufficient_seed_counts(sample_rows())
    with pytest.raises(ValueError, match="duplicate"):
        rebuild_tables(pd.concat([counts, counts.iloc[:1]]), manifest())
    with pytest.raises(ValueError, match="restricted"):
        rebuild_tables(counts, {**manifest(), "data_kind": "synthetic_validation"})


def test_changed_interval_is_rejected():
    original = original_tables(sample_rows())["primary"]
    changed = original.copy()
    changed.loc[0, "simultaneous_lo"] += .01
    with pytest.raises(ValueError, match="simultaneous_lo"):
        compare_table(original, changed, "primary")


def test_rebuild_only_needs_small_release_files(tmp_path):
    counts = sufficient_seed_counts(sample_rows())
    counts.to_csv(tmp_path / "seed_counts.csv.gz", index=False, float_format="%.17g")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest()))
    tables = original_tables(sample_rows())
    for name, table in tables.items():
        table.to_csv(tmp_path / f"{name}.csv", index=False)
    observed = tables["levels"][(tables["levels"].group == "all") & (tables["levels"].metric == "placed")]
    reproduction = observed[["window", "variant", "arm"]].copy()
    reproduction["observed_pp"] = 100 * observed["mean"]
    reproduction["published_pp"] = reproduction.observed_pp
    reproduction["difference_pp"] = 0
    reproduction.to_csv(tmp_path / "reproduction.csv", index=False)
    filenames = ["seed_counts.csv.gz", "manifest.json", "reproduction.csv", *[f"{name}.csv" for name in tables]]
    release = {"source_kind": "reconstructed_rental_counterfactual",
               "released_file_sha256": {name: sha256(tmp_path / name) for name in filenames}}
    (tmp_path / "release_manifest.json").write_text(json.dumps(release))
    result = verify_release(tmp_path)
    assert result["packet_or_raw_inputs_used"] is False
    assert all(check["matches"] for check in result["checks"])
    assert (tmp_path / "rebuild_validation.json").exists()
    with (tmp_path / "seed_counts.csv.gz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash differs"):
        verify_release(tmp_path)
