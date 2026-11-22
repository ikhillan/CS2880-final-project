"""Small independent checks for menu coverage, access transitions and pairing."""
import numpy as np
import pandas as pd
import pytest

from hiddenoptions.matching import Instance, NOT_EDGE
from experiments_jobs.anchor_access_metrics import evaluate, transition, aggregate_levels, paired_contrasts


def market():
    # Posting 0 has two interchangeable slots. Applicant 3 has no options.
    rb = np.array([[0, 1, 2, NOT_EDGE],
                   [NOT_EDGE, NOT_EDGE, 0, 1],
                   [0, 1, NOT_EDGE, NOT_EDGE],
                   [NOT_EDGE] * 4])
    rs = np.where(rb < NOT_EDGE, np.broadcast_to(np.arange(4)[:, None], rb.shape), NOT_EDGE)
    inst = Instance(rb, rs, np.arange(4), np.arange(4) + 4, np.ones(8, dtype=int))
    groups = {"lower": np.array([True, False, True, False]),
              "higher": np.array([False, True, False, True])}
    return inst, groups, np.array([0, 0, 1, 2])


def value(result, metric):
    item = result[metric]
    return item["numerator"] / item["denominator"]


def test_coverage_denominators_and_unmatched_rank():
    inst, groups, slots = market()
    shown = np.array([[False, False, True, False],
                      [False, False, True, False],
                      [True, True, False, False],
                      [False] * 4])
    outcome = evaluate(inst, shown, np.array([2, -1, 0, -1]),
                       np.array([0, 2, 1, -1]), groups, slots)
    all_people = outcome["all"]
    assert all_people["coverage_conditional"] == {"numerator": 2, "denominator": 3}
    assert all_people["coverage_population"] == {"numerator": 2, "denominator": 4}
    assert value(all_people, "placed") == .5
    assert value(all_people, "top_choice_included") == 2 / 3
    # Posting-level ranks: applicant 0's second posting has cost 1/2, not 2/3.
    assert value(all_people, "normalized_rank") == .625
    assert value(all_people, "normalized_rank_eligible") == .5
    assert outcome["degree_0"]["coverage_conditional"] == {"numerator": 0, "denominator": 0}
    assert outcome["degree_0"]["placed"] == {"numerator": 0, "denominator": 1}
    assert value(outcome["degree_0"], "normalized_rank") == 1
    assert outcome["degree_6_10"]["placed"] == {"numerator": 0, "denominator": 0}


def test_transitions_conserve_placements_and_capture_worsening():
    inst, groups, slots = market()
    before, after = np.array([0, 2, -1, -1]), np.array([2, -1, 0, -1])
    result = transition(inst, before, after, groups, slots)
    for group in result.values():
        assert group["gross_entrants"]["numerator"] - group["gross_exits"]["numerator"] == (
            group["after_placed"]["numerator"] - group["baseline_placed"]["numerator"])
        assert sum(group[m]["numerator"] for m in
                   ("gross_exits", "gross_entrants", "same_job", "job_switch", "both_unmatched")) == group["group_count"]["numerator"]
    all_people = result["all"]
    assert value(all_people, "gross_exits") == .25
    assert value(all_people, "exit_rate") == .5
    assert value(all_people, "entry_rate") == .5
    assert value(all_people, "rank_worse") == .5
    assert value(all_people, "worse_among_retained") == 1
    assert value(all_people, "normalized_rank_change") == .125
    assert result["degree_0"]["rank_unchanged"] == {"numerator": 1, "denominator": 1}
    assert result["degree_0"]["exit_rate"] == {"numerator": 0, "denominator": 0}


def test_capacity_slot_switch_is_same_job_and_same_rank():
    inst, groups, slots = market()
    result = transition(inst, np.array([0, 2, -1, -1]), np.array([1, 2, -1, -1]), groups, slots)
    assert result["all"]["same_job"]["numerator"] == 2
    assert result["all"]["job_switch"]["numerator"] == 0
    assert result["all"]["rank_worse"]["numerator"] == 0
    assert result["all"]["normalized_rank_change"]["numerator"] == 0


def test_invalid_matching_is_rejected():
    inst, groups, slots = market()
    with pytest.raises(ValueError, match="same capacity slot"):
        transition(inst, np.array([0, 2, 0, -1]), np.array([0, 2, -1, -1]), groups, slots)


def rows():
    records = []
    # Mean seed rate is .5, whereas pooling all persons would yield .2.
    for seed, n in ((0, 1), (1, 4)):
        for market_id in ("a", "b"):
            for arm in ("U", "A"):
                for metric in ("placed", "coverage_conditional"):
                    records.append(dict(window="frozen", variant="primary", seed=seed,
                                        market=f"{seed}-{market_id}", arm=arm, group="all",
                                        metric=metric, numerator=int(arm == "A" and seed == 0), denominator=n))
    return pd.DataFrame(records)


def test_seed_rate_estimand_and_shared_repeatable_bootstrap():
    data = rows()
    levels = aggregate_levels(data, reps=1000, expected_seeds=2)
    assert set(levels.loc[levels.arm == "A", "mean"]) == {.5}
    first = paired_contrasts(data, [("A", "U")], reps=1000, rng_seed=19, expected_seeds=2)
    second = paired_contrasts(data, [("A", "U")], reps=1000, rng_seed=19, expected_seeds=2)
    pd.testing.assert_frame_equal(first, second)
    assert set(first["mean"]) == {.5}
    assert set(first.seeds_available) == {2}
    assert set(first.paired_markets) == {4}
    assert set(first.family_size) == {2}
    assert (first.simultaneous_lo <= first.lo).all()
    assert (first.simultaneous_hi >= first.hi).all()
    # Identical outcomes have identical intervals because they share draws.
    assert first.lo.nunique() == first.hi.nunique() == 1


def test_empty_denominators_remain_undefined_and_report_support():
    data = rows()
    data.loc[(data.seed == 1) & (data.metric == "coverage_conditional"), ["numerator", "denominator"]] = 0
    result = paired_contrasts(data, [("A", "U")], reps=1000, expected_seeds=2)
    row = result[result.metric == "coverage_conditional"].iloc[0]
    assert np.isnan(row["mean"]) and np.isnan(row.lo) and np.isnan(row.simultaneous_lo)
    assert row.seeds_available == 1 and row.seeds_total == 2
    assert row.available_seed_mean == 1
    assert "undefined_seed_rate" in row.status
    assert 0 < row.valid_bootstrap_draws < 1000


def test_missing_pairs_and_duplicate_records_do_not_pass_silently():
    data = rows()
    with pytest.raises(ValueError, match="Incomplete paired markets"):
        paired_contrasts(data.iloc[1:], [("A", "U")], expected_seeds=2)
    disclosed = paired_contrasts(data.iloc[1:], [("A", "U")], expected_seeds=2, strict_pairs=False)
    affected = disclosed[disclosed.metric == "placed"].iloc[0]
    assert affected.unpaired_markets == 1 and np.isnan(affected["mean"])
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_levels(pd.concat([data, data.iloc[:1]]), expected_seeds=2)


def test_missing_expected_seed_does_not_create_a_complete_panel():
    result = aggregate_levels(rows(), reps=100, expected_seeds=20)
    assert result["mean"].isna().all()
    assert result.lo.isna().all()
    assert result.status.str.contains("incomplete_seed_panel").all()


def test_both_arms_missing_one_market_are_detected_from_panel():
    data = rows()
    data = data[~((data.market == "0-a") & (data.metric == "placed"))]
    with pytest.raises(ValueError, match="Incomplete paired markets"):
        paired_contrasts(data, [("A", "U")], expected_seeds=2)
    levels = aggregate_levels(data, reps=100, expected_seeds=2)
    affected = levels[levels.metric == "placed"]
    assert affected["mean"].isna().all()
    assert set(affected.missing_markets) == {1}
    assert affected.status.str.contains("incomplete_market_panel").all()


def test_max_deviation_interval_uses_shared_centered_draws():
    data = rows().iloc[:0].copy()
    records = []
    for seed in range(20):
        for arm in ("A", "U"):
            for metric in ("placed", "coverage_conditional"):
                effect = seed / 100 if metric == "placed" else (.1 * np.sin(seed))
                records.append(dict(window="frozen", variant="primary", market=str(seed),
                                    seed=seed, arm=arm, group="all", metric=metric,
                                    numerator=.5 + effect * (arm == "A"), denominator=1))
    data = pd.DataFrame(records)
    result = paired_contrasts(data, [("A", "U")], reps=2000, rng_seed=87)
    draws = np.random.default_rng(87).integers(0, 20, size=(2000, 20))
    effects = np.column_stack([.1 * np.sin(np.arange(20)), np.arange(20) / 100])
    expected = np.quantile(np.max(np.abs(effects[draws].mean(axis=1) - effects.mean(axis=0)), axis=1), .95)
    np.testing.assert_allclose(result.simultaneous_critical_value, expected)
    np.testing.assert_allclose(result.simultaneous_lo, result["mean"] - expected)
    np.testing.assert_allclose(result.simultaneous_hi, result["mean"] + expected)
    assert result.simultaneous_family_complete.all()
    assert set(result.family_size) == {2}


def test_undefined_family_endpoint_does_not_reduce_simultaneous_family():
    data = rows()
    data.loc[data.metric == "coverage_conditional", ["numerator", "denominator"]] = 0
    result = paired_contrasts(data, [("A", "U")], reps=1000, expected_seeds=2)
    placed = result[result.metric == "placed"].iloc[0]
    assert np.isfinite(placed["mean"]) and np.isfinite(placed.lo)
    assert result.simultaneous_lo.isna().all()
    assert not result.simultaneous_family_complete.any()
    assert set(result.family_size) == {2}


def test_absent_requested_endpoint_cannot_shrink_the_primary_family():
    data = rows()
    data = data[data.metric == "placed"]
    with pytest.raises(ValueError, match="Missing requested endpoints"):
        paired_contrasts(data, [("A", "U")], expected_seeds=2,
                         metrics=["placed", "coverage_conditional"], groups=["all"])


def test_tie_scenarios_are_not_counted_as_independent_markets_or_seeds():
    original = rows().assign(tie=-1, baseline="")
    scenario = original.assign(tie=0)
    result = aggregate_levels(pd.concat([original, scenario]), reps=100, expected_seeds=2)
    assert set(result.tie) == {-1, 0}
    assert set(result.seeds_total) == {2}
    assert set(result.markets) == {4}
    # Average tie counts (fractional counts allowed) before analysis in the CLI.
    averaged = rows()
    averaged["numerator"] = averaged.numerator / 2
    averaged["denominator"] = averaged.denominator / 2
    collapsed = aggregate_levels(averaged, reps=100, expected_seeds=2)
    assert set(collapsed.loc[collapsed.arm == "A", "mean"]) == {.5}


def test_reference_coverage_collapses_slots_but_assignment_still_uses_menu_edge():
    inst, groups, slots = market()
    shown = inst.E.copy()
    shown[0, 1] = False
    out = evaluate(inst, shown, np.array([0, 2, -1, -1]),
                   np.array([1, 2, -1, -1]), groups, slots)
    assert out["all"]["coverage_conditional"] == {"numerator": 2, "denominator": 2}
    with pytest.raises(ValueError, match="capacity slot absent"):
        evaluate(inst, shown, np.array([1, 2, -1, -1]),
                 np.array([0, 2, -1, -1]), groups, slots)
