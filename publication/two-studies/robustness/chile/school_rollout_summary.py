"""Summarize prospective policy adoption Monte Carlo replicates within each archived cohort."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(rows, own):
    fractions = [0, 0.25, 0.5, 0.75, 1]
    if rows.duplicated(["pair", "half", "uptake"]).any():
        raise ValueError("Duplicate rollout records")
    expected = pd.MultiIndex.from_product([rows.pair.unique(), [0, 1], fractions], names=["pair", "half", "uptake"])
    if not expected.difference(rows.set_index(expected.names).index).empty:
        raise ValueError("Missing rollout records")
    for col in ("year", "n", "n_same"):
        if rows[col].nunique() != 1:
            raise ValueError("Inconsistent cohort description")
    year, n, n_same = (int(rows[col].iloc[0]) for col in ("year", "n", "n_same"))
    metrics = ["placed", "full_covered", "recorded_covered", "named",
               "gains", "losses", "fixed_gains", "fixed_losses"]
    ends = rows[rows.uptake.isin([0, 1])].groupby("uptake")[metrics]
    if (ends.nunique().to_numpy() != 1).any():
        raise ValueError("Endpoint assignments changed between replications")
    start, finish = (ends.first().loc[p] for p in (0, 1))
    pair_means = rows.groupby(["uptake", "pair"])[metrics].mean()
    records = []
    for p in fractions:
        whole = pair_means.loc[p].copy()
        for key in ("placed", "full_covered", "recorded_covered"):
            whole[f"{key}_delta"] = whole[key] - start[key]
            linear = (1 - p) * start[key] + p * finish[key]
            whole[f"{key}_minus_linear"] = whole[key] - linear
        whole["fixed_net_delta"] = whole.fixed_gains - whole.fixed_losses
        # At p=.5 each row uses B in exactly one half. Its coverage sums to both endpoint indicators.
        if p == 0.5:
            assert (whole.full_covered_minus_linear == 0).all()
            assert (whole.recorded_covered_minus_linear == 0).all()
        for metric, series in whole.items():
            estimate, mcse = series.mean(), series.sem()
            records.append(dict(year=year, uptake=p, metric=metric, applicants=n,
                                fixed_rows=n_same, pairs=len(series), count=estimate,
                                mcse_count=mcse, pp=100 * estimate / n, mcse_pp=100 * mcse / n))
    if len(own) != rows.pair.nunique() or own.pair.nunique() != len(own):
        raise ValueError("Incomplete midpoint direct effects")
    if not (own.year.eq(year).all() and own.n.eq(n).all() and own.n_changed.eq(n-n_same).all()):
        raise ValueError("Own effect denominators differ")
    series = own.direct_delta_placed
    records.append(dict(year=year, uptake=0.5, metric="midpoint_direct", applicants=n,
                        fixed_rows=n_same, pairs=len(series), count=series.mean(),
                        mcse_count=series.sem(), pp=100 * series.mean()/n, mcse_pp=100 * series.sem()/n))
    return pd.DataFrame(records)


def lottery_table(rows, budget):
    """Compare paired policies within each new strict priority realization."""
    rows = rows.copy()
    for outcome in ("placed", "full_covered", "named"):
        rows[outcome] = rows[f"rank_{outcome}"] - rows[f"blind_{outcome}"]
    rows["tradeoff"] = (rows.full_covered > 0) & (rows.placed < 0)
    output = []
    for (year, model), group in rows.groupby(["year", "lottery"], sort=True):
        observed = model == "observed"
        assert group.draw.nunique() == len(group)
        if observed:
            assert len(group) == 1
            endpoint = budget[(budget.year == year) & (budget.k == 5)].iloc[0]
            for arm in ("rank", "blind"):
                for outcome in ("placed", "full_covered", "named"):
                    col = f"{arm}_{outcome}"
                    assert endpoint[col] == group[col].iloc[0]
        for outcome in ("placed", "full_covered", "named"):
            v = group[outcome]
            output.append(dict(year=year, lottery=model, outcome=outcome, draws=len(v),
                               mean_delta=v.mean(), mcse=0 if observed else v.sem(),
                               min_delta=v.min(), max_delta=v.max(),
                               joint_sign_draws=int(group.tradeoff.sum())))
    return pd.DataFrame(output)


def main(inputs, output, evidence):
    summaries, budgets = [], []
    for year in (2022, 2023):
        prefix = inputs / f"school-{year}"
        records = pd.read_csv(str(prefix)+"-rollout.csv")
        own = pd.read_csv(str(prefix)+"-own.csv")
        quota = pd.read_csv(str(prefix)+"-quota.csv")
        roll = summarize(records, own)
        orig = pd.read_csv(evidence / "school-levels.csv")
        known = orig[(orig.year == year) & (orig.reference == "full_DA") & (orig.group == "all")].iloc[0]
        official = orig[(orig.year == year) & (orig.reference == "recorded_offer") & (orig.group == "all")].iloc[0]
        for _, q in quota.iterrows():
            assert q.gains - q.losses == q.rank_placed - q.blind_placed
            assert 0 <= q.fixed_losses <= q.losses and 0 <= q.fixed_gains <= q.gains
            assert q.blind_named <= q.blind_full_covered and q.rank_named <= q.rank_full_covered
            if q.k == 5:
                assert q.blind_placed == known.blind_placed and q.rank_placed == known.rank_placed
                assert q.blind_full_covered == known.blind_shown and q.rank_full_covered == known.rank_shown
                assert q.blind_recorded_covered == official.blind_shown and q.rank_recorded_covered == official.rank_shown
        budgets.append(quota)
        summaries.append(roll)
    pd.concat(summaries, ignore_index=True).to_csv(output / "school-rollout-summary.csv", index=False)
    budgets = pd.concat(budgets, ignore_index=True)
    budgets.to_csv(output / "school-budget-check.csv", index=False)
    lottery = pd.concat([pd.read_csv(inputs / f"school-{year}-lottery.csv") for year in (2022, 2023)])
    lottery_table(lottery, budgets).to_csv(output / "school-lottery-summary.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Directory of per-year aggregate rollout counts")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-evidence", type=Path, required=True, help="Two-study evidence directory")
    args = parser.parse_args()
    main(args.inputs, args.output, args.baseline_evidence)
