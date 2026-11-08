"""Summarize results/<name>/runs.jsonl into CSV tables and a short report.

    python -m experiments.analyze core

Intervals are 95% bootstrap intervals that resample whole seed blocks, as on the slides. Every statistic is computed per seed
block first, because all markets sharing a seed form one block.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GROUP_FIELDS = ["no_feasible_listing", "feasible", "shown", "hidden_count", "hidden_share", "access_after", "hidden_blockers_before", "match_before", "match_after",
                "unmatched_despite_options_before", "unmatched_despite_options_after", "rank_score_before", "rank_score_after", "rank_score_change", "worse_off", "better_off",
                "lost_match", "rent_after", "rent_burden_after"]
GROUPS = ["all", "lower_income", "higher_income", "children", "no_children"]
BY = ["variant", "n", "exposure", "method"]


def load(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base, groups = [], []
    with open(ROOT / "results" / name / "runs.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            plans, mono = r["extra"].get("optimal_plans") or {}, r["extra"].get("nonmonotone") or {}
            base.append({**{k: r[k] for k in ["market_id", "variant", "borough", "month", "n", "seed", "exposure", "policy", "method", "cost", "certified", "lower_bound", "support_cost", "seconds",
                                              "master_solves", "n_cuts", "timed_out", "n_firms", "stable"]},
                         "gap": None if r["lower_bound"] is None else r["cost"] - r["lower_bound"], "n_plans": plans.get("n_plans"), "n_allocations": plans.get("n_allocations"),
                         "plans_complete": plans.get("complete"), "lower_change_low": (plans.get("lower_income_change_range") or [None, None])[0],
                         "lower_change_high": (plans.get("lower_income_change_range") or [None, None])[1], "added_tried": mono.get("tried"), "added_broke": mono.get("broke"),
                         "audited_renter_side": r["outcomes"]["n_firms_audited_renter_side"], "audited_listing_side": r["outcomes"]["n_firms_audited_listing_side"],
                         "refused": r["exposure_stats"].get("refused"), "dropped_outside_E": r["exposure_stats"].get("dropped_outside_E")})
            for g in GROUPS:
                groups.append({"market_id": r["market_id"], "variant": r["variant"], "n": r["n"], "seed": r["seed"], "exposure": r["exposure"], "method": r["method"], "group": g, **{f: r["outcomes"][g][f] for f in GROUP_FIELDS}})
    return pd.DataFrame(base), pd.DataFrame(groups)


def bootstrap(df: pd.DataFrame, value: str, reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    per_seed = df.groupby("seed")[value].mean().dropna().to_numpy()
    if len(per_seed) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = per_seed[rng.integers(0, len(per_seed), (reps, len(per_seed)))].mean(1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def with_ci(df: pd.DataFrame, by: list[str], values: list[str]) -> pd.DataFrame:
    rows = []
    for key, part in df.groupby(by, dropna=False):
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        for v in values:
            if part[v].notna().any():
                low, high = bootstrap(part, v)
                row.update({v: part[v].mean(), f"{v}_lo": low, f"{v}_hi": high})
        rows.append(row)
    return pd.DataFrame(rows)


def md(df: pd.DataFrame, digits: int = 2) -> str:
    d = df.copy()
    for c in d.select_dtypes("number"):
        d[c] = d[c].round(digits)
    rows = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    rows += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("--plan-method", default="exact_prefix", help="method whose plans the group tables describe")
    args = ap.parse_args()
    out = ROOT / "results" / args.name
    base, groups = load(args.name)
    if not base.stable.where(base.method != "no_audit", True).all():
        raise SystemExit("a returned plan failed the full-market check: do not use these results")

    methods = base.groupby(BY).agg(markets=("cost", "size"), cost=("cost", "mean"), certified=("certified", "mean"), gap_max=("gap", "max"), seconds_median=("seconds", "median"),
                                   seconds_max=("seconds", "max"), master_solves=("master_solves", "mean"), n_firms=("n_firms", "mean")).reset_index()
    exact = base[base.method == args.plan_method]
    if len(exact):
        exact = exact.assign(gap_support=exact.cost - exact.support_cost, at_support=exact.cost == exact.support_cost)
        at_min = base.merge(exact[["market_id", "exposure", "cost", "certified"]].rename(columns={"cost": "best", "certified": "best_certified"}), on=["market_id", "exposure"])
        at_min = at_min[at_min.best_certified]
        methods = methods.merge(at_min.assign(at_min=at_min.cost == at_min.best).groupby(BY).at_min.mean().reset_index(), on=BY, how="left")
    methods.to_csv(out / "methods.csv", index=False)

    table = with_ci(groups, ["variant", "n", "exposure", "method", "group"], GROUP_FIELDS)
    table.to_csv(out / "groups.csv", index=False)

    lines = [f"# Results: {args.name}\n", f"{base.market_id.nunique()} markets. Every returned plan passed the full-market check.\n", "## Audit methods\n", md(methods), ""]
    if len(exact):
        theory = exact.groupby(["variant", "n", "exposure"]).agg(markets=("cost", "size"), certified=("certified", "mean"), cda=("cost", "mean"), csupport=("support_cost", "mean"), cda_equals_support=("at_support", "mean"),
                                                                 plans=("n_plans", "mean"), allocations=("n_allocations", "mean"), added=("added_tried", "sum"), added_broke=("added_broke", "sum"),
                                                                 lower_change_low=("lower_change_low", "mean"), lower_change_high=("lower_change_high", "mean"),
                                                                 refused=("refused", "sum"), dropped_outside_E=("dropped_outside_E", "sum")).reset_index().dropna(axis=1, how="all")
        theory.to_csv(out / "theory.csv", index=False)
        lines += ["## Theory checks on the exact plans\n", md(theory), ""]
        focus = table[(table.method == args.plan_method) & table.group.isin(["lower_income", "higher_income"])]
        keep = ["variant", "n", "exposure", "group", "no_feasible_listing", "feasible", "shown", "hidden_share", "hidden_blockers_before", "match_before", "match_after", "rank_score_before", "rank_score_after", "worse_off", "better_off"]
        lines += ["## Lower vs. higher income, after the exact plan\n", md(focus[keep]), "", "Rank score change with 95% seed-block intervals:\n",
                  md(focus[["variant", "n", "exposure", "group", "rank_score_change", "rank_score_change_lo", "rank_score_change_hi", "worse_off_lo", "worse_off_hi"]], 3), ""]
    (out / "summary.md").write_text("\n".join(lines))
    print(f"wrote {out / 'summary.md'}, methods.csv, groups.csv")


if __name__ == "__main__":
    main()
