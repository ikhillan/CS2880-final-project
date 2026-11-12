"""Summarize results/<name>/ into CSV tables and a short report (Todd's protocol, sections 2, 6, 8 and 9).

    python -m experiments.analyze stage1_pilot --treated claude_k5 --control uniform_k5

Reads runs.jsonl / markets.jsonl and any shards/runs-*.jsonl / shards/markets-*.jsonl.

Standardized group rates (protocol section 6): within each borough-month cell, a group's rate is the sum of its renters' Y_b over
all draws divided by the number of its renters drawn there. Cells are weighted with months equal and boroughs at their ACS
recent-mover household shares. Cells with no observations (e.g. Staten Island) make the full-geography target unestimable;
the report gives the covered weight and labels the renormalized estimate as a different target.

Intervals resample whole seed blocks (all policies and cells of a block stay together) and recompute every ratio and weight.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hiddenoptions.data import borough_weights, load_renters

ROOT = Path(__file__).resolve().parents[1]
GROUP_FIELDS = ["no_feasible_listing", "feasible", "shown", "hidden_share", "blocker_rate", "can_hide_share", "blocker_rate_given_can_hide",
                "price_percentile_shown", "match_before", "match_after", "match_full_disclosure", "top3_before", "prefer_full_to_initial",
                "rank_score_before", "rank_score_after", "rank_score_change", "worse_off", "better_off", "lost_match",
                "unavoidable_loss", "severity_positive", "severity_mean", "rent_burden_after",
                "scarcity_no_option_in_cohort", "scarcity_none_in_draw", "scarcity_unmatched_full", "scarcity_matched_full"]


def _read_jsonl(paths):
    rows = []
    for p in paths:
        with open(p) as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    return rows


def load(name: str):
    base = ROOT / "results" / name
    runs = _read_jsonl([base / "runs.jsonl"] * (base / "runs.jsonl").exists() + sorted(glob.glob(str(base / "shards" / "runs-*.jsonl"))))
    markets = _read_jsonl([base / "markets.jsonl"] * (base / "markets.jsonl").exists() + sorted(glob.glob(str(base / "shards" / "markets-*.jsonl"))))
    if not runs:
        raise SystemExit(f"no runs found under {base}")
    return runs, markets


def cell_weights(cells: list[tuple[str, str]]) -> dict:
    """Months equal, boroughs at ACS recent-mover shares, over the requested cells."""
    shares = borough_weights(load_renters())
    months = sorted({m for _, m in cells})
    return {(b, m): shares.get(b, 0.0) / len(months) for b, m in cells}


def standardized(counts: pd.DataFrame, weights: dict) -> tuple[float, float]:
    """Sum over cells of w_c x (sum of counts / sum of n), and the weight actually covered."""
    g = counts.groupby(["borough", "month"])[["count", "n"]].sum()
    g = g[g.n > 0]
    covered = sum(weights.get(c, 0.0) for c in g.index)
    if covered == 0:
        return float("nan"), 0.0
    rate = sum(weights.get(c, 0.0) * row["count"] / row["n"] for c, row in g.iterrows()) / covered
    return float(rate), float(covered / sum(weights.values()))


def group_counts(no_audit: pd.DataFrame, exposure: str, group: str, conditional: bool = False) -> pd.DataFrame:
    d = no_audit[no_audit.exposure == exposure]
    count = d[f"{group}_blocker_count"] if not conditional else d[f"{group}_blocker_rate_given_can_hide"].fillna(0) * d[f"{group}_can_hide_count"]
    n = d[f"{group}_n"] if not conditional else d[f"{group}_can_hide_count"]
    return pd.DataFrame({"borough": d.borough, "month": d.month, "seed": d.seed, "count": count, "n": n})


def contrast_T(no_audit: pd.DataFrame, treated: str, control: str, weights: dict, conditional: bool = False) -> dict:
    rates = {}
    coverage = 1.0
    for p in (treated, control):
        for g in ("lower_income", "higher_income"):
            rates[(p, g)], cov = standardized(group_counts(no_audit, p, g, conditional), weights)
            coverage = min(coverage, cov)
    T = (rates[(treated, "lower_income")] - rates[(control, "lower_income")]) - (rates[(treated, "higher_income")] - rates[(control, "higher_income")])
    return {"T": T, "coverage": coverage, **{f"B_{g}_{p}": v for (p, g), v in rates.items()}}


def bootstrap(fn, frame: pd.DataFrame, reps: int, seed: int = 0) -> tuple[float, float]:
    seeds = frame.seed.unique()
    if len(seeds) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    by_seed = {s: g for s, g in frame.groupby("seed")}
    stats = []
    for _ in range(reps):
        draw = rng.choice(seeds, size=len(seeds), replace=True)
        sample = pd.concat([by_seed[s].assign(seed=i) for i, s in enumerate(draw)])
        v = fn(sample)
        if np.isfinite(v):
            stats.append(v)
    return (float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))) if stats else (float("nan"), float("nan"))


def flatten(runs) -> tuple[pd.DataFrame, pd.DataFrame]:
    base, groups = [], []
    for r in runs:
        row = {k: r.get(k) for k in ["market_id", "variant", "borough", "month", "n", "seed", "design", "exposure", "policy", "method", "cost",
                                     "certified", "lower_bound", "support_cost", "seconds", "master_solves", "timed_out", "n_firms", "stable"]}
        row["cost_scale"] = r.get("cost_scale", 1)
        row["protected_renters"] = r["extra"].get("protected_renters")
        row["cost_chain_ok"] = r["extra"].get("cost_chain_ok")
        stats = r.get("exposure_stats") or {}
        for k in ("complete_valid", "partial_fill", "full_fallback", "refused", "unparseable", "missing", "whole_set"):
            row[f"llm_{k}"] = stats.get(k)
        for g in ("lower_income", "higher_income"):
            o = r["outcomes"][g]
            for f in ("n", "blocker_count", "can_hide_count", "blocker_rate_given_can_hide"):
                row[f"{g}_{f}"] = o.get(f)
        base.append(row)
        for g in ("all", "lower_income", "higher_income", "children", "no_children"):
            groups.append({"market_id": r["market_id"], "variant": r["variant"], "n": r["n"], "seed": r["seed"], "exposure": r["exposure"],
                           "method": r["method"], "group": g, **{f: r["outcomes"][g].get(f) for f in GROUP_FIELDS}})
    return pd.DataFrame(base), pd.DataFrame(groups)


def md(df: pd.DataFrame, digits: int = 3) -> str:
    if df.empty:
        return "(none)"
    d = df.copy()
    for c in d.select_dtypes("number"):
        d[c] = d[c].round(digits)
    rows = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    rows += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(rows)


def protection_table(base: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Premium Delta_G = C_G - C_DA per market and policy, with solver-identification bounds (protocol section 8)."""
    out = []
    for (variant, n, exposure), d in base.groupby(["variant", "n", "exposure"]):
        piv = d.set_index(["market_id", "method"])
        for group in ("lower", "all"):
            m_da, m_g = "exact_prefix", f"exact_prefix_protect_{group}"
            ids = sorted(set(d.market_id[d.method == m_da]) & set(d.market_id[d.method == m_g]))
            if not ids:
                continue
            rows = []
            for mid in ids:
                da, pg = piv.loc[(mid, m_da)], piv.loc[(mid, m_g)]
                scale = da.cost_scale or 1
                lo = max(0.0, (pg.lower_bound - da.cost) / scale) if pg.lower_bound is not None else 0.0
                hi = (pg.cost - (da.lower_bound if da.lower_bound is not None else 0)) / scale
                rows.append({"borough": da.borough, "month": da.month, "seed": da.seed, "lo": lo, "hi": hi,
                             "exact": (pg.cost - da.cost) / scale if (pg.certified and da.certified) else np.nan,
                             "certified": bool(pg.certified and da.certified), "empty_group": (pg.protected_renters or 0) == 0})
            r = pd.DataFrame(rows)
            out.append({"variant": variant, "n": n, "exposure": exposure, "objective": f"protect_{group}", "markets": len(r),
                        "certified_share": r.certified.mean(), "empty_group_share": r.empty_group.mean(),
                        "positive_share_low": (r.lo > 0).mean(), "positive_share_high": (r.hi > 0).mean(),
                        "mean_premium_low": r.lo.mean(), "mean_premium_high": r.hi.mean(), "mean_premium_certified": r.exact.mean()})
    return pd.DataFrame(out)


def switch_table(markets: list, weights: dict, reps: int) -> pd.DataFrame:
    rows = []
    for m in markets:
        for name, sw in (m.get("group_switch") or {}).items():
            for combo in ("AA", "BA", "AB", "BB"):
                for g, gn in (("lower", "lower_income"), ("higher", "higher_income")):
                    rows.append({"name": name, "variant": m.get("variant"), "borough": m["borough"], "month": m["month"], "seed": m["seed"],
                                 "combo": combo, "group": gn, "count": sw[combo][f"{g}_count"], "n": sw[combo][f"{g}_n"]})
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)

    def Z(frame, combo):
        lo, _ = standardized(frame[(frame.combo == combo) & (frame.group == "lower_income")], weights)
        hi, _ = standardized(frame[(frame.combo == combo) & (frame.group == "higher_income")], weights)
        return lo - hi

    def decompose(frame):
        z = {c: Z(frame, c) for c in ("AA", "BA", "AB", "BB")}
        s_l = ((z["BA"] - z["AA"]) + (z["BB"] - z["AB"])) / 2
        s_h = ((z["AB"] - z["AA"]) + (z["BB"] - z["BA"])) / 2
        return z, s_l, s_h

    out = []
    for (name, variant), frame in d.groupby(["name", "variant"]):
        z, s_l, s_h = decompose(frame)
        t = z["BB"] - z["AA"]
        lo_l, hi_l = bootstrap(lambda f: decompose(f)[1], frame, reps)
        lo_h, hi_h = bootstrap(lambda f: decompose(f)[2], frame, reps)
        out.append({"switch": name, "variant": variant, "Z_UU": z["AA"], "Z_CU": z["BA"], "Z_UC": z["AB"], "Z_CC": z["BB"],
                    "S_L": s_l, "S_L_lo": lo_l, "S_L_hi": hi_l, "S_H": s_h, "S_H_lo": lo_h, "S_H_hi": hi_h, "T": t,
                    "S_L_plus_S_H_equals_T": bool(abs(s_l + s_h - t) < 1e-12)})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("--treated", default="claude_k5", help="exposure for the primary contrast T")
    ap.add_argument("--control", default="uniform_k5")
    ap.add_argument("--plan-method", default="exact_prefix", help="method whose plans the group tables describe")
    ap.add_argument("--reps", type=int, default=2000, help="bootstrap replicates (the protocol's final analysis uses 10,000)")
    args = ap.parse_args()
    out = ROOT / "results" / args.name
    runs, markets = load(args.name)
    base, groups = flatten(runs)
    bad = base[(base.method != "no_audit") & (base.stable == False)]  # noqa: E712
    if len(bad):
        raise SystemExit(f"{len(bad)} returned plans failed the full-market or protection check: do not use these results")
    statuses = pd.Series([m.get("status", "ok") for m in markets]).value_counts().to_dict()
    lines = [f"# Results: {args.name}\n", f"{base.market_id.nunique()} markets with results; market status counts: {statuses}. "
             "Every returned plan passed its full-market (and protection) check.\n"]

    methods = base.groupby(["variant", "n", "exposure", "method"]).agg(
        markets=("cost", "size"), cost=("cost", "mean"), certified=("certified", "mean"), seconds_median=("seconds", "median"),
        seconds_max=("seconds", "max"), n_firms=("n_firms", "mean")).reset_index()
    methods.to_csv(out / "methods.csv", index=False)
    lines += ["## Audit methods\n", md(methods), ""]

    cells = sorted({(b, m) for b, m in zip(base.borough, base.month) if b})
    weights = cell_weights(cells) if cells else {}
    no_audit = base[base.method == "no_audit"]
    policies = set(no_audit.exposure)
    if {args.treated, args.control} <= policies and weights:
        lines.append(f"## Primary contrast T: {args.treated} vs {args.control}\n")
        rows = []
        for variant, d in no_audit.groupby("variant"):
            for conditional in (False, True):
                res = contrast_T(d, args.treated, args.control, weights, conditional)
                lo, hi = bootstrap(lambda f: contrast_T(f, args.treated, args.control, weights, conditional)["T"], d, args.reps)
                rows.append({"variant": variant, "renters": "can_hide only" if conditional else "all", "T_pp": 100 * res["T"],
                             "T_lo_pp": 100 * lo, "T_hi_pp": 100 * hi, "coverage": res["coverage"],
                             **{k: v for k, v in res.items() if k.startswith("B_")}})
            for month, dm in d.groupby("month"):
                res = contrast_T(dm, args.treated, args.control, {c: w for c, w in weights.items() if c[1] == month})
                rows.append({"variant": variant, "renters": f"all, {month}", "T_pp": 100 * res["T"], "coverage": res["coverage"]})
        t_table = pd.DataFrame(rows)
        t_table.to_csv(out / "contrast_T.csv", index=False)
        lines += [md(t_table), "", "Coverage below 1 means some cells (e.g. Staten Island) had no observations, so the estimate is "
                  "renormalized over the covered cells and is a different target from the full-geography contrast.\n"]
    else:
        lines.append(f"## Primary contrast T\n\nNot computed: needs no_audit rows for both {args.treated} and {args.control}.\n")

    sw = switch_table(markets, weights, args.reps)
    if not sw.empty:
        sw.to_csv(out / "group_switch.csv", index=False)
        lines += ["## Group-switch decomposition (A = control, B = treated)\n", md(sw), ""]

    prot = protection_table(base, weights)
    if not prot.empty:
        prot.to_csv(out / "protection.csv", index=False)
        lines += ["## Protection premium (bounds hold even for interrupted solves)\n", md(prot), ""]

    g = groups.groupby(["variant", "n", "exposure", "method", "group"])[GROUP_FIELDS].mean().reset_index()
    g.to_csv(out / "groups.csv", index=False)
    focus = g[(g.method.isin(["no_audit", args.plan_method])) & g.group.isin(["lower_income", "higher_income"])]
    keep = ["variant", "n", "exposure", "method", "group", "no_feasible_listing", "blocker_rate", "can_hide_share", "price_percentile_shown",
            "match_before", "match_after", "worse_off", "unavoidable_loss", "severity_positive", "severity_mean"]
    lines += ["## Lower vs. higher income\n", md(focus[keep]), ""]
    scar = g[(g.method == "no_audit") & g.group.isin(["lower_income", "higher_income"])][
        ["variant", "n", "exposure", "group", "scarcity_no_option_in_cohort", "scarcity_none_in_draw", "scarcity_unmatched_full", "scarcity_matched_full"]]
    lines += ["## Scarcity categories (partition renters)\n", md(scar.drop_duplicates(["variant", "n", "group"])), ""]
    (out / "summary.md").write_text("\n".join(lines))
    print(f"wrote {out / 'summary.md'} and CSV tables")


if __name__ == "__main__":
    main()
