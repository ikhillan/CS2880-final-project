"""Summarize results/<name>/ into CSV tables and a short report (Todd's protocol, sections 2, 6, 8 and 9).

    python -m experiments.analyze stage1_pilot --treated claude_k5 --control uniform_k5

Reads runs.jsonl / markets.jsonl and any shards/runs-*.jsonl / shards/markets-*.jsonl.

Standardized group rates (protocol section 6): within each borough-month cell, a group's rate is the sum of its renters' Y_b over
all draws divided by the number of its renters drawn there. Cells are weighted with months equal and boroughs at their ACS
recent-mover household shares. Cells with no observations (e.g. Staten Island) make the full-geography target unestimable;
the report gives the covered weight and labels the renormalized estimate as a different target.

Both income groups and policies use the same supported cells and paired markets. Bootstrap intervals retain the observed
cell support and resample whole seed blocks (all policies and cells of a block stay together), recomputing every ratio.
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
    # run_market appends one summary on every resume, including when only new AI arms ran.
    unique = {}
    for record in markets:
        mid = record["market_id"]
        switch = dict(unique.get(mid, {}).get("group_switch", {}))
        for name, values in (record.get("group_switch") or {}).items():
            if name in switch and switch[name] != values:
                raise ValueError(f"Conflicting group-switch outcomes for {mid}: {name}")
            switch[name] = values
        unique[mid] = {**record, "group_switch": switch}
    return runs, list(unique.values())


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
    return estimate_t(no_audit, treated, control, weights, conditional, 0)


def paired_markets(frame, treated, control):
    """Keep exactly one observation of each policy in each retained market."""
    if treated == control:
        raise ValueError("Treated and control arms must differ.")
    relevant = frame[frame.exposure.isin([treated, control])]
    if relevant.duplicated(["market_id", "exposure"]).any():
        raise ValueError("Duplicate no-audit market/policy rows; reconcile overlapping runs or shards before analysis.")
    arms = relevant.groupby("market_id").exposure.size()
    paired = arms.index[arms == 2]
    return relevant[relevant.market_id.isin(paired)], len(paired), len(arms) - len(paired)


def joint_rates(counts, weights, arms, reps):
    """Use one common cell target and one seed-block draw for all policies and both groups.

    The arrays have axes (arm, group), or (bootstrap draw, arm, group).
    If a bootstrap sample loses an observed target cell, that draw is undefined.
    """
    groups = ["lower_income", "higher_income"]
    seeds = sorted(counts.seed.unique())
    cells = [c for c, w in weights.items() if w > 0]
    shape = (len(seeds), len(cells), len(arms), len(groups), 2)
    cube = np.zeros(shape, dtype=float)
    cell_index = {c: i for i, c in enumerate(cells)}
    si = {seed: i for i, seed in enumerate(seeds)}
    for (seed, b, m, a, g), row in counts.groupby(["seed", "borough", "month", "arm", "group"])[["count", "n"]].sum().iterrows():
        if (b, m) in cell_index:
            cube[si[seed], cell_index[(b, m)], arms.index(a), groups.index(g)] = row.to_numpy()
    total = cube.sum(axis=0)
    supported = (total[..., 1] > 0).all(axis=(1, 2))
    w = np.array([weights[c] for c in cells], dtype=float)[supported]
    resamples = np.full((reps, len(arms), len(groups)), np.nan)
    if not len(w):
        return np.full((len(arms), len(groups)), np.nan), resamples, 0., 0
    coverage = w.sum() / sum(weights.values())
    w /= w.sum()
    total = total[supported]
    point = np.einsum("c,cag->ag", w, total[..., 0] / total[..., 1])
    if len(seeds) < 2 or not reps:
        return point, resamples, coverage, len(w)

    flat = cube[:, supported].reshape(len(seeds), -1)
    rng = np.random.default_rng(0)
    # Block frequencies implement paired resampling without duplicating DataFrames.
    for start in range(0, reps, 128):
        size = min(128, reps - start)
        sample_ids = rng.choice(len(seeds), size=(size, len(seeds)), replace=True)
        freq = np.zeros((size, len(seeds)), dtype=int)
        np.add.at(freq, (np.arange(size)[:, None], sample_ids), 1)
        simulated = (freq @ flat).reshape(size, len(w), len(arms), len(groups), 2)
        with np.errstate(invalid="ignore"):
            ratio = simulated[..., 0] / simulated[..., 1]
        resamples[start:start + size] = np.einsum("c,icag->iag", w, ratio)
    return point, resamples, coverage, len(w)


def interval(draws):
    """An interval for a fixed target is unavailable if a draw loses its denominator."""
    if not len(draws) or not np.isfinite(draws).all():
        return np.nan, np.nan
    return tuple(np.quantile(draws, [.025, .975]))


def estimate_t(frame, treated, control, weights, conditional, reps):
    pairs, paired, unpaired = paired_markets(frame, treated, control)
    arms = [treated, control]
    groups = ["lower_income", "higher_income"]
    counts = pd.concat([group_counts(pairs, a, g, conditional).assign(arm=a, group=g)
                        for a in arms for g in groups])
    rates, draws, coverage, cells = joint_rates(counts, weights, arms, reps)
    by_group = rates[0] - rates[1]
    changes = draws[:, 0] - draws[:, 1]
    sample_t = changes[:, 0] - changes[:, 1]
    low, high = interval(sample_t)
    return {"T": by_group[0] - by_group[1], "T_lo": low, "T_hi": high, "coverage": coverage,
            "supported_cells": cells, "paired_markets": paired, "unpaired_markets": unpaired,
            "valid_bootstrap_draws": int(np.isfinite(sample_t).sum()),
            **{f"B_{g}_{a}": rates[i, j] for i, a in enumerate(arms) for j, g in enumerate(groups)}}


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
                hi = (pg.cost - (da.lower_bound if pd.notna(da.lower_bound) else 0)) / scale
                rows.append({"borough": da.borough, "month": da.month, "seed": da.seed, "lo": lo, "hi": hi,
                             "exact": (pg.cost - da.cost) / scale if (pg.certified and da.certified) else np.nan,
                             "certified": bool(pg.certified and da.certified), "empty_group": (pg.protected_renters or 0) == 0})
            r = pd.DataFrame(rows)
            r = r.assign(certified_share=r.certified.astype(float), empty_group_share=r.empty_group.astype(float),
                         positive_share_low=(r.lo > 0).astype(float), positive_share_high=(r.hi > 0).astype(float),
                         mean_premium_low=r.lo, mean_premium_high=r.hi, mean_premium_certified=r.exact)
            target = weights.get((variant, n), {})
            if not target:
                continue
            entry = {"variant": variant, "n": n, "exposure": exposure, "objective": f"protect_{group}", "markets": len(r)}
            for key in ["certified_share", "empty_group_share", "positive_share_low", "positive_share_high",
                        "mean_premium_low", "mean_premium_high", "mean_premium_certified"]:
                entry["unweighted_" + key] = r[key].mean()
                rate, cov = standardized(r.assign(count=r[key], n=r[key].notna().astype(int)), target)
                entry[key] = rate
                if key == "mean_premium_certified":
                    entry["certified_cell_coverage"] = cov
                else:
                    entry["coverage"] = cov
            out.append(entry)
    return pd.DataFrame(out)


def switch_table(markets: list, weights: dict, reps: int) -> pd.DataFrame:
    rows = []
    arms = ["AA", "BA", "AB", "BB"]
    for m in markets:
        for name, sw in (m.get("group_switch") or {}).items():
            for a in arms:
                for g in ("lower", "higher"):
                    rows.append({"name": name, "variant": m["variant"], "size": m["n"], "borough": m["borough"],
                                 "month": m["month"], "seed": m["seed"], "arm": a, "group": g + "_income",
                                 "count": sw[a][f"{g}_count"], "n": sw[a][f"{g}_n"]})
    if not rows:
        return pd.DataFrame()
    out = []
    for (name, variant, size), frame in pd.DataFrame(rows).groupby(["name", "variant", "size"]):
        rates, draws, coverage, cells = joint_rates(frame, weights.get((variant, size), {}), arms, reps)
        z = rates[:, 0] - rates[:, 1]
        dz = draws[:, :, 0] - draws[:, :, 1]

        def components(gaps):
            aa, ba, ab, bb = np.moveaxis(gaps, -1, 0)
            return ((ba - aa) + (bb - ab)) / 2, ((ab - aa) + (bb - ba)) / 2

        s_l, s_h = components(z)
        d_l, d_h = components(dz)
        lo_l, hi_l = interval(d_l)
        lo_h, hi_h = interval(d_h)
        t = z[3] - z[0]
        labels = ["UU", "CU", "UC", "CC"]
        out.append({"switch": name, "variant": variant, "n": size, "coverage": coverage, "supported_cells": cells,
                    "valid_bootstrap_draws": int(np.isfinite(d_l).sum()),
                    **{f"Z_{label}": z[i] for i, label in enumerate(labels)},
                    **{f"B_{group}_{label}": rates[i, g] for i, label in enumerate(labels)
                       for g, group in enumerate(("lower", "higher"))},
                    "S_L": s_l, "S_L_lo": lo_l, "S_L_hi": hi_l, "S_H": s_h, "S_H_lo": lo_h, "S_H_hi": hi_h,
                    "T": t, "S_L_plus_S_H_equals_T": bool(abs(s_l + s_h - t) < 1e-12)})
    return pd.DataFrame(out)


def planned_weights(markets, base, specs=None):
    """Track coverage by planned variant and size, including unestimable cells."""
    if specs is not None:
        rows = [(variant, s.n_renters, s.borough, s.month) for s, variant in specs]
    else:
        rows = [(m["variant"], m["n"], m["borough"], m["month"]) for m in markets if m.get("borough")]
        rows.extend(base[["variant", "n", "borough", "month"]].itertuples(index=False, name=None))
    cells = {}
    for variant, size, borough, month in rows:
        if borough:
            cells.setdefault((variant, size), set()).add((borough, month))
    return {panel: cell_weights(sorted(grid)) for panel, grid in cells.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("--config", type=Path, help="planned housing grid if the config filename differs from the result name")
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

    # Weights cover every planned cell, including unestimable ones (e.g. Staten Island), so coverage reports their absence.
    config = args.config or ROOT / "configs" / (args.name + ".json")
    specs = None
    if args.config or config.is_file():
        from experiments.config import expand_labeled, load_config
        design = load_config(config)
        if "boroughs" in design.get("grid", {}):
            specs = expand_labeled(design)
    weights = planned_weights(markets, base, specs)
    if specs is None:
        lines.append("Coverage uses the cells in the logged market and run records. A missing whole cell "
                     "cannot be identified without the planned housing config.\n")
    no_audit = base[base.method == "no_audit"]
    policies = set(no_audit.exposure)
    if {args.treated, args.control} <= policies and weights:
        lines.append(f"## Primary contrast T: {args.treated} vs {args.control}\n")
        lines.append("The primary endpoint includes all renters. The conditional endpoint restricts to renters whose shortlist can omit an option.\n")
        rows = []
        for (variant, size), d in no_audit.groupby(["variant", "n"]):
            if (variant, size) not in weights:
                continue
            target = weights[(variant, size)]
            def add_result(frame, w, label, conditional=False, reps=0):
                result = estimate_t(frame, args.treated, args.control, w, conditional, reps)
                for metric in ("T", "T_lo", "T_hi"):
                    result[metric + "_pp"] = 100 * result.pop(metric)
                rows.append({"variant": variant, "n": size, "renters": label, **result})

            add_result(d, target, "all", reps=args.reps)
            add_result(d, target, "can_hide only", conditional=True, reps=args.reps)
            for month, dm in d.groupby("month"):
                add_result(dm, {c: w for c, w in target.items() if c[1] == month}, f"all, {month}", reps=args.reps)
        t_table = pd.DataFrame(rows)
        t_table.to_csv(out / "contrast_T.csv", index=False)
        lines += [md(t_table), "", "Coverage below 1 means some cells (e.g. Staten Island) had no observations, so the estimate is "
                  "renormalized over cells with positive denominators in both groups and both arms. Unpaired markets are excluded; including a cell in the analysis requires paired observations. Intervals use fixed cell support. A blank interval with invalid bootstrap draws indicates a lost denominator.\n"]
    else:
        lines.append(f"## Primary contrast T\n\nNot computed: needs no_audit rows for both {args.treated} and {args.control}.\n")

    sw = switch_table(markets, weights, args.reps)
    if not sw.empty:
        sw.to_csv(out / "group_switch.csv", index=False)
        lines += ["## Group-switch decomposition (A = control, B = treated)\n", md(sw), ""]

    prot = protection_table(base, weights)
    if not prot.empty:
        prot.to_csv(out / "protection.csv", index=False)
        lines += ["## Protection premium (bounds hold even for interrupted solves)\n", md(prot), "",
                  "Estimates weight cells by the panel design. Columns prefixed unweighted_ are raw market averages. "
                  "Coverage counts cells with paired solver observations; missing individual solves within those cells are not imputed.\n"]

    g = groups.groupby(["variant", "n", "exposure", "method", "group"])[GROUP_FIELDS].mean().reset_index()
    g.to_csv(out / "groups.csv", index=False)
    focus = g[(g.method.isin(["no_audit", args.plan_method])) & g.group.isin(["lower_income", "higher_income"])]
    keep = ["variant", "n", "exposure", "method", "group", "no_feasible_listing", "blocker_rate", "can_hide_share", "price_percentile_shown",
            "match_before", "match_after", "worse_off", "unavoidable_loss", "severity_positive", "severity_mean"]
    lines += ["## Lower vs. higher income (raw market averages)\n", md(focus[keep]), ""]
    scar = g[(g.method == "no_audit") & g.group.isin(["lower_income", "higher_income"])][
        ["variant", "n", "exposure", "group", "scarcity_no_option_in_cohort", "scarcity_none_in_draw", "scarcity_unmatched_full", "scarcity_matched_full"]]
    lines += ["## Scarcity categories (raw market averages)\n", md(scar.drop_duplicates(["variant", "n", "group"])), ""]
    (out / "summary.md").write_text("\n".join(lines))
    print(f"wrote {out / 'summary.md'} and CSV tables")


if __name__ == "__main__":
    main()
