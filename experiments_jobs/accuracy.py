"""Accuracy vs placement for the jobs markets, as in the Chile and NYC rental tests.

    python -m experiments_jobs.accuracy configs/jobs_frozen.json            # primary variant
    python -m experiments_jobs.accuracy configs/jobs_frozen.json --variant all

Definitions (the team's: "did the list contain where the person would end up"):
    accuracy   among seekers who get a job under full information (seeker-proposing deferred acceptance on every
               feasible job), the share whose full-information job is on their shortlist
    placed     share of all seekers who get a job when deferred acceptance runs on the shortlists alone
    top shown  share of seekers whose own most-preferred feasible job is on their shortlist (context only)

The claim to test: a rule that is MORE accurate than random places FEWER people. For each pair below the script reports
the paired difference (rule minus random, same markets) with a 95% interval that resamples whole seed blocks, as in
analyze.py, and labels the result only when both intervals exclude zero.

Reads the saved GPT answers; makes no API calls. Writes results/<name>/accuracy.md and accuracy.csv.
This analysis was added after the frozen comparison was run: log it under later_additions in jobs_manifest.json.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments_jobs.config import domain_modules, expand_labeled, load_config
from hiddenoptions.matching import deferred_acceptance

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("all", "lower_wage", "higher_wage")
# (rule, baseline) pairs compared, by exposure name in the config; missing names are skipped
PAIRS = [("llm_k5", "uniform_k5"), ("llm_q50", "uniform_q50"), ("transparent_k5", "uniform_k5")]
LABELS = {"uniform_k5": "Random (5 jobs)", "llm_k5": "GPT (5 jobs)", "transparent_k5": "Honest ranker (5 jobs)",
          "salary_max_k5": "Highest pay (5 jobs)", "salary_min_k5": "Lowest pay (5 jobs)",
          "under_leveled_k5": "Most junior (5 jobs)", "uniform_q50": "Random (half list)", "llm_q50": "GPT (half list)"}


def seed_ci(per_seed: pd.Series, reps: int = 2000) -> tuple[float, float]:
    x = per_seed.dropna().to_numpy()
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(0)
    means = x[rng.integers(0, len(x), (reps, len(x)))].mean(1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def seeker_rows(market, e0: np.ndarray, full_b: np.ndarray, top_slot: np.ndarray) -> list[dict]:
    """One row per seeker for one shortlist rule. e0 lights every slot of a shown posting, so checking the seeker's own
    full-information slot is the same as checking its posting."""
    short_b, _ = deferred_acceptance(market.inst, e0)
    lower = market.renters.lower_wage.to_numpy(bool)
    rows = []
    for b in range(market.inst.n_b):
        has_full = full_b[b] >= 0
        rows.append({"group": "lower_wage" if lower[b] else "higher_wage",
                     "has_full_match": float(has_full),
                     "accurate": float(e0[b, full_b[b]]) if has_full else np.nan,
                     "top_shown": float(e0[b, top_slot[b]]) if top_slot[b] >= 0 else np.nan,
                     "placed": float(short_b[b] >= 0)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--variant", default="primary", help='variant label, or "all"')
    args = ap.parse_args()
    cfg = load_config(args.config)
    markets_mod, exposure_mod, _ = domain_modules(cfg)
    catalog = markets_mod.JobCatalog(months=tuple(cfg["grid"]["months"]))
    llm_path = ROOT / cfg["llm_results"] if cfg.get("llm_results") else None
    llm = exposure_mod.JobLlmRecommendations(llm_path) if llm_path and llm_path.exists() else None
    if llm is None:
        print("note: no saved GPT answers found; GPT rules will be skipped")

    specs = [(s, lab) for s, lab in expand_labeled(cfg) if args.variant == "all" or lab == args.variant]
    rows = []
    for i, (spec, label) in enumerate(specs, 1):
        market = markets_mod.build_market(spec, catalog)
        inst = market.inst
        full_b, _ = deferred_acceptance(inst, inst.E)
        top_slot = np.where(inst.E.any(1), np.argmin(inst.rank_b, axis=1), -1)
        for exp in cfg["exposures"]:
            built = exposure_mod.build_exposure(market, exp, llm)
            if built is None:
                continue
            for r in seeker_rows(market, built[0], full_b, top_slot):
                rows.append({"variant": label, "exposure": exp["name"], "seed": spec.seed, "market": spec.market_id, **r})
        if i % 40 == 0 or i == len(specs):
            print(f"  {i}/{len(specs)} markets", flush=True)
    df = pd.DataFrame(rows)

    def subset(group):
        return df if group == "all" else df[df.group == group]

    # levels by rule and group
    levels = []
    for group in GROUPS:
        d = subset(group)
        for exp, part in d.groupby("exposure", sort=False):
            row = {"rule": LABELS.get(exp, exp), "exposure": exp, "group": group}
            for col in ("accurate", "placed", "top_shown"):
                per_seed = part.groupby("seed")[col].mean()  # NaNs (no full-information job) are skipped
                lo, hi = seed_ci(per_seed)
                row.update({col: part[col].mean(), f"{col}_lo": lo, f"{col}_hi": hi})
            row["full_match_share"] = part.has_full_match.mean()
            levels.append(row)
    levels = pd.DataFrame(levels)

    # paired differences: rule minus random on the same markets
    diffs = []
    for rule, base in PAIRS:
        if rule not in set(df.exposure) or base not in set(df.exposure):
            continue
        for group in GROUPS:
            d = subset(group)
            row = {"rule": LABELS.get(rule, rule), "baseline": LABELS.get(base, base), "group": group}
            for col in ("accurate", "placed"):
                a = d[d.exposure == rule].groupby("seed")[col].mean()
                b = d[d.exposure == base].groupby("seed")[col].mean()
                diff = (a - b).dropna()
                lo, hi = seed_ci(diff)
                row.update({f"d_{col}": diff.mean(), f"d_{col}_lo": lo, f"d_{col}_hi": hi})
            more_acc = row["d_accurate_lo"] > 0
            less_acc = row["d_accurate_hi"] < 0
            fewer = row["d_placed_hi"] < 0
            more = row["d_placed_lo"] > 0
            if more_acc and fewer:
                verdict = "more accurate, fewer placed (supports the claim)"
            elif more_acc and more:
                verdict = "more accurate, more placed (against the claim)"
            elif less_acc and fewer:
                verdict = "less accurate, fewer placed (claim does not apply)"
            elif less_acc and more:
                verdict = "less accurate, more placed"
            else:
                verdict = "no clear difference on at least one measure"
            row["verdict"] = verdict
            diffs.append(row)
    diffs = pd.DataFrame(diffs)

    out = ROOT / "results" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    levels.to_csv(out / "accuracy.csv", index=False)
    diffs.to_csv(out / "accuracy_pairs.csv", index=False)

    def md(d: pd.DataFrame) -> str:
        d = d.copy()
        for c in d.select_dtypes("number"):
            d[c] = d[c].round(3)
        lines = ["| " + " | ".join(d.columns) + " |", "|" + "---|" * len(d.columns)]
        return "\n".join(lines + ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in d.itertuples(index=False)])

    lv = levels[["rule", "group", "accurate", "accurate_lo", "accurate_hi", "placed", "placed_lo", "placed_hi", "top_shown", "full_match_share"]]
    pr = diffs[["rule", "baseline", "group", "d_accurate", "d_accurate_lo", "d_accurate_hi", "d_placed", "d_placed_lo", "d_placed_hi", "verdict"]] if len(diffs) else diffs
    text = (f"# Accuracy vs placement: {cfg['name']} ({args.variant})\n\n"
            "Accuracy: share of seekers (with a full-information job) whose full-information job was on their shortlist. "
            "Placed: share of all seekers matched using only their shortlist. 95% intervals resample seed blocks.\n\n"
            "## Levels\n\n" + md(lv) + "\n\n## Rule minus random (paired, same markets)\n\n" + md(pr) + "\n")
    (out / "accuracy.md").write_text(text)
    print("\n" + md(pr))
    print(f"\nwrote {out / 'accuracy.md'}, accuracy.csv, accuracy_pairs.csv")


if __name__ == "__main__":
    main()
