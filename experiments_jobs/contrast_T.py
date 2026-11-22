"""The rental headline contrast T, computed for the jobs markets.

    python -m experiments_jobs.contrast_T jobs_replication
    python -m experiments_jobs.contrast_T jobs_frozen_noaudit --reps 10000

B_g(arm) is the share of group-g seekers with a hidden blocking pair (a feasible job they were not shown, that would take
them, and that they prefer to their match) when deferred acceptance runs on the arm's shortlists with no audit.
T = [B_lower(treated) - B_higher(treated)] - [B_lower(control) - B_higher(control)], in percentage points: how much the
treated rule widens the lower- minus higher-wage gap relative to random lists. This is the definition in
experiments/analyze.py, which reports T for the rentals (findings/gpt-luna-rentals/*/contrast_T.csv).

One difference from the rentals: analyze.py weights each cell by the borough's ACS recent-mover share. Jobs cells are
occupations with no comparable population weight in the design, so each cell (occupation x month) counts equally.

Rows, per variant: all seekers; "can_hide only" (seekers with more feasible jobs than shown, the only ones who can have a
hidden option of their own); and each month. 95% intervals resample whole seed blocks, as in analyze.py.
Needs runs.jsonl rows that carry blocker_count (written by run.py since 2026-11-21; jobs_frozen predates it, so use
configs/jobs_frozen_noaudit.json). Writes results/<name>/contrast_T.csv and contrast_T.md. Makes no API calls.
This analysis was added after the frozen run: logged under later_additions in jobs_manifest.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("lower_income", "higher_income")


def load(name: str, arms: set[str]) -> pd.DataFrame:
    rows = []
    for line in open(ROOT / "results" / name / "runs.jsonl"):
        r = json.loads(line)
        if r["method"] != "no_audit" or r["exposure"] not in arms:
            continue
        for g in GROUPS:
            o = r["outcomes"].get(g)
            if not o or o.get("n", 0) == 0:
                continue
            if "blocker_count" not in o:
                raise SystemExit(f"{name}: runs.jsonl has no blocker_count (run predates it); rerun with methods ['no_audit']")
            rate_h = o.get("blocker_rate_given_can_hide")
            rows.append({"variant": r["variant"], "cell": r["borough"], "month": r["month"], "seed": r["seed"],
                         "arm": r["exposure"], "group": g, "count": o["blocker_count"], "n": o["n"],
                         "count_h": (rate_h or 0.0) * o["can_hide_count"], "n_h": o["can_hide_count"]})
    return pd.DataFrame(rows)


def estimate(d: pd.DataFrame, treated: str, control: str, conditional: bool, reps: int, rng) -> dict:
    """B per (arm, group), equal cell weights; T and a seed-block bootstrap interval."""
    c, n = ("count_h", "n_h") if conditional else ("count", "n")
    d = d.assign(cm=d.cell + "|" + d.month)
    seeds = np.sort(d.seed.unique())
    cells = np.sort(d.cm.unique())
    tables = {}
    for arm in (treated, control):
        for g in GROUPS:
            x = d[(d.arm == arm) & (d.group == g)]
            tables[arm, g] = [x.pivot_table(index="seed", columns="cm", values=v, aggfunc="sum")
                              .reindex(index=seeds, columns=cells).fillna(0).to_numpy() for v in (c, n)]

    def B(weights):  # weights: (draws, seeds) multiplicities -> (draws,) equal-cell-weighted rate per arm/group
        out = {}
        for key, (cnt, tot) in tables.items():
            num, den = weights @ cnt, weights @ tot
            with np.errstate(invalid="ignore", divide="ignore"):
                rate = np.where(den > 0, num / den, np.nan)
            out[key] = np.nanmean(rate, axis=1)
        return out

    def T(b):
        return (b[treated, "lower_income"] - b[treated, "higher_income"]) - (b[control, "lower_income"] - b[control, "higher_income"])

    point = B(np.ones((1, len(seeds))))
    draws = np.stack([np.bincount(rng.integers(0, len(seeds), len(seeds)), minlength=len(seeds)) for _ in range(reps)])
    boot = T(B(draws.astype(float)))
    boot = boot[np.isfinite(boot)]
    lo, hi = (np.quantile(boot, [0.025, 0.975]) if len(boot) else (np.nan, np.nan))
    return {"markets": d[d.arm == treated].drop_duplicates(["cm", "seed"]).shape[0], "valid_bootstrap_draws": len(boot),
            **{f"B_{g}_{a}": float(point[a, g][0]) for a in (treated, control) for g in GROUPS},
            "T_pp": 100 * float(T(point)[0]), "T_lo_pp": 100 * float(lo), "T_hi_pp": 100 * float(hi)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("--treated", nargs="+", default=["llm_k5", "llm_q50", "transparent_k5"])
    ap.add_argument("--reps", type=int, default=10000)
    args = ap.parse_args()
    controls = {t: ("uniform_q50" if t.endswith("q50") else "uniform_k5") for t in args.treated}
    d = load(args.name, set(args.treated) | set(controls.values()))
    rng = np.random.default_rng(0)
    rows = []
    for variant in d.variant.unique():
        v = d[d.variant == variant]
        for t in args.treated:
            if t not in set(v.arm):
                continue
            subsets = [("all", v, False), ("can_hide only", v, True)] + [(f"all, {m}", v[v.month == m], False)
                                                                         for m in sorted(v.month.unique())]
            for label, x, cond in subsets:
                rows.append({"variant": variant, "treated": t, "control": controls[t], "seekers": label,
                             **estimate(x, t, controls[t], cond, args.reps, rng)})
    out = pd.DataFrame(rows)
    path = ROOT / "results" / args.name
    out.to_csv(path / "contrast_T.csv", index=False)
    show = out[["variant", "treated", "seekers", "markets", "T_pp", "T_lo_pp", "T_hi_pp"]].round(2)
    md = "\n".join(["| " + " | ".join(show.columns) + " |", "|" + "---|" * len(show.columns)]
                   + ["| " + " | ".join(str(x) for x in r) + " |" for r in show.itertuples(index=False)])
    (path / "contrast_T.md").write_text(f"# Contrast T: {args.name}\n\nCells weighted equally; {args.reps} seed-block draws.\n\n{md}\n")
    print(md)
    print(f"wrote {path / 'contrast_T.csv'} and contrast_T.md")


if __name__ == "__main__":
    main()
