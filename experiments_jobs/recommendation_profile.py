"""What does each exposure policy actually show? The protocol's observed-recommendation endpoints.

    python -m experiments_jobs.recommendation_profile configs/jobs_frozen.json [--variant primary]

For every seeker with at least two feasible postings, and for every exposure policy (the LLM and the rule-based ones), compute
over the postings shown:
    salary_pct     mean within-options percentile of posted salary (0 = lowest-paying option, 1 = highest)
    seniority_pct  mean within-options percentile of seniority (0 = most junior option, 1 = most senior)
    level_gap      mean (posting seniority - seeker experience) of what is shown, minus the same mean over all options
                   (negative = shown jobs are more junior than the seeker's options on average)
Random exposure gives about 0.5, 0.5 and 0 in expectation. Percentiles use average ranks, so ties count as half.
Means by policy and wage group, with 95% intervals that resample whole seed blocks, as in analyze.py.

Market-level concentration (does a policy point many seekers at the same postings?):
    coverage       distinct postings shown to anyone / postings feasible for anyone in the market
    top3_share     share of all recommendations that go to the market's three most-recommended postings
    hhi            Herfindahl index of recommendations across postings (1/n when spread evenly over n postings)
    unseen_share   postings feasible for someone but shown to no one
Random exposure spreads recommendations; herding shows up as low coverage, high top3_share and high hhi.

Cross-group overlap (are higher- and lower-wage seekers sent to the same postings?), per market:
    shown_to_both      among postings both groups could take, the share recommended to at least one seeker of each group
    hi_recs_shared     share of higher-wage recommendations going to postings also recommended to a lower-wage seeker
    lo_recs_shared     the same, the other way round
    hi_lost_to_lo      share of higher-wage recommendations whose posting is filled by a lower-wage seeker in the
                       provisional matching (DA on what was shown, before audits)
    lo_lost_to_hi      the same, the other way round

Writes results/<name>/rec_profile.md, rec_profile.csv, rec_concentration.csv, rec_overlap.csv. No API calls; reads the saved
LLM answers.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments_jobs.config import domain_modules, expand_labeled, load_config
from hiddenoptions.matching import deferred_acceptance

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["salary_pct", "seniority_pct", "level_gap"]


def pct_rank(x: np.ndarray) -> np.ndarray:
    """Average-rank percentile in [0, 1]."""
    r = pd.Series(x).rank(method="average").to_numpy()
    return (r - 1) / (len(x) - 1)


def profile_market(market, e0: np.ndarray) -> list[dict]:
    s2p = market.slot_to_posting
    first = np.unique(s2p, return_index=True)[1]
    n_post = len(first)
    E_post = np.zeros((market.inst.n_b, n_post), bool)
    shown_post = np.zeros_like(E_post)
    for slot, post in enumerate(s2p):
        E_post[:, post] |= market.inst.E[:, slot]
        shown_post[:, post] |= e0[:, slot]
    salary = market.listings.salary.to_numpy(float)[first]
    seniority = market.listings.seniority.to_numpy(float)[first]
    experience = market.renters.experience.to_numpy(float)
    lower = market.renters.lower_wage.to_numpy(bool)
    rows = []
    for b in range(market.inst.n_b):
        opts = np.nonzero(E_post[b])[0]
        if len(opts) < 2:
            continue
        shown = shown_post[b, opts]
        if not shown.any():
            continue
        gap = seniority[opts] - experience[b]
        rows.append({"group": "lower_wage" if lower[b] else "higher_wage",
                     "salary_pct": float(pct_rank(salary[opts])[shown].mean()),
                     "seniority_pct": float(pct_rank(seniority[opts])[shown].mean()),
                     "level_gap": float(gap[shown].mean() - gap.mean()),
                     "n_options": len(opts), "n_shown": int(shown.sum())})
    return rows


def concentration(market, e0: np.ndarray) -> dict:
    s2p = market.slot_to_posting
    n_post = int(s2p.max()) + 1
    E_post = np.zeros((market.inst.n_b, n_post), bool)
    shown_post = np.zeros_like(E_post)
    for slot, post in enumerate(s2p):
        E_post[:, post] |= market.inst.E[:, slot]
        shown_post[:, post] |= e0[:, slot]
    reachable = E_post.any(0)
    counts = shown_post.sum(0)[reachable].astype(float)
    total = counts.sum()
    if total == 0:
        return {}
    share = counts / total
    return {"coverage": float((counts > 0).mean()), "top3_share": float(np.sort(share)[::-1][:3].sum()),
            "hhi": float((share ** 2).sum()), "unseen_share": float((counts == 0).mean()), "reachable_postings": int(reachable.sum())}


def overlap(market, e0: np.ndarray) -> dict:
    s2p = market.slot_to_posting
    n_post = int(s2p.max()) + 1
    E_post = np.zeros((market.inst.n_b, n_post), bool)
    shown_post = np.zeros_like(E_post)
    for slot, post in enumerate(s2p):
        E_post[:, post] |= market.inst.E[:, slot]
        shown_post[:, post] |= e0[:, slot]
    lower = market.renters.lower_wage.to_numpy(bool)
    if lower.all() or not lower.any():
        return {}
    both_eligible = E_post[lower].any(0) & E_post[~lower].any(0)
    to_lo, to_hi = shown_post[lower].any(0), shown_post[~lower].any(0)
    # Who fills each posting in the provisional matching: +1 lower-wage, -1 higher-wage, 0 unfilled
    _, match_s = deferred_acceptance(market.inst, e0)
    filled_by = np.zeros(n_post)
    for slot, b in enumerate(match_s):
        if b >= 0:
            filled_by[s2p[slot]] = 1 if lower[b] else -1
    hi_recs, lo_recs = shown_post[~lower], shown_post[lower]
    out = {"both_eligible": int(both_eligible.sum())}
    if both_eligible.any():
        out["shown_to_both"] = float((to_lo & to_hi)[both_eligible].mean())
    if hi_recs.sum():
        out["hi_recs_shared"] = float((hi_recs * to_lo).sum() / hi_recs.sum())
        out["hi_lost_to_lo"] = float((hi_recs * (filled_by == 1)).sum() / hi_recs.sum())
    if lo_recs.sum():
        out["lo_recs_shared"] = float((lo_recs * to_hi).sum() / lo_recs.sum())
        out["lo_lost_to_hi"] = float((lo_recs * (filled_by == -1)).sum() / lo_recs.sum())
    return out


def ci(df: pd.DataFrame, col: str, reps: int = 2000) -> tuple[float, float]:
    per_seed = df.groupby("seed")[col].mean().dropna().to_numpy()
    if len(per_seed) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(0)
    means = per_seed[rng.integers(0, len(per_seed), (reps, len(per_seed)))].mean(1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--variant", default="primary", help="variant label to profile (default primary)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    markets_mod, exposure_mod, _ = domain_modules(cfg)
    catalog = markets_mod.JobCatalog(months=tuple(cfg["grid"]["months"]))
    llm_path = ROOT / cfg["llm_results"] if cfg.get("llm_results") else None
    llm = exposure_mod.JobLlmRecommendations(llm_path) if llm_path and llm_path.exists() else None

    rows, conc, ovl = [], [], []
    specs = [s for s, label in expand_labeled(cfg) if label == args.variant]
    for i, spec in enumerate(specs, 1):
        market = markets_mod.build_market(spec, catalog)
        for exp in cfg["exposures"]:
            built = exposure_mod.build_exposure(market, exp, llm)
            if built is None:
                continue
            for r in profile_market(market, built[0]):
                rows.append({"exposure": exp["name"], "seed": spec.seed, "cell": spec.cell, **r})
            c = concentration(market, built[0])
            if c:
                conc.append({"exposure": exp["name"], "seed": spec.seed, "cell": spec.cell, **c})
            o = overlap(market, built[0])
            if o:
                ovl.append({"exposure": exp["name"], "seed": spec.seed, "cell": spec.cell, **o})
        if i % 40 == 0 or i == len(specs):
            print(f"  {i}/{len(specs)} markets", flush=True)

    df = pd.DataFrame(rows)
    out_rows = []
    for (exp, group), part in df.groupby(["exposure", "group"]):
        row = {"exposure": exp, "group": group, "seekers": len(part)}
        for f in FIELDS:
            lo, hi = ci(part, f)
            row.update({f: part[f].mean(), f"{f}_lo": lo, f"{f}_hi": hi})
        out_rows.append(row)
    table = pd.DataFrame(out_rows)
    cdf = pd.DataFrame(conc)
    conc_rows = []
    for exp, part in cdf.groupby("exposure"):
        row = {"exposure": exp, "markets": len(part)}
        for f in ("coverage", "top3_share", "hhi", "unseen_share"):
            lo, hi = ci(part, f)
            row.update({f: part[f].mean(), f"{f}_lo": lo, f"{f}_hi": hi})
        conc_rows.append(row)
    conc_table = pd.DataFrame(conc_rows)
    odf = pd.DataFrame(ovl)
    OV = ["shown_to_both", "hi_recs_shared", "lo_recs_shared", "hi_lost_to_lo", "lo_lost_to_hi"]
    ovl_rows = []
    for exp, part in odf.groupby("exposure"):
        row = {"exposure": exp, "markets": len(part)}
        for f in OV:
            if f in part and part[f].notna().any():
                lo, hi = ci(part, f)
                row.update({f: part[f].mean(), f"{f}_lo": lo, f"{f}_hi": hi})
        ovl_rows.append(row)
    ovl_table = pd.DataFrame(ovl_rows)
    ovl_show = ovl_table[["exposure", "markets"] + [f for f in OV if f in ovl_table]]
    by_cell = df.groupby(["exposure", "cell", "group"])[FIELDS].mean().reset_index()

    out = ROOT / "results" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "rec_profile.csv", index=False)
    conc_table.to_csv(out / "rec_concentration.csv", index=False)
    ovl_table.to_csv(out / "rec_overlap.csv", index=False)
    by_cell.to_csv(out / "rec_profile_by_cell.csv", index=False)

    def md(d):
        d = d.copy()
        for c in d.select_dtypes("number"):
            d[c] = d[c].round(3)
        lines = ["| " + " | ".join(d.columns) + " |", "|" + "---|" * len(d.columns)]
        return "\n".join(lines + ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)])

    show = table[["exposure", "group", "seekers", "salary_pct", "salary_pct_lo", "salary_pct_hi",
                  "seniority_pct", "seniority_pct_lo", "seniority_pct_hi", "level_gap"]]
    (out / "rec_profile.md").write_text(
        f"# What each policy shows: {cfg['name']}, variant {args.variant}\n\n"
        "Within each seeker's feasible postings: mean salary and seniority percentile of what is shown (random = 0.5), and the "
        "seniority-minus-experience gap of what is shown relative to all options (random = 0). Seekers with at least two "
        "options. 95% seed-block intervals.\n\n" + md(show) +
        "\n\n## Concentration across seekers (per market)\n\n" + md(conc_table[["exposure", "markets", "coverage", "top3_share",
        "top3_share_lo", "top3_share_hi", "hhi", "unseen_share"]]) + "\n\n## Cross-group overlap (per market)\n\n" + md(ovl_show) +
        "\n\n## By cell\n\n" + md(by_cell) + "\n")
    print(md(show))
    print("\nConcentration across seekers:")
    print(md(conc_table[["exposure", "markets", "coverage", "top3_share", "top3_share_lo", "top3_share_hi", "hhi", "unseen_share"]]))
    print("\nCross-group overlap:")
    print(md(ovl_show))
    print(f"\nwrote {out / 'rec_profile.md'}, rec_profile.csv, rec_profile_by_cell.csv")


if __name__ == "__main__":
    main()
