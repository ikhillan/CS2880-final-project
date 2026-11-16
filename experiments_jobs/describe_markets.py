"""Are the jobs cells and sampled markets usable? No preferences, exposure or API needed. Writes results/<name>/viability.md.

    python -m experiments_jobs.describe_markets configs/jobs_pilot.json

Three tables:
  cells     per cell x month: postings, employers, top-employer share, salary quartiles, seekers by group, share with options
  markets   per cell x variant over the config's seeds: the four-way partition by group, feasible degree, firms, max matching
  partition the protocol's four-way partition (no option in the cell / option in the cell, none in the draw / unmatched under
            full DA / matched under full DA), pooled by group, which must be reported in the paper for every group
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from experiments_jobs.config import expand_labeled, load_config
from hiddenoptions.jobs.markets import JobCatalog
from hiddenoptions.jobs.markets import build_market
from hiddenoptions.markets import maximum_matching_size
from hiddenoptions.matching import deferred_acceptance

ROOT = Path(__file__).resolve().parents[1]


def table(df: pd.DataFrame) -> str:
    d = df.copy()
    for c in d.select_dtypes("number"):
        d[c] = d[c].round(2)
    rows = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    rows += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(rows)


def weighted_share(mask, w):
    return float(100 * w[mask].sum() / w.sum()) if w.sum() else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    labeled = expand_labeled(cfg)
    catalog = JobCatalog(months=tuple(cfg["grid"]["months"]))

    cells = []
    seen = set()
    for spec, _ in labeled:
        key = (spec.cell, spec.month, spec.share, spec.level_gap, spec.donors)
        if key in seen:
            continue
        seen.add(key)
        s, p = catalog.frame(*key)
        w = s.weight.to_numpy(float)
        firm_counts = p.firm_id.value_counts()
        cells.append({
            "cell": spec.cell, "month": spec.month, "level_gap": spec.level_gap, "postings": len(p), "employers": p.firm_id.nunique(),
            "top employer %": round(100 * firm_counts.iloc[0] / len(p), 1) if len(p) else None,
            "salary p25": p.salary.quantile(0.25), "salary p50": p.salary.quantile(0.5), "salary p75": p.salary.quantile(0.75),
            "seeker records": len(s), "lower %": weighted_share(s.lower_wage.to_numpy(bool), w),
            "lower with option %": weighted_share((s.group == "G+").to_numpy(), w) / max(weighted_share(s.lower_wage.to_numpy(bool), w), 1e-9) * 100,
            "higher with option %": weighted_share(((s.group == "higher") & (s.n_feasible_cell >= 1)).to_numpy(), w)
                                    / max(weighted_share(~s.lower_wage.to_numpy(bool), w), 1e-9) * 100,
            "lower median options": s.n_feasible_cell[s.lower_wage].median(),
            "higher median options": s.n_feasible_cell[~s.lower_wage].median(),
        })

    markets, partition = [], []
    for spec, label in labeled:
        m = build_market(spec, catalog)
        E = m.inst.E
        match_b, _ = deferred_acceptance(m.inst, E)
        n_cell = m.renters.n_feasible_cell.to_numpy()
        n_draw = E.sum(1)
        low = m.renters.lower_wage.to_numpy(bool)
        status = np.select([n_cell == 0, n_draw == 0, match_b < 0], ["no option in cell", "none in draw", "unmatched, full DA"], "matched, full DA")
        for g, mask in (("lower", low), ("higher", ~low)):
            for st in ("no option in cell", "none in draw", "unmatched, full DA", "matched, full DA"):
                partition.append({"variant": label, "group": g, "status": st, "share": float((status[mask] == st).mean()) if mask.any() else np.nan})
        markets.append({"variant": label, "cell": spec.cell, "month": spec.month, "seed": spec.seed,
                        "lower none in draw %": 100 * float((n_draw[low] == 0).mean()), "higher none in draw %": 100 * float((n_draw[~low] == 0).mean()),
                        "lower median options": float(np.median(n_draw[low])), "higher median options": float(np.median(n_draw[~low])),
                        "employer firms": m.n_firms_listing, "postings per firm": m.listings.posting_id.nunique() / max(m.n_firms_listing, 1),
                        "max matching %": 100 * maximum_matching_size(E) / spec.n_seekers, "components": m.inst.n_components})

    md = pd.DataFrame(markets).drop(columns=["seed"]).groupby(["variant", "cell"]).mean(numeric_only=True).reset_index()
    pt = pd.DataFrame(partition).groupby(["variant", "group", "status"]).share.mean().unstack("status").reset_index()
    pt = pt[["variant", "group", "no option in cell", "none in draw", "unmatched, full DA", "matched, full DA"]]
    out = ROOT / "results" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    text = (f"# Market viability: {cfg['name']}\n\nCells x months from the config; {cfg['grid']['seeds']} seeds per market cell.\n\n"
            "## Four-way partition by group (share of sampled seekers)\n\n" + table(pt) +
            "\n\n## Sampled markets, mean over seeds and months\n\n" + table(md) +
            "\n\n## Cells (occupation x month), seekers weighted by PWGTP\n\n" + table(pd.DataFrame(cells)) + "\n")
    (out / "viability.md").write_text(text)
    print(f"wrote {out / 'viability.md'}")
    print(table(pt))


if __name__ == "__main__":
    main()
