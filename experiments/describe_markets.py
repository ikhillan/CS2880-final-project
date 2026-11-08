"""Are the cells and sampled markets usable? Needs no preferences, exposure or API. Writes results/viability.md.

    python -m experiments.describe_markets
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hiddenoptions.markets import Catalog, MarketSpec, build_market, maximum_matching_size

ROOT = Path(__file__).resolve().parents[1]
BOROUGHS = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]


def table(df: pd.DataFrame) -> str:
    rows = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    rows += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def main():
    catalog = Catalog()
    cells, markets = [], []
    for month in catalog.listings:
        for borough in BOROUGHS:
            for share in (0.30, 0.40, 0.50):
                f, l = catalog.frame(borough, month, share)
                w = f.weight
                s = lambda g: round(100 * w[f.group == g].sum() / w.sum(), 1)
                gp = f[f.group == "G+"]
                cells.append({"month": month, "borough": borough, "share": share, "listings": len(l), "mover_records": len(f), "G+ %": s("G+"), "G0 %": s("G0"),
                              "higher %": s("higher"), "G+ median options": gp.n_feasible_cell.median() if len(gp) else "-",
                              "higher median options": f[f.group == "higher"].n_feasible_cell.median()})
    for design in ("natural", "stratified"):
        for borough in BOROUGHS[:4]:
            for n in (24, 64):
                rows = []
                for seed in range(20):
                    m = build_market(MarketSpec(borough, "2026-08", seed, n, n, design=design), catalog)
                    low, E = m.renters.lower_income.to_numpy(bool), m.inst.E
                    rows.append({"lower no option %": 100 * (E[low].sum(1) == 0).mean(), "lower median options": np.median(E[low].sum(1)), "higher median options": np.median(E[~low].sum(1)),
                                 "listing firms": m.n_firms_listing, "listings per firm": n / m.n_firms_listing, "max matching %": 100 * maximum_matching_size(E) / n, "components": m.inst.n_components})
                markets.append({"design": design, "borough": borough, "n": n, **pd.DataFrame(rows).mean().round(1).to_dict()})
    out = ROOT / "results" / "viability.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("# Market viability\n\nAugust 2026 markets, 20 seeds each, rent at most 40% of income.\n\n## Sampled markets\n\n" + table(pd.DataFrame(markets))
                   + "\n\n## Cells (borough x month), weighted by WGTP, recent movers\n\n" + table(pd.DataFrame(cells)) + "\n")
    print(f"wrote {out}")
    print(table(pd.DataFrame(markets)))


if __name__ == "__main__":
    main()
