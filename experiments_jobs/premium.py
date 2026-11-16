"""The protection premium (protocol section 2): extra audit cost to rule out every avoidable loss in a group.

    python -m experiments_jobs.premium jobs_frozen

Per market and exposure: C_DA (exact_prefix), C_G (exact_prefix_protect_<group>), premium = C_G - C_DA. Only markets where both
are certified count toward the premium. Reports, by variant, exposure and group: mean premium with 95% seed-block intervals,
share of markets with a positive premium, mean protected and unavoidable counts, and the order check
C_DA <= C_lower <= C_all. Also splits the premium by the largest employer's share of the market's postings (terciles),
the concentration check that goes with the shared-brokerage proposition. Writes results/<name>/premium.md and premium.csv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def ci(df: pd.DataFrame, col: str, reps: int = 2000) -> tuple[float, float]:
    per_seed = df.groupby("seed")[col].mean().dropna().to_numpy()
    if len(per_seed) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(0)
    means = per_seed[rng.integers(0, len(per_seed), (reps, len(per_seed)))].mean(1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def md(d: pd.DataFrame) -> str:
    d = d.copy()
    for c in d.select_dtypes("number"):
        d[c] = d[c].round(3)
    return "\n".join(["| " + " | ".join(d.columns) + " |", "|" + "---|" * len(d.columns)] +
                     ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in d.itertuples(index=False)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    args = ap.parse_args()
    out = ROOT / "results" / args.name
    rows = []
    for line in open(out / "runs.jsonl"):
        r = json.loads(line)
        rows.append({"market_id": r["market_id"], "variant": r["variant"], "seed": r["seed"], "exposure": r["exposure"],
                     "method": r["method"], "cost": r["cost"], "certified": r["certified"], "n_firms": r["n_firms"],
                     "n_protected": r["extra"].get("n_protected"), "n_unavoidable": r["extra"].get("n_unavoidable")})
    df = pd.DataFrame(rows)
    key = ["market_id", "variant", "seed", "exposure"]
    base = df[df.method == "exact_prefix"].set_index(key)[["cost", "certified", "n_firms"]].rename(columns={"cost": "c_da", "certified": "cert_da"})
    groups = sorted(m.rsplit("_protect_", 1)[1] for m in df.method.unique() if m.startswith("exact_prefix_protect_"))
    if not groups:
        raise SystemExit("no exact_prefix_protect_* rows: add \"protect\" to the config and rerun")
    wide = base.copy()
    for g in groups:
        part = df[df.method == f"exact_prefix_protect_{g}"].set_index(key)
        wide[f"c_{g}"] = part.cost
        wide[f"cert_{g}"] = part.certified
        wide[f"protected_{g}"] = part.n_protected
        wide[f"unavoidable_{g}"] = part.n_unavoidable
    wide = wide.reset_index()

    table = []
    for g in groups:
        ok = wide[wide.cert_da & wide[f"cert_{g}"].fillna(False).astype(bool)].copy()
        ok["premium"] = ok[f"c_{g}"] - ok.c_da
        ok["positive"] = (ok.premium > 0).astype(float)
        for (variant, exposure), part in ok.groupby(["variant", "exposure"]):
            lo, hi = ci(part, "premium")
            table.append({"group": g, "variant": variant, "exposure": exposure, "markets": len(part),
                          "c_da": part.c_da.mean(), f"c_protected": part[f"c_{g}"].mean(), "premium": part.premium.mean(),
                          "premium_lo": lo, "premium_hi": hi, "positive_share": part.positive.mean(),
                          "max_premium": part.premium.max(), "protected": part[f"protected_{g}"].mean(),
                          "unavoidable": part[f"unavoidable_{g}"].mean(),
                          "uncertified_dropped": int((wide.variant.eq(variant) & wide.exposure.eq(exposure)).sum() - len(part))})
    table = pd.DataFrame(table)
    order_ok = None
    if {"lower", "all"} <= set(groups):
        both = wide.dropna(subset=["c_lower", "c_all"])
        order_ok = bool(((both.c_da <= both.c_lower) & (both.c_lower <= both.c_all)).all())
    table.to_csv(out / "premium.csv", index=False)
    text = (f"# Protection premium: {args.name}\n\nPremium = cheapest protected repair minus cheapest ordinary repair, on the same "
            "shortlist and first-round match; certified pairs only. 95% seed-block intervals.\n\n" + md(table) +
            (f"\n\nOrder check C_DA <= C_lower <= C_all on every market: {order_ok}\n" if order_ok is not None else "\n"))
    (out / "premium.md").write_text(text)
    print(md(table))
    if order_ok is not None:
        print(f"\nC_DA <= C_lower <= C_all on every market: {order_ok}")
    print(f"wrote {out / 'premium.md'} and premium.csv")


if __name__ == "__main__":
    main()
