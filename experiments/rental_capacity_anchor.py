"""Capacity decomposition and one reserved (anchor) slot for the NYC rental markets: the jobs analysis
(experiments_jobs/capacity_anchor.py) applied to the GPT rental runs, so both studies can be read the same way.

    python -m experiments.rental_capacity_anchor configs/heldout_primary.json --llm path/to/gpt_recs.jsonl

Part 1: change in placed = change in capacity - change in shortfall, each rule minus random (uniform_k5) on the same
markets. Capacity is the largest matching the shown pairs allow; shortfall is the part of it deferred acceptance misses.
Part 2: each renter's list keeps 5 slots, but one is reserved for a listing from a largest assignment over all feasible
pairs that reuses shown pairs where it can. GPT's last-ranked pick is dropped to make room; the honest ranker drops its
lowest-scored listing (u_observable); random drops one at random (seeded).

Same functions as the jobs script (capacity, anchor assignment, list-length check); listings have one unit each.
Arms: uniform_k5, gpt_k5, transparent_k5 (claude_k5 too if its replies are in the --llm file). Means are unweighted over
markets, as in experiments/rental_accuracy.py, so placement levels can be checked against findings/gpt-luna-rentals/
accuracy.txt; 95% intervals resample seed blocks. Markets the catalog cannot build (InsufficientSupply) are skipped,
as in the rental runs. No API calls. Writes results/<name>/rental_capacity_anchor.md and three csv files.
"""
from __future__ import annotations

import argparse
import hashlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.config import expand_labeled, load_config
from experiments_jobs.capacity_anchor import apply_anchor, capacity
from hiddenoptions.exposure import LlmRecommendations, build_exposure
from hiddenoptions.markets import Catalog, InsufficientSupply, build_market
from hiddenoptions.matching import deferred_acceptance

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("uniform_k5", "gpt_k5", "transparent_k5", "claude_k5")
LABELS = {"uniform_k5": "Random (5)", "gpt_k5": "GPT (5)", "transparent_k5": "Honest ranker (5)", "claude_k5": "Claude Haiku (5)"}
_cat, _llm = None, None


def _init(path):
    global _cat, _llm
    _cat, _llm = Catalog(), LlmRecommendations(path)


def drop_order(market, exp):
    name, policy = exp["name"], exp["policy"]
    rng = np.random.default_rng(int(hashlib.sha256(f"{market.spec.market_id}|{market.spec.seed}|anchor|{name}".encode()).hexdigest()[:8], 16))
    if policy == "llm":
        recs = _llm.table.get(market.spec.market_id, {}).get(name, {})

        def order(b, current):
            cur = set(current.tolist())
            ranked = [int(s) for s in (recs.get((int(b), exp.get("rep", 0))) or {}).get("listings", []) if int(s) in cur]
            return [s for s in current.tolist() if s not in ranked] + ranked[::-1]  # last-ranked pick dropped first
        return order
    if policy == "transparent":
        u = np.asarray(market.u_observable)
        return lambda b, current: current[np.argsort(u[b, current], kind="stable")]  # lowest score dropped first
    return lambda b, current: current[rng.permutation(len(current))]


def one(task):
    spec, label, exposures = task
    try:
        m = build_market(spec, _cat)
    except InsufficientSupply:
        return []
    inst, E = m.inst, np.asarray(m.inst.E, bool)
    n, s2p = inst.n_b, np.arange(E.shape[1])
    lower = m.renters.lower_income.to_numpy(bool)
    full_b, _ = deferred_acceptance(inst, E)
    base = {"variant": label, "seed": spec.seed, "borough": spec.borough, "month": spec.month,
            "full_capacity": 100 * capacity(E) / n, "full_placed": 100 * float((full_b >= 0).mean())}
    rows = []
    for exp in exposures:
        built = build_exposure(m, exp, _llm)
        if built is None:
            continue
        e0 = np.asarray(built[0], bool)
        placed = deferred_acceptance(inst, e0)[0] >= 0
        cap = capacity(e0)
        e1, changed = apply_anchor(E, e0, s2p, drop_order(m, exp))
        anchored = deferred_acceptance(inst, e1)[0] >= 0
        rows.append({**base, "exposure": exp["name"], "capacity": 100 * cap / n, "placed": 100 * placed.mean(),
                     "shortfall": 100 * (cap - placed.sum()) / n,
                     "placed_lower": 100 * placed[lower].mean() if lower.any() else np.nan,
                     "placed_higher": 100 * placed[~lower].mean() if (~lower).any() else np.nan,
                     "anchor_placed": 100 * anchored.mean(),
                     "anchor_placed_lower": 100 * anchored[lower].mean() if lower.any() else np.nan,
                     "anchor_placed_higher": 100 * anchored[~lower].mean() if (~lower).any() else np.nan,
                     "lists_changed": 100 * changed / n, "list_length_kept": bool((e1.sum(1) == e0.sum(1)).all())})
    return rows


def seed_ci(s: pd.Series, reps: int) -> tuple[float, float]:
    x = s.dropna().to_numpy()
    rng = np.random.default_rng(0)
    m = x[rng.integers(0, len(x), (reps, len(x)))].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def md(d: pd.DataFrame) -> str:
    d = d.copy()
    for c in d.select_dtypes("number"):
        d[c] = d[c].round(2 if c == "capacity_share_of_drop" else 1)
    return "\n".join(["| " + " | ".join(d.columns) + " |", "|" + "---|" * len(d.columns)]
                     + ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in d.itertuples(index=False)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--llm", required=True, help="saved replies (jsonl), e.g. gunzipped findings/gpt-luna-rentals/<window>/gpt_recs.jsonl.gz")
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()
    cfg = load_config(args.config)
    exposures = [e for e in cfg["exposures"] if e["name"] in ARMS]
    tasks = [(s, label, exposures) for s, label in expand_labeled(cfg)]
    with Pool(args.workers, initializer=_init, initargs=(str(Path(args.llm).resolve()),)) as pool:
        df = pd.DataFrame([r for rows in pool.imap_unordered(one, tasks, chunksize=4) for r in rows])
    if not df.list_length_kept.all():
        print("warning: the anchor changed a list's length; check apply_anchor")

    per_seed = lambda d, c: d.groupby("seed")[c].mean()
    ci = lambda s: (s.mean(), *seed_ci(s, args.reps))
    levels, decomp, anchor = [], [], []
    for variant, dv in df.groupby("variant", sort=False):
        b = dv[dv.exposure == "uniform_k5"]
        for name in [a for a in ARMS if a in set(dv.exposure)]:
            d = dv[dv.exposure == name]
            rule = LABELS[name]
            levels.append({"variant": variant, "rule": rule, "markets": len(d), "full_capacity": d.full_capacity.mean(),
                           "full_placed": d.full_placed.mean(), "capacity": d.capacity.mean(), "placed": d.placed.mean(),
                           "shortfall": d.shortfall.mean(), "placed_lower": d.placed_lower.mean(), "placed_higher": d.placed_higher.mean()})
            a = {"variant": variant, "rule": rule, "lists_changed": d.lists_changed.mean(), "placed": d.placed.mean(),
                 "anchor_placed": d.anchor_placed.mean()}
            for tag, before, after in (("gain", "placed", "anchor_placed"), ("gain_lower", "placed_lower", "anchor_placed_lower"),
                                       ("gain_higher", "placed_higher", "anchor_placed_higher")):
                a[tag], a[f"{tag}_lo"], a[f"{tag}_hi"] = ci(per_seed(d, after) - per_seed(d, before))
            if name != "uniform_k5":
                a["anchored_vs_random"], a["anchored_vs_random_lo"], a["anchored_vs_random_hi"] = ci(per_seed(d, "anchor_placed") - per_seed(b, "placed"))
                r = {"variant": variant, "rule": rule, "baseline": LABELS["uniform_k5"]}
                for col in ("placed", "capacity", "shortfall", "placed_lower", "placed_higher"):
                    r[f"d_{col}"], r[f"d_{col}_lo"], r[f"d_{col}_hi"] = ci(per_seed(d, col) - per_seed(b, col))
                r["capacity_share_of_drop"] = r["d_capacity"] / r["d_placed"] if abs(r["d_placed"]) > 1e-9 else np.nan
                decomp.append(r)
            anchor.append(a)
    levels, decomp, anchor = pd.DataFrame(levels), pd.DataFrame(decomp), pd.DataFrame(anchor)
    out = ROOT / "results" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    levels.to_csv(out / "rental_capacity_levels.csv", index=False)
    decomp.to_csv(out / "rental_capacity_decomposition.csv", index=False)
    anchor.to_csv(out / "rental_anchor.csv", index=False)
    show_d = decomp[["variant", "rule", "d_placed", "d_placed_lo", "d_placed_hi", "d_capacity", "d_shortfall", "capacity_share_of_drop"]]
    show_a = anchor[[c for c in ("variant", "rule", "lists_changed", "placed", "anchor_placed", "gain", "gain_lo", "gain_hi",
                                 "anchored_vs_random", "anchored_vs_random_lo", "anchored_vs_random_hi") if c in anchor]]
    text = (f"# Rental capacity decomposition and anchor: {cfg['name']}\n\nPercentage points of renters; unweighted means over "
            f"markets; 95% seed-block intervals ({args.reps} draws).\n\n## Levels\n\n{md(levels)}\n\n## Rule minus random\n\n{md(show_d)}"
            f"\n\n## One reserved slot\n\n{md(show_a)}\n")
    (out / "rental_capacity_anchor.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
