"""Why GPT's shortlists place fewer people, and whether one reserved slot fixes it (jobs markets).

    python -m experiments_jobs.capacity_anchor configs/jobs_frozen.json            # primary variant
    python -m experiments_jobs.capacity_anchor configs/jobs_frozen.json --variant all

Part 1, capacity decomposition (paper section 6). For each shortlist rule:
    capacity    the most seekers that COULD be placed using only the shown pairs (largest matching of the shown graph)
    placed      seekers actually placed by seeker-proposing deferred acceptance on the shown pairs
    shortfall   capacity - placed: places the shortlists allow but deferred acceptance does not reach
so that, against the random rule of the same list length on the same markets,
    change in placed = change in capacity - change in shortfall.
A drop in capacity means the lists themselves cannot hold everyone (many people sent to the same few jobs). A rise in
shortfall means the room was there but applicants collided on the way.

Part 2, anchor (paper section 5). Find a largest assignment of seekers to jobs over ALL feasible pairs that reuses as many
already-shown pairs as possible. Each assigned seeker whose job is not on their list gets it added, and the rule's last
choice is dropped, so every list keeps its length. Then deferred acceptance runs again. What is dropped:
    GPT             its last-ranked pick
    honest ranker   the shown job with the lowest score
    other rules     a shown job at random (seeded)

All numbers are percentage points of seekers. 95% intervals resample whole seed blocks, as in analyze.py.
Reads the saved GPT answers; makes no API calls. Writes results/<name>/capacity_anchor.md and three csv files.
Added after the frozen comparison was run: log it under later_additions in jobs_manifest.json.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from experiments_jobs.config import domain_modules, expand_labeled, load_config
from hiddenoptions.matching import deferred_acceptance

ROOT = Path(__file__).resolve().parents[1]
REPS = 2000  # seed-block bootstrap draws; --reps 10000 for the final tables, as the protocol asks
BIG = 10_000  # cost of an infeasible pair: larger than any market, so size is maximized before anything else
LABELS = {"uniform_k5": "Random (5 jobs)", "llm_k5": "GPT (5 jobs)", "transparent_k5": "Honest ranker (5 jobs)",
          "salary_max_k5": "Highest pay (5 jobs)", "salary_min_k5": "Lowest pay (5 jobs)",
          "under_leveled_k5": "Most junior (5 jobs)", "uniform_q50": "Random (half list)", "llm_q50": "GPT (half list)"}


def seed_ci(per_seed: pd.Series, reps: int | None = None) -> tuple[float, float]:
    x = per_seed.dropna().to_numpy()
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(0)
    means = x[rng.integers(0, len(x), (reps or REPS, len(x)))].mean(1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def capacity(graph: np.ndarray) -> int:
    """Size of a largest matching in a seeker x slot graph."""
    if not graph.any():
        return 0
    rows, cols = linear_sum_assignment(graph.astype(int), maximize=True)
    return int(graph[rows, cols].sum())


def anchor_assignment(E: np.ndarray, shown: np.ndarray) -> np.ndarray:
    """A largest assignment on E that uses as few hidden pairs as possible. Returns the slot per seeker, or -1."""
    cost = np.where(shown & E, 0, np.where(E, 1, BIG))
    rows, cols = linear_sum_assignment(cost)
    out = np.full(E.shape[0], -1)
    keep = E[rows, cols]
    out[rows[keep]] = cols[keep]
    return out


def posting_graph(graph: np.ndarray, s2p: np.ndarray) -> np.ndarray:
    out = np.zeros((graph.shape[0], int(s2p.max()) + 1), bool)
    np.logical_or.at(out, (slice(None), s2p), graph)
    return out


def apply_anchor(E: np.ndarray, shown: np.ndarray, s2p: np.ndarray, drop_order) -> tuple[np.ndarray, int]:
    """Add each seeker's anchor job to their list and drop one shown job, keeping list length.
    drop_order(b, shown_postings) returns the shown postings in the order they should be dropped (first = dropped first).
    Returns the new seeker x slot graph and the number of lists changed."""
    assign = anchor_assignment(E, shown)
    shown_post = posting_graph(shown, s2p)
    changed = 0
    for b, slot in enumerate(assign):
        if slot < 0:
            continue
        p = int(s2p[slot])
        if shown_post[b, p]:
            continue
        current = np.nonzero(shown_post[b])[0]
        if not len(current):
            continue  # an empty list (e.g. no usable GPT answer) stays empty, so no list ever grows
        shown_post[b, int(drop_order(b, current)[0])] = False
        shown_post[b, p] = True
        changed += 1
    return shown_post[:, s2p] & E, changed


def make_drop_order(market, exp: dict, llm):
    """The rule-specific choice of which shown job gives up its slot."""
    policy, name = exp["policy"], exp["name"]
    seed = int(hashlib.sha256(f"{market.spec.market_id}|{market.spec.seed}|anchor|{name}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    if policy == "llm" and llm is not None:
        recs = llm.table.get(market.spec.market_id, {}).get(name, {})

        def order(b, current):
            ranked = [p for p in (recs.get((int(b), exp.get("rep", 0))) or {}).get("listings", []) if p in set(current.tolist())]
            rest = [p for p in current.tolist() if p not in ranked]
            return rest + ranked[::-1]  # GPT's last-ranked pick goes first
        return order
    if policy == "transparent" and hasattr(market, "score_deterministic"):
        score = np.asarray(market.score_deterministic)

        def order(b, current):
            s = score[b, current] if score.ndim == 2 else score[current]
            return current[np.argsort(s, kind="stable")]  # lowest score goes first
        return order
    return lambda b, current: current[rng.permutation(len(current))]


def baseline_name(exp: dict, exposures: list[dict]) -> str | None:
    """The random rule with the same list length."""
    for other in exposures:
        if other["policy"] == "random" and other.get("k") == exp.get("k") and other.get("q") == exp.get("q"):
            return other["name"]
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--variant", default="primary", help='variant label, or "all"')
    ap.add_argument("--reps", type=int, default=2000, help="bootstrap draws (protocol final analysis: 10000)")
    args = ap.parse_args()
    global REPS
    REPS = args.reps
    cfg = load_config(args.config)
    markets_mod, exposure_mod, _ = domain_modules(cfg)
    catalog = markets_mod.JobCatalog(months=tuple(cfg["grid"]["months"]))
    llm_path = ROOT / cfg["llm_results"] if cfg.get("llm_results") else None
    llm = exposure_mod.JobLlmRecommendations(llm_path) if llm_path and llm_path.exists() else None
    if llm is None:
        print("note: no saved GPT answers found; GPT rules will be skipped")

    specs = [(s, lab) for s, lab in expand_labeled(cfg) if args.variant == "all" or lab == args.variant]
    if not specs:
        raise SystemExit(f"no markets for variant {args.variant!r}")
    rows = []
    for i, (spec, label) in enumerate(specs, 1):
        market = markets_mod.build_market(spec, catalog)
        inst, s2p = market.inst, np.asarray(market.slot_to_posting)
        E = np.asarray(inst.E, bool)
        n = inst.n_b
        lower = market.renters.lower_wage.to_numpy(bool)
        full_b, _ = deferred_acceptance(inst, E)
        base = {"variant": label, "seed": spec.seed, "market": spec.market_id,
                "full_capacity": 100 * capacity(E) / n, "full_placed": 100 * float((full_b >= 0).mean())}
        for exp in cfg["exposures"]:
            built = exposure_mod.build_exposure(market, exp, llm)
            if built is None:
                continue
            e0 = np.asarray(built[0], bool)
            match_b, _ = deferred_acceptance(inst, e0)
            placed = match_b >= 0
            cap = capacity(e0)
            e1, changed = apply_anchor(E, e0, s2p, make_drop_order(market, exp, llm))
            anchored = deferred_acceptance(inst, e1)[0] >= 0
            rows.append({**base, "exposure": exp["name"],
                         "capacity": 100 * cap / n, "placed": 100 * placed.mean(), "shortfall": 100 * (cap - placed.sum()) / n,
                         "placed_lower": 100 * placed[lower].mean(), "placed_higher": 100 * placed[~lower].mean(),
                         "anchor_capacity": 100 * capacity(e1) / n, "anchor_placed": 100 * anchored.mean(),
                         "anchor_placed_lower": 100 * anchored[lower].mean(), "anchor_placed_higher": 100 * anchored[~lower].mean(),
                         "lists_changed": 100 * changed / n,
                         "list_length_kept": bool((posting_graph(e1, s2p).sum(1) == posting_graph(e0, s2p).sum(1)).all())})
        if i % 40 == 0 or i == len(specs):
            print(f"  {i}/{len(specs)} markets", flush=True)
    df = pd.DataFrame(rows)
    if not df.list_length_kept.all():
        print("warning: the anchor changed a list's length in some market; check apply_anchor")

    def mean_ci(series_by_seed: pd.Series) -> dict:
        lo, hi = seed_ci(series_by_seed)
        return {"mean": series_by_seed.mean(), "lo": lo, "hi": hi}

    def per_seed(d: pd.DataFrame, col: str) -> pd.Series:
        return d.groupby("seed")[col].mean()

    levels, decomp, anchor = [], [], []
    for variant, dv in df.groupby("variant", sort=False):
        for exp in cfg["exposures"]:
            name = exp["name"]
            d = dv[dv.exposure == name]
            if d.empty:
                continue
            rule = LABELS.get(name, name)
            levels.append({"variant": variant, "rule": rule, "markets": len(d),
                           "full_capacity": d.full_capacity.mean(), "full_placed": d.full_placed.mean(),
                           "capacity": d.capacity.mean(), "placed": d.placed.mean(), "shortfall": d.shortfall.mean()})

            # part 2: anchor gain for this rule, and the anchored rule against plain random
            a = {"variant": variant, "rule": rule, "lists_changed": d.lists_changed.mean(),
                 "placed": d.placed.mean(), "anchor_placed": d.anchor_placed.mean()}
            for tag, before, after in (("gain", "placed", "anchor_placed"), ("gain_lower", "placed_lower", "anchor_placed_lower"),
                                       ("gain_higher", "placed_higher", "anchor_placed_higher")):
                s = mean_ci(per_seed(d, after) - per_seed(d, before))
                a.update({tag: s["mean"], f"{tag}_lo": s["lo"], f"{tag}_hi": s["hi"]})
            base_name = baseline_name(exp, cfg["exposures"])
            b = dv[dv.exposure == base_name] if base_name else dv.iloc[0:0]
            if not b.empty and base_name != name:
                s = mean_ci(per_seed(d, "anchor_placed") - per_seed(b, "placed"))
                a.update({"anchored_vs_random": s["mean"], "anchored_vs_random_lo": s["lo"], "anchored_vs_random_hi": s["hi"]})
            anchor.append(a)

            # part 1: paired decomposition against the random rule of the same list length
            if b.empty or base_name == name:
                continue
            r = {"variant": variant, "rule": rule, "baseline": LABELS.get(base_name, base_name)}
            for col in ("placed", "capacity", "shortfall"):
                s = mean_ci(per_seed(d, col) - per_seed(b, col))
                r.update({f"d_{col}": s["mean"], f"d_{col}_lo": s["lo"], f"d_{col}_hi": s["hi"]})
            r["capacity_share_of_drop"] = r["d_capacity"] / r["d_placed"] if abs(r["d_placed"]) > 1e-9 else np.nan
            decomp.append(r)
    levels, decomp, anchor = pd.DataFrame(levels), pd.DataFrame(decomp), pd.DataFrame(anchor)

    out = ROOT / "results" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    levels.to_csv(out / "capacity_levels.csv", index=False)
    decomp.to_csv(out / "capacity_decomposition.csv", index=False)
    anchor.to_csv(out / "anchor.csv", index=False)

    def md(d: pd.DataFrame) -> str:
        if d.empty:
            return "(nothing to show)"
        d = d.copy()
        for c in d.select_dtypes("number"):
            d[c] = d[c].round(1 if c != "capacity_share_of_drop" else 2)
        lines = ["| " + " | ".join(d.columns) + " |", "|" + "---|" * len(d.columns)]
        return "\n".join(lines + ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in d.itertuples(index=False)])

    text = (f"# Capacity decomposition and anchor: {cfg['name']} ({args.variant})\n\n"
            "All numbers are percentage points of seekers; 95% intervals resample seed blocks. "
            "Added after the frozen comparison (exploratory).\n\n"
            "## Levels\n\nfull_capacity and full_placed use every feasible pair; the rest use the shortlists only.\n\n" + md(levels)
            + "\n\n## Part 1: rule minus random, same markets\n\nd_placed = d_capacity - d_shortfall. "
              "capacity_share_of_drop is d_capacity / d_placed.\n\n" + md(decomp)
            + "\n\n## Part 2: one reserved slot (anchor), list length unchanged\n\n"
              "gain = placed with the anchor minus placed without. anchored_vs_random compares the anchored rule with "
              "plain random of the same length.\n\n" + md(anchor) + "\n")
    (out / "capacity_anchor.md").write_text(text)
    print("\nPart 1: rule minus random\n" + md(decomp))
    print("\nPart 2: anchor\n" + md(anchor))
    print(f"\nwrote {out / 'capacity_anchor.md'} and capacity_levels.csv, capacity_decomposition.csv, anchor.csv")


if __name__ == "__main__":
    main()
