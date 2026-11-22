"""Group-switch decomposition of T for the jobs markets (protocol section 2), as the rentals report it.

    python -m experiments_jobs.group_switch configs/jobs_frozen.json
    python -m experiments_jobs.group_switch configs/jobs_replication.json --variants primary natural --reps 10000

For each market and each pair (A = random lists, B = the treated rule), deferred acceptance runs four times: lower-wage
seekers see A or B, crossed with higher-wage seekers seeing A or B (hiddenoptions.metrics.group_switch, the function the
rental runner uses). With Z = B_lower - B_higher, the gap in the share with a hidden blocking pair:
    S_L = [(Z_BA - Z_AA) + (Z_BB - Z_AB)] / 2   effect of switching lower-wage seekers' lists, averaged over both orders
    S_H = [(Z_AB - Z_AA) + (Z_BB - Z_BA)] / 2   the same for higher-wage seekers' lists
    S_L + S_H = Z_BB - Z_AA = T                 (checked)
Same formulas as experiments/analyze.py switch_table, except that cells (occupation x month) count equally, as in
experiments_jobs/contrast_T.py; T here must equal contrast_T's T on the same run (checked when contrast_T.csv exists).

Rebuilds markets and reads saved GPT answers (no API calls). Writes results/<name>/group_switch.csv, group_switch.md and
group_switch_markets.jsonl. Added after the frozen run: logged under later_additions in jobs_manifest.json.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(var, "1")

import numpy as np
import pandas as pd

from experiments_jobs.config import domain_modules, expand_labeled, load_config
from hiddenoptions.metrics import group_switch

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("AA", "BA", "AB", "BB")  # letters: policy shown to lower-wage, then higher-wage seekers
DEFAULT_PAIRS = [("uniform_k5", "llm_k5"), ("uniform_q50", "llm_q50"), ("uniform_k5", "transparent_k5")]
_state = {}


def _init(cfg):
    markets, exposure, _ = domain_modules(cfg)
    _state.update(cfg=cfg, markets=markets, exposure=exposure,
                  catalog=markets.JobCatalog(months=tuple(cfg["grid"]["months"])),
                  llm=exposure.JobLlmRecommendations(str(ROOT / cfg["llm_results"])) if cfg.get("llm_results") else None)


def _one(task):
    spec, label, pairs = task
    s = _state
    market = s["markets"].build_market(spec, s["catalog"])
    by_name = {e["name"]: e for e in s["cfg"]["exposures"]}
    e0 = {}
    for name in {n for p in pairs for n in p}:
        built = s["exposure"].build_exposure(market, by_name[name], s["llm"])
        if built is not None:
            e0[name] = built[0]
    out = {"market_id": spec.market_id, "variant": label, "cell": spec.cell, "month": spec.month, "seed": spec.seed, "switch": {}}
    for a, b in pairs:
        if a in e0 and b in e0:
            out["switch"][f"{b}_vs_{a}"] = group_switch(market, e0[a], e0[b])
    return out


def decompose(frame: pd.DataFrame, reps: int, rng) -> dict:
    """frame: one row per market x arm, with lower/higher counts and sizes. Equal cell weights, seed-block bootstrap."""
    frame = frame.assign(cm=frame.cell + "|" + frame.month)
    seeds, cells = np.sort(frame.seed.unique()), np.sort(frame.cm.unique())
    tab = {}
    for arm in ARMS:
        x = frame[frame.arm == arm]
        for g in ("lower", "higher"):
            tab[arm, g] = [x.pivot_table(index="seed", columns="cm", values=v, aggfunc="sum").reindex(index=seeds, columns=cells)
                           .fillna(0).to_numpy() for v in (f"{g}_count", f"{g}_n")]

    def Z(w):  # (draws, seeds) -> dict arm -> (draws,) gap, plus B levels
        z, B = {}, {}
        for arm in ARMS:
            for g in ("lower", "higher"):
                num, den = w @ tab[arm, g][0], w @ tab[arm, g][1]
                with np.errstate(invalid="ignore", divide="ignore"):
                    B[arm, g] = np.nanmean(np.where(den > 0, num / den, np.nan), axis=1)
            z[arm] = B[arm, "lower"] - B[arm, "higher"]
        return z, B

    def parts(z):
        s_l = ((z["BA"] - z["AA"]) + (z["BB"] - z["AB"])) / 2
        s_h = ((z["AB"] - z["AA"]) + (z["BB"] - z["BA"])) / 2
        return s_l, s_h, z["BB"] - z["AA"]

    z0, B0 = Z(np.ones((1, len(seeds))))
    s_l, s_h, t = (float(v[0]) for v in parts(z0))
    draws = np.stack([np.bincount(rng.integers(0, len(seeds), len(seeds)), minlength=len(seeds)) for _ in range(reps)]).astype(float)
    bl, bh, bt = parts(Z(draws)[0])
    q = lambda v: np.nanquantile(v, [0.025, 0.975]) * 100
    (l_lo, l_hi), (h_lo, h_hi), (t_lo, t_hi) = q(bl), q(bh), q(bt)
    labels = {"AA": "UU", "BA": "TU", "AB": "UT", "BB": "TT"}  # U = random, T = treated; lower-wage letter first
    return {"markets": int(frame[frame.arm == "AA"].shape[0]),
            **{f"B_{g}_{labels[a]}": float(B0[a, g][0]) for a in ARMS for g in ("lower", "higher")},
            "S_L_pp": 100 * s_l, "S_L_lo": l_lo, "S_L_hi": l_hi, "S_H_pp": 100 * s_h, "S_H_lo": h_lo, "S_H_hi": h_hi,
            "T_pp": 100 * t, "T_lo": t_lo, "T_hi": t_hi, "S_L_plus_S_H_equals_T": bool(abs(s_l + s_h - t) < 1e-12)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--variants", nargs="+", default=["primary", "natural"])
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    cfg = load_config(args.config)
    names = {e["name"] for e in cfg["exposures"]}
    pairs = [p for p in DEFAULT_PAIRS if set(p) <= names]
    out = ROOT / "results" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    tasks = [(s, label, pairs) for s, label in expand_labeled(cfg) if label in args.variants]
    print(f"{cfg['name']}: {len(tasks)} markets, pairs {pairs}", flush=True)
    rows = []
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(cfg,)) as pool, \
            open(out / "group_switch_markets.jsonl", "w") as fh:
        for i, r in enumerate(pool.imap_unordered(_one, tasks), 1):
            fh.write(json.dumps(r) + "\n")
            for sw, arms in r["switch"].items():
                for arm in ARMS:
                    rows.append({"switch": sw, "variant": r["variant"], "cell": r["cell"], "month": r["month"], "seed": r["seed"],
                                 "arm": arm, **arms[arm]})
            if i % 80 == 0:
                print(f"  {i}/{len(tasks)} markets", flush=True)
    d = pd.DataFrame(rows)
    rng = np.random.default_rng(0)
    res = pd.DataFrame([{"switch": sw, "variant": v, **decompose(f, args.reps, rng)} for (sw, v), f in d.groupby(["switch", "variant"])])
    ct = out / "contrast_T.csv"
    if ct.exists():  # T must match contrast_T.py on the same run
        c = pd.read_csv(ct)
        c = c[c.seekers == "all"].assign(switch=lambda x: x.treated + "_vs_" + x.control)[["switch", "variant", "T_pp"]]
        m = res.merge(c, on=["switch", "variant"], suffixes=("", "_contrast"))
        res["T_matches_contrast_T"] = res.apply(lambda r: bool(np.isclose(r.T_pp, m.loc[(m.switch == r.switch) & (m.variant == r.variant), "T_pp_contrast"]).all()), axis=1)
    res.to_csv(out / "group_switch.csv", index=False)
    show = res[["switch", "variant", "markets", "S_L_pp", "S_L_lo", "S_L_hi", "S_H_pp", "S_H_lo", "S_H_hi", "T_pp"]].round(2)
    md = "\n".join(["| " + " | ".join(show.columns) + " |", "|" + "---|" * len(show.columns)]
                   + ["| " + " | ".join(str(x) for x in r) + " |" for r in show.itertuples(index=False)])
    (out / "group_switch.md").write_text(f"# Group switch: {cfg['name']}\n\nCells weighted equally; {args.reps} seed-block draws; pp.\n\n{md}\n")
    print(md)
    if "T_matches_contrast_T" in res:
        print("T matches contrast_T:", bool(res.T_matches_contrast_T.all()))
    print(f"wrote {out / 'group_switch.csv'}")


if __name__ == "__main__":
    main()
