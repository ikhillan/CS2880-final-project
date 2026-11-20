"""Accuracy vs placement for the NYC rental markets, with the jobs definition (experiments_jobs/accuracy.py):
accuracy = among renters matched under full information (DA on every feasible pair), share whose full-information
listing is on their shortlist; placed = share matched when DA runs on the shortlists alone.
Usage: python -m experiments.rental_accuracy configs/stage1_pilot.json configs/heldout_primary.json configs/heldout_replication.json
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from experiments.config import expand_labeled, load_config
from hiddenoptions.exposure import LlmRecommendations, build_exposure
from hiddenoptions.markets import Catalog, InsufficientSupply, build_market
from hiddenoptions.matching import deferred_acceptance

ARMS = ["uniform_k5", "claude_k5", "gpt_k5"]
_cat, _llm = None, None


def init(path):
    global _cat, _llm
    _cat, _llm = Catalog(), LlmRecommendations(path)


def one(task):
    spec, label, exposures = task
    try:
        m = build_market(spec, _cat)
    except InsufficientSupply:
        return None
    full_b, _ = deferred_acceptance(m.inst, m.inst.E)
    matched_full = full_b >= 0
    out = {"label": label, "cfg_seed": spec.seed}
    for ex in exposures:
        built = build_exposure(m, ex, _llm)
        if built is None:
            return None
        e0 = built[0]
        b, _ = deferred_acceptance(m.inst, e0)
        idx = np.where(matched_full)[0]
        out[ex["name"]] = (float(np.mean(e0[idx, full_b[idx]])) if len(idx) else np.nan, float(np.mean(b >= 0)))
    return out


def main():
    root = Path(".").resolve()
    for path in sys.argv[1:]:
        cfg = load_config(path)
        exposures = [e for e in cfg["exposures"] if e["name"] in ARMS]
        tasks = [(s, label, exposures) for s, label in expand_labeled(cfg)]
        with Pool(7, initializer=init, initargs=(str(root / cfg["llm_results"]),)) as pool:
            rows = [r for r in pool.map(one, tasks, chunksize=8) if r]
        for label in ("enriched", "primary"):
            rs = [r for r in rows if r["label"] == label]
            base = np.array([r["uniform_k5"] for r in rs])
            line = [f"{cfg['name']:20s} {label:9s} n={len(rs)}"]
            for a in ARMS:
                x = np.array([r[a] for r in rs])
                d = x - base
                rng = np.random.default_rng(0)
                boot = np.array([np.nanmean(d[rng.integers(0, len(d), len(d))], axis=0) for _ in range(2000)])
                lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
                line.append(f"{a}: acc {100*np.nanmean(x[:,0]):.1f} placed {100*np.nanmean(x[:,1]):.1f}"
                            + ("" if a == "uniform_k5" else f" (d acc {100*np.nanmean(d[:,0]):+.1f} [{100*lo[0]:+.1f},{100*hi[0]:+.1f}],"
                               f" d placed {100*np.nanmean(d[:,1]):+.1f} [{100*lo[1]:+.1f},{100*hi[1]:+.1f}])"))
            print(" | ".join(line), flush=True)


if __name__ == "__main__":
    main()
