"""Audit a subset of screening replays against the team's separate serial DA."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from screening import load_evaluator, shocks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    args = p.parse_args()
    root = args.root.resolve()
    dest = Path(__file__).resolve().parent
    base = root / "work/ec-ai-learn"
    sys.path.insert(0, str(root / "work/todd-project"))
    from hiddenoptions.matching import Instance, NOT_EDGE, deferred_acceptance

    evaluator = load_evaluator(dest.parent / "learned/evaluate.py")
    counts = pd.read_csv(dest / "screen-counts.csv.gz")
    tests = 0
    for split in ("summer", "winter"):
        keys = json.loads((base / f"{split}.json").read_text())["keys"]
        with np.load(base / f"{split}.npz") as file:
            reduced = {k: file[k][:24] for k in ("e", "pref", "x", "lower", "uniform", "transparent", "u")}
        with np.load(base / f"{split}-scores.npz") as file:
            scores = {k: file[k][:24] for k in ("pointwise", "context")}
        with np.load(base / f"{split}-su.npz") as file:
            su = file["mask"][:24]
        for i, key in enumerate(keys[:24]):
            e, pref, x, low, uniform, transparent, u = (reduced[k][i] for k in
                ("e", "pref", "x", "lower", "uniform", "transparent", "u"))
            masks = dict(uniform=uniform<5, transparent=transparent<5, su=su[i])
            for name, s in dict(transparent=u, **{k: v[i] for k,v in scores.items()}).items():
                for anchor in (False,True):
                    if name == "transparent" and not anchor:
                        continue
                    masks[name+("_anchor" if anchor else "")] = evaluator.make_mask(s,e,key,anchor)
            noise = shocks(key,e.shape)
            for beta in (0.,2.5):
                score = np.where(e,beta*x[...,4].astype(float)+noise,-np.inf)
                # Direct counts of higher scores compute the ranks without double argsort.
                rank = (score[:,None,:] < score[None,:,:]).sum(axis=1)
                inst = Instance(np.where(e,pref.astype(int),NOT_EDGE),np.where(e,rank,NOT_EDGE),
                                np.zeros(64,int),np.zeros(64,int),np.ones(1))
                for name, mask in masks.items():
                    for rollout in ("all","higher"):
                        graph = mask if rollout == "all" else np.where(low[:,None],masks["uniform"],mask)
                        hit = deferred_acceptance(inst,graph)[0]>=0
                        seed,design,month,borough = key
                        row = counts[(counts.split==split)&(counts.seed==seed)&(counts.design==design)&
                                     (counts.month==month)&(counts.borough==borough)&(counts.policy==name)&
                                     (counts.screening==beta)&(counts.rollout==rollout)]
                        assert len(row)==1
                        assert row[["total_matched","lower_matched","higher_matched"]].iloc[0].tolist()==[
                            hit.sum(), (hit&low).sum(), (hit&~low).sum()]
                        tests += 1
    protocol = sha256((dest/"SCREENING-PROTOCOL.md").read_bytes()).hexdigest()
    assert protocol=="d7d4d82a0b1fc1e7451590719cce4d0d35f75702c327523648bb233f4d47e7b8"
    (dest/"screen-audit.json").write_text(json.dumps(
        {"serial_da_checks":tests,"protocol_sha":protocol,"allocation_rows":len(counts)},indent=2)+"\n")


if __name__ == "__main__":
    main()
