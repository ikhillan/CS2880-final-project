"""Replay outcomes with two landlord income coefficients on the same new shocks."""
import argparse
from hashlib import blake2b
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def load_evaluator(path):
    spec = importlib.util.spec_from_file_location("learned_evaluation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shocks(key, shape):
    payload = json.dumps(["screen-priority-v1", *key], separators=(",", ":")).encode()
    rng = np.random.default_rng(int.from_bytes(blake2b(payload, digest_size=8).digest(), "little"))
    return rng.standard_normal(shape)


def evaluate(data, split, evaluator):
    meta = json.loads((data / f"{split}.json").read_text())
    selected = np.array([key[0] < meta["seed_start"] + 8 for key in meta["keys"]])
    keys = [key for key, take in zip(meta["keys"], selected) if take]
    assert len(keys) == 8 * 24
    with np.load(data / f"{split}.npz") as raw:
        e, pref, lower, ur, tr, u = (raw[k][selected] for k in
                                      ("e", "pref", "lower", "uniform", "transparent", "u"))
        ratio = raw["x"][selected, :, :, 4].astype(float)
    with np.load(data / f"{split}-scores.npz") as scorefile:
        scores = {p: scorefile[p][selected] for p in ["context", "pointwise"]}
    with np.load(data / f"{split}-su.npz") as su:
        masks = {"uniform": ur < 5, "transparent": tr < 5, "su": su["mask"][selected]}
    for label, score in dict(transparent=u, **scores).items():
        for anchored in (False, True):
            if not anchored and label == "transparent":
                continue
            mask = np.stack([evaluator.make_mask(s, feas, key, anchored)
                             for s, feas, key in zip(score, e, keys)])
            masks[label + ("_anchor" if anchored else "")] = mask
    noise = np.stack([shocks(key, allowed.shape) for key, allowed in zip(keys, e)])
    rows = []
    for coefficient in (2.5, 0.0):
        observed = np.where(e, coefficient * ratio + noise, -np.inf)
        seller = np.argsort(np.argsort(-observed, axis=1, kind="stable"), axis=1)
        seller = np.where(e, seller, 127).astype(np.uint8)
        for label, mask in masks.items():
            assert not (mask & ~e).any()
            assert np.array_equal(mask.sum(2), np.minimum(5, e.sum(2)))
            for rollout in ("all", "higher"):
                graph = mask if rollout == "all" else np.where(lower[..., None], masks["uniform"], mask)
                if rollout == "higher":
                    assert np.array_equal(graph[lower], masks["uniform"][lower])
                matched = evaluator.da(pref, seller, graph) < e.shape[2]
                for i, (seed, design, month, borough) in enumerate(keys):
                    lo = lower[i]
                    m = matched[i]
                    rows.append(dict(split=split, seed=seed, design=design, month=month,
                                     borough=borough, weight=meta["weights"][borough],
                                     screening=coefficient, policy=label, rollout=rollout,
                                     lower_n=lo.sum(), higher_n=(~lo).sum(), total_n=len(lo),
                                     lower_matched=(m & lo).sum(), higher_matched=(m & ~lo).sum(),
                                     total_matched=m.sum()))
    return pd.DataFrame(rows)


def summarize(counts):
    indices = ["split", "seed", "design", "screening", "policy", "rollout"]
    block = []
    for group in ("lower", "higher", "total"):
        w = counts["weight"]
        partial = counts[indices].assign(
            num=w * counts[group + "_matched"], den=w * counts[group + "_n"]
        ).groupby(indices, as_index=False).sum()
        primary = partial.assign(group=group, rate=100 * partial.num / partial.den,
                                 estimand="ratio_of_weighted_counts")
        block.append(primary)
        # Companion estimate uses the same fixed geographic weights for each group.
        cell = counts[indices + ["weight"]].assign(
            rate=100 * counts[group + "_matched"] / counts[group + "_n"])
        assert cell.rate.notna().all()
        cell = cell.assign(num=cell.weight * cell.rate, den=cell.weight)
        cell = cell.groupby(indices, as_index=False)[["num", "den"]].sum()
        cell = cell.assign(group=group, rate=cell.num / cell.den,
                           estimand="fixed_weight_cell_rates")
        block.append(cell)
    block = pd.concat(block, ignore_index=True)
    idx = [i for i in indices if i != "policy"] + ["group", "estimand"]
    u = block[block.policy == "uniform"][idx + ["rate"]].rename(columns={"rate": "u"})
    block = block.merge(u, on=idx, validate="many_to_one")
    block["delta"] = block.rate - block.u
    idx = [i for i in indices if i != "screening"] + ["group", "estimand"]
    reference = block[block.screening == 2.5][idx + ["delta"]].rename(columns={"delta": "reference"})
    block = block.merge(reference, on=idx, validate="many_to_one")
    block["change"] = block.delta - block.reference
    ci = student_t.ppf(.975, 7) / np.sqrt(8)
    summary = []
    for key, group in block.groupby([i for i in indices if i != "seed"] + ["group", "estimand"]):
        assert len(group) == 8
        row = dict(zip([i for i in indices if i != "seed"] + ["group", "estimand"], key))
        for col in ("rate", "delta", "change"):
            mean = group[col].mean()
            width = ci * group[col].std()
            row.update({col: mean, col + "_lo": mean - width, col + "_hi": mean + width})
        summary.append(row)
    return block, pd.DataFrame(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--evaluator", type=Path,
                        default=Path(__file__).resolve().parent.parent / "learned" / "evaluate.py")
    args = parser.parse_args()
    evaluator = load_evaluator(args.evaluator)
    counts = pd.concat([evaluate(args.data, s, evaluator) for s in ("summer", "winter")])
    assert len(counts) == 2 * 192 * 2 * 8 * 2
    args.out.mkdir(exist_ok=True, parents=True)
    counts.to_csv(args.out / "screen-counts.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    block, summary = summarize(counts)
    block.to_csv(args.out / "screen-blocks.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    summary.to_csv(args.out / "screen-summary.csv", index=False)


if __name__ == "__main__":
    main()
