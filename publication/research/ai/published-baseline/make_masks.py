"""Create fixed-budget masks using only observable inputs in prepared panels."""
import argparse
from hashlib import blake2b
import json
from pathlib import Path

import numpy as np

from su_baseline import su_topk


def keyed_policy(u, v, e, key, k):
    payload = json.dumps(["su22-topk-v1", *key], separators=(",", ":")).encode()
    seed = int.from_bytes(blake2b(payload, digest_size=8).digest(), "little")
    rng = np.random.default_rng(seed)
    rows = rng.permutation(e.shape[0])
    cols = rng.permutation(e.shape[1])
    ix = np.ix_(rows, cols)
    out = np.zeros_like(e, dtype=bool)
    out[ix] = su_topk(u[ix], v[ix], e[ix], k)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("splits", nargs="+")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    for split in args.splits:
        with np.load(args.data / f"{split}.npz", allow_pickle=False) as data:
            e, u, v = (data[name] for name in ["e", "u", "v"])
        keys = json.loads((args.data / f"{split}.json").read_text())["keys"]
        if len(keys) != len(e):
            raise ValueError("Market keys and tensors differ in length")
        mask = np.stack([keyed_policy(u[i], v[i], e[i], key, args.k)
                         for i, key in enumerate(keys)])
        assert not (mask & ~e.astype(bool)).any()
        assert (mask.sum(2) == np.minimum(args.k, e.sum(2))).all()
        out = args.data / f"{split}-su.npz"
        np.savez_compressed(out, mask=mask)
        print(out, len(mask), int(mask.sum()), flush=True)


if __name__ == "__main__":
    main()
