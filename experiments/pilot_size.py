"""Calculate the protocol's pre-holdout block count from the pilot blocker counts."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.analyze import cell_weights
from experiments.config import expand_labeled, load_config
from experiments.pilot_outcomes import BASE, CONFIG, GROUPS


def jackknife_month(counts, design_weights):
    """Keep the supported group/cell target fixed over all leave-one-block-out samples."""
    groups = counts.set_index(["seed", "borough", "group"]).sort_index()
    seeds = list(groups.index.get_level_values("seed").unique())
    n_seeds = len(seeds)
    if n_seeds < 2 or groups.index.has_duplicates:
        raise ValueError("Need at least two independent blocks and unique group rows")
    boroughs = list(design_weights)
    idx = pd.MultiIndex.from_product([seeds, boroughs, GROUPS], names=groups.index.names)
    table = groups.reindex(idx, fill_value=0)
    counts_by_group = table[["haiku", "uniform", "n"]].to_numpy().reshape(n_seeds, len(boroughs), 2, 3)
    combined = counts_by_group.sum(axis=0)
    support = (combined[:, :, 2] > 0).all(axis=1)
    missing = [b for b, keep in zip(boroughs, support) if not keep]
    if not support.any():
        return None, None, missing
    weights = np.array(list(design_weights.values()))[support]
    weights /= weights.sum()

    def contrast(cube):
        num = cube[support, :, 0] - cube[support, :, 1]
        den = cube[support, :, 2]
        if (den == 0).any():
            return np.nan
        changes = weights @ (num / den)
        return changes[0] - changes[1]

    dropped = np.array([contrast(combined - block) for block in counts_by_group])
    variance = (n_seeds - 1) / n_seeds * np.sum((dropped - dropped.mean()) ** 2)
    return float(contrast(combined)), float(np.sqrt(variance)) if np.isfinite(variance) else None, missing


def planning_record(frame, weights):
    records = []
    for month, rows in frame.groupby("month"):
        monthly = {b: w for (b, m), w in weights.items() if m == month and w > 0}
        point, se, missing = jackknife_month(rows, monthly)
        records.append({"month": month, "T": point, "jackknife_se": se,
                        "unavailable_boroughs": missing, "pilot_blocks": int(rows.seed.nunique())})
    if len(records) != 3 or not all(r["pilot_blocks"] == 20 for r in records):
        raise ValueError("The planning rule requires the three development months and 20 seed blocks")
    se_available = all(r["jackknife_se"] is not None for r in records)
    scale = math.sqrt(20) * max(r["jackknife_se"] for r in records) if se_available else None
    unclamped = math.ceil((1.96 * scale / 0.01) ** 2) if se_available else None
    supported_plan = min(400, max(100, unclamped)) if se_available else None
    fallback = any(r["unavailable_boroughs"] for r in records) or not se_available
    return {"monthly": records, "s_supported": scale, "uncapped_B_supported": unclamped,
            "bounded_B_supported": supported_plan, "empty_cell_or_failed_variance": fallback,
            "protocol_B": 400 if fallback else supported_plan}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--counts", type=Path, default=BASE / "paired-outcomes-counts.csv")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    df = pd.read_csv(args.counts)
    primary = df[(df.variant == "primary") & (df.metric == "hidden_blocker")]
    planned = expand_labeled(load_config(CONFIG))
    weights = cell_weights(sorted({(spec.borough, spec.month) for spec, design in planned if design == "primary"}))
    record = planning_record(primary, weights)
    output = json.dumps(record, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
