"""Reuse saved recommender answers in a config whose markets differ only in preference weights.

    python -m experiments_jobs.remap_llm configs/jobs_frozen.json configs/jobs_frozen_calibrated.json

The market RNG streams ignore the preference weights (w_wage, seeker_noise, ...), so two configs that differ only there draw
the same seekers and postings with the same feasible sets. The recommender prompt carries no preference weights either.
Only market_id changes, because it hashes every spec field. That breaks the lookup of saved answers, and it also changes
the seed that shuffles posting order in the prompt. This script pairs markets by (cell, month, size, seed, variant) and
rewrites market_id, so the saved answers are reused. Caveat to report: the answers were given to prompts with the posting
order of the source config. A second one: random and rule-based shortlists seed on market_id, so those arms are redrawn (the recommender's lists are reused as they are). It checks that each paired market has the same seekers, postings and feasibility before
remapping, and makes no API calls.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments_jobs.config import domain_modules, expand_labeled, load_config

ROOT = Path(__file__).resolve().parents[1]


def key(spec, label):
    return (spec.cell, spec.month, spec.n_seekers, spec.n_postings, spec.seed, label)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("target")
    args = ap.parse_args()
    src, dst = load_config(args.source), load_config(args.target)
    markets, _, _ = domain_modules(dst)
    src_specs = {key(s, l): s for s, l in expand_labeled(src)}
    catalog = markets.JobCatalog(months=tuple(dst["grid"]["months"]))
    id_map = {}
    for spec, label in expand_labeled(dst):
        old = src_specs.get(key(spec, label))
        if old is None:
            raise SystemExit(f"no source market for {spec.market_id} ({label})")
        a, b = markets.build_market(old, catalog), markets.build_market(spec, catalog)
        same = (np.array_equal(a.inst.E, b.inst.E) and a.listings.index.equals(b.listings.index)
                and a.renters.index.equals(b.renters.index))
        if not same:
            raise SystemExit(f"{spec.market_id}: seekers, postings or feasibility differ from {old.market_id}; cannot remap")
        id_map[old.market_id] = spec.market_id
    src_recs, out = ROOT / src["llm_results"], ROOT / dst["llm_results"]
    rows = [json.loads(line) for line in src_recs.read_text().splitlines()]
    kept = [{**r, "market_id": id_map[r["market_id"]], "remapped_from": r["market_id"]} for r in rows if r["market_id"] in id_map]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in kept))
    print(f"{len(id_map)} markets paired and checked; {len(kept):,} of {len(rows):,} answers written to {out}")


if __name__ == "__main__":
    main()
