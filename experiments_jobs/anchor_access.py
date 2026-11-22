"""Export and replay a frozen jobs panel without new LLM calls.

Run ``python -m experiments_jobs.anchor_access --help``. Export must run in an
authorized environment containing the original private inputs and saved menus.
Evaluation reads only an anonymous NumPy packet (allow_pickle=False).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import scipy

from experiments_jobs.capacity_anchor import capacity, posting_graph
from experiments_jobs.config import expand_labeled, load_config
from hiddenoptions.matching import Instance, NOT_EDGE, deferred_acceptance

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ("configs/jobs_frozen.json", "configs/jobs_replication.json")
FORMAT_VERSION = 1
CONTRASTS = [
    ("anchor_gpt", "gpt"), ("anchor_gpt", "random"),
    ("anchor_gpt", "anchor_random"), ("balanced_gpt", "anchor_gpt"),
    ("anchor_random", "random"), ("anchor_score", "score"),
    ("anchor_gpt", "score"), ("anchor_gpt", "anchor_score"),
    ("balanced_score", "anchor_gpt"), ("score_anchor_gpt", "anchor_gpt"),
    ("anchor_gpt_ties", "gpt"), ("anchor_gpt_ties", "random"),
    ("anchor_gpt_ties", "anchor_random_ties"),
]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def versions():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "pandas": pd.__version__}


def inputs(data_root):
    return [data_root / "data/job_postings/raw/revelio_job_postings.csv",
            data_root / "census_pums/acs_pums_2020_2024_nyc_persons.csv",
            data_root / "data/reference/cpi_u_us_city_average.csv",
            *[data_root / load_config(ROOT / c)["llm_results"] for c in CONFIGS]]


def preflight(data_root):
    records = [{"path": str(p), "exists": p.is_file(),
                "bytes": p.stat().st_size if p.is_file() else None} for p in inputs(data_root)]
    return {"status": "ready" if all(x["exists"] for x in records) else "missing_inputs",
            "inputs": records, "versions": versions(), "api_calls_required": 0}


def save_packet(out, metadata, arrays):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as stream:
        np.savez_compressed(stream, metadata=np.array(json.dumps(metadata, sort_keys=True)), **arrays)
    (out.with_suffix(".manifest.json")).write_text(json.dumps(
        {"packet_sha256": sha256(out), **metadata}, indent=2) + "\n")


def export_panel(data_root, out, variants):
    from hiddenoptions.jobs import markets as jm, exposure as je
    from experiments_jobs.coordinated_policies import normalize_score_ranks

    check = preflight(data_root)
    if check["status"] != "ready":
        print(json.dumps(check, indent=2))
        raise SystemExit("Missing original inputs. No markets or GPT outcomes were fabricated or skipped.")
    provenance = {str(p.relative_to(data_root)): sha256(p) for p in inputs(data_root)}
    expected = json.loads((ROOT / "revelio_2025_manifest.json").read_text())["sha256"]
    if provenance["data/job_postings/raw/revelio_job_postings.csv"] != expected:
        raise ValueError("Revelio input differs from the pinned extract. Version a separate study before continuing.")
    jm.ROOT = data_root
    jm.POSTINGS_FILE = str(data_root / "data/job_postings/raw/revelio_job_postings.csv")
    jm.CPI_CSV = data_root / "data/reference/cpi_u_us_city_average.csv"
    jm._POSTINGS_CACHE.clear()
    arrays, markets = {}, []
    for config_path in CONFIGS:
        cfg = load_config(ROOT / config_path)
        window = "frozen" if cfg["name"] == "jobs_frozen" else "replication"
        catalog = jm.JobCatalog(months=tuple(cfg["grid"]["months"]))
        llm = je.JobLlmRecommendations(data_root / cfg["llm_results"])
        exposures = {e["name"]: e for e in cfg["exposures"]}
        for spec, label in expand_labeled(cfg):
            if label not in variants:
                continue
            market = jm.build_market(spec, catalog)
            E, s2p = np.asarray(market.inst.E), np.asarray(market.slot_to_posting)
            epost = posting_graph(E, s2p)
            record = {"window": window, "variant": label, "market": spec.market_id,
                      "seed": spec.seed, "cell": spec.cell, "month": spec.month,
                      "spec": asdict(spec)}
            prefix = f"m{len(markets):05d}_"
            block = {"E": E, "rank_b": market.inst.rank_b, "rank_s": market.inst.rank_s,
                     "s2p": s2p, "lower_wage": market.renters.lower_wage.to_numpy(bool),
                     "score_ranks": normalize_score_ranks(E, s2p, market.score_deterministic)}
            for arm, name in (("random", "uniform_k5"), ("gpt", "llm_k5"), ("score", "transparent_k5")):
                built = je.build_exposure(market, exposures[name], llm)
                if built is None:
                    raise ValueError(f"Missing {arm} menu in {spec.market_id}")
                shown, stats = built
                if arm == "gpt" and any(stats.get(k, 0) for k in ("missing", "refused", "invalid_ids", "dropped_outside_E")):
                    raise ValueError(f"Incomplete GPT cache: {spec.market_id}: {stats}")
                if not np.array_equal(posting_graph(shown, s2p).sum(1), np.minimum(5, epost.sum(1))):
                    raise ValueError(f"Unequal or incomplete menu budget: {spec.market_id}: {arm}")
                block["shown_" + arm] = shown
            recs = llm.table[spec.market_id]["llm_k5"]
            ranked = np.full(epost.shape, -1, dtype=np.int32)
            for b in range(E.shape[0]):
                choices = recs.get((b, 0), {}).get("listings", [])
                if len(choices) != len(set(choices)):
                    raise ValueError("Duplicate cached GPT picks")
                ranked[b, :len(choices)] = choices
            block["gpt_order"] = ranked
            arrays.update({prefix + name: value for name, value in block.items()})
            markets.append(record)
            if len(markets) % 40 == 0:
                print(f"Exported {len(markets)} markets", flush=True)
    expected_count = 400 * len(variants)
    if len(markets) != expected_count:
        raise ValueError(f"Expected {expected_count} markets, exported {len(markets)}")
    files = ["hiddenoptions/jobs/markets.py", "hiddenoptions/jobs/exposure.py",
             "hiddenoptions/matching.py", "experiments_jobs/anchor_access.py", *CONFIGS]
    metadata = {"format_version": FORMAT_VERSION, "data_kind": "frozen_jobs_replay",
                "versions": versions(), "input_hashes": provenance,
                "source_hashes": {p: sha256(ROOT / p) for p in files},
                "markets": markets, "api_calls": 0,
                "contains_raw_identifiers_or_salaries": False}
    save_packet(out, metadata, arrays)
    print(f"Saved {len(markets)} markets to {out}; SHA256 {sha256(out)}")


def drop_factory(arm, record, block, per_person_random=False):
    name = record.get("exposure_names", {"gpt": "llm_k5", "random": "uniform_k5", "score": "transparent_k5"})[arm]
    key = f"{record['market']}|{record['seed']}|anchor|{name}"
    rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))
    if arm == "gpt":
        def order(b, current):
            ranked = [int(p) for p in block["gpt_order"][b] if p >= 0 and p in set(current.tolist())]
            return [p for p in current.tolist() if p not in ranked] + ranked[::-1]
        return order
    if arm == "score":
        return lambda b, current: current[np.argsort(block["score_ranks"][b, current], kind="stable")]
    if per_person_random:
        def order(b, current):
            seed = int(hashlib.sha256(f"{key}|person|{b}".encode()).hexdigest()[:8], 16)
            return current[np.random.default_rng(seed).permutation(len(current))]
        return order
    return lambda b, current: current[rng.permutation(len(current))]


def menu_seed(record, suffix):
    return int(hashlib.sha256(f"{record['market']}|access-v1|{suffix}".encode()).hexdigest()[:8], 16)


def build_menus(record, block):
    from experiments_jobs.coordinated_policies import (
        original_anchor, score_anchor, balanced_menu, cached_rank_relevance)
    E, s2p = block["E"], block["s2p"]
    menus = {arm: block["shown_" + arm] for arm in ("random", "gpt", "score")}
    for arm in ("random", "gpt", "score"):
        menus["anchor_" + arm] = original_anchor(E, menus[arm], s2p, drop_factory(arm, record, block))[0]
    scores = block["score_ranks"]
    cached = [[int(p) for p in row if p >= 0] for row in block["gpt_order"]]
    gpt_scores = cached_rank_relevance(E, menus["gpt"], s2p, cached)
    menus["balanced_gpt"] = balanced_menu(E, menus["gpt"], s2p, gpt_scores, menu_seed(record, "balanced"))
    menus["balanced_score"] = balanced_menu(E, menus["score"], s2p, scores, menu_seed(record, "balanced"))
    menus["score_anchor_gpt"] = score_anchor(E, menus["gpt"], s2p, drop_factory("gpt", record, block),
                                             scores, menu_seed(record, "score-anchor"))[0]
    return menus


def flatten(values, common):
    return [{**common, "group": group, "metric": metric, **counts}
            for group, metrics in values.items() for metric, counts in metrics.items()]


def replay_market(record, block, ties=20):
    from experiments_jobs.anchor_access_metrics import evaluate, transition
    from experiments_jobs.coordinated_policies import permuted_anchor

    n, m = block["E"].shape
    inst = Instance(block["rank_b"], block["rank_s"], np.zeros(n, int), np.zeros(m, int), np.ones(1, int))
    s2p = block["s2p"]
    lower = block["lower_group" if "lower_group" in block else "lower_wage"].astype(bool)
    group_label = "income" if record.get("domain") == "housing" else "wage"
    groups = {"all": np.ones(n, bool), "lower_" + group_label: lower, "higher_" + group_label: ~lower}
    if not np.array_equal(inst.E, block["E"]):
        raise ValueError("Packet feasibility differs from its rank-array support")
    full = deferred_acceptance(inst, inst.E)[0]
    menus = build_menus(record, block)
    quotas = posting_graph(block["shown_gpt"], s2p).sum(1)
    rows, checks = [], []
    common = {k: record[k] for k in ("window", "variant", "market", "seed", "cell", "month")}

    def score(menus_now, tie):
        matches = {}
        for arm, shown in menus_now.items():
            if np.any(shown & ~inst.E) or not np.array_equal(posting_graph(shown, s2p).sum(1), quotas):
                raise AssertionError(f"Feasibility/budget violation: {common}: {arm}")
            match = deferred_acceptance(inst, shown)[0]
            matches[arm] = match
            rows.extend(flatten(evaluate(inst, shown, match, full, groups, s2p),
                                {**common, "arm": arm, "baseline": "", "tie": tie}))
            original = "random" if "random" in arm else ("score" if arm in ("score", "anchor_score", "balanced_score") else "gpt")
            checks.append({**common, "arm": arm, "tie": tie, "n": n,
                           "placed": int((match >= 0).sum()), "capacity": capacity(shown),
                           "lists_changed": int(np.any(shown != block["shown_" + original], axis=1).sum()),
                           "full_capacity": capacity(inst.E), "menu_budget_kept": True})
        return matches

    matches = score(menus, -1)
    for arm, base in CONTRASTS:
        if arm in matches and base in matches:
            rows.extend(flatten(transition(inst, matches[base], matches[arm], groups, s2p),
                                {**common, "arm": arm, "baseline": base, "tie": -1}))
    for tie in range(ties):
        permuted = {}
        for arm in ("gpt", "random"):
            permuted["anchor_" + arm + "_ties"] = permuted_anchor(
                inst.E, menus[arm], s2p, drop_factory(arm, record, block, per_person_random=True),
                menu_seed(record, f"tie-{tie}"))[0]
        varied = score(permuted, tie)
        for arm, base in CONTRASTS:
            if arm in varied and (base in varied or base in matches):
                baseline = varied[base] if base in varied else matches[base]
                rows.extend(flatten(transition(inst, baseline, varied[arm], groups, s2p),
                                    {**common, "arm": arm, "baseline": base, "tie": tie}))
    return rows, checks


def evaluate_packet(packet, out, shard, shards, ties, allow_synthetic=False):
    start = time.monotonic()
    with np.load(packet, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if metadata["format_version"] != FORMAT_VERSION:
            raise ValueError("Unsupported packet version")
        if metadata["data_kind"] not in ("frozen_jobs_replay", "frozen_rental_replay") and not allow_synthetic:
            raise ValueError("Synthetic validation packets are not empirical jobs inputs")
        rows, checks = [], []
        selected = [i for i, r in enumerate(metadata["markets"]) if r["seed"] % shards == shard]
        for j, i in enumerate(selected, 1):
            record = metadata["markets"][i]
            prefix = f"m{i:05d}_"
            block = {key[len(prefix):]: data[key] for key in data.files if key.startswith(prefix)}
            values, audit = replay_market(record, block, ties=ties)
            rows.extend(values)
            checks.extend(audit)
            if j % 20 == 0 or j == len(selected):
                print(f"Shard {shard}/{shards}: {j}/{len(selected)} markets", flush=True)
    if not selected:
        raise ValueError("Empty shard")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"metrics-{shard:02d}.csv.gz", index=False)
    pd.DataFrame(checks).to_csv(out / f"checks-{shard:02d}.csv.gz", index=False)
    manifest = {"data_kind": metadata["data_kind"], "packet_sha256": sha256(packet),
                "shard": shard, "shards": shards, "tie_draws": ties, "markets": len(selected),
                "versions": versions(), "elapsed_seconds": time.monotonic() - start,
                "panel_manifest": {k: metadata[k] for k in ("planned_markets", "skipped_markets", "markets", "source_hashes", "input_hashes") if k in metadata},
                "source_hashes": {f: sha256(ROOT / f) for f in (
                    "experiments_jobs/anchor_access.py", "experiments_jobs/coordinated_policies.py",
                    "experiments_jobs/anchor_access_metrics.py", "experiments_jobs/capacity_anchor.py",
                    "hiddenoptions/matching.py")}}
    (out / f"run-{shard:02d}.json").write_text(json.dumps(manifest, indent=2) + "\n")


def synthetic_packet(out, count=8):
    """Invented markets exclusively for end-to-end software validation."""
    from experiments_jobs.coordinated_policies import normalize_score_ranks
    arrays, records = {}, []
    for i in range(count):
        rng, n = np.random.default_rng(9000 + i), 6
        E = rng.random((n, n)) < .7
        s2p = np.arange(n)
        rb = np.where(E, rng.random((n, n)).argsort(axis=1).argsort(axis=1), NOT_EDGE)
        rs = np.where(E, rng.random((n, n)).argsort(axis=0).argsort(axis=0), NOT_EDGE)
        scores = rng.normal(size=(n, n))
        block = {"E": E, "rank_b": rb, "rank_s": rs, "s2p": s2p,
                 "lower_wage": np.arange(n) < n // 2,
                 "score_ranks": normalize_score_ranks(E, s2p, scores)}
        order = np.full((n, n), -1, int)
        for arm in ("gpt", "random", "score"):
            shown = np.zeros_like(E)
            for b in range(n):
                choices = np.flatnonzero(E[b])
                choices = choices[np.argsort(-scores[b, choices])] if arm == "score" else rng.permutation(choices)
                chosen = choices[:min(3, len(choices))]
                shown[b, chosen] = True
                if arm == "gpt":
                    order[b, :len(chosen)] = chosen
            block["shown_" + arm] = shown
        block["gpt_order"] = order
        arrays.update({f"m{i:05d}_" + k: v for k, v in block.items()})
        records.append({"window": "synthetic_validation", "variant": "invented", "market": f"invented-{i}",
                        "seed": i, "cell": "synthetic", "month": "none"})
    save_packet(out, {"format_version": FORMAT_VERSION, "data_kind": "synthetic_validation",
                      "versions": versions(), "markets": records, "api_calls": 0}, arrays)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("preflight", help="Report required files; never starts model calls")
    p.add_argument("--data-root", type=Path, default=ROOT)
    p.add_argument("--out", type=Path)
    p = sub.add_parser("export", help="Run in the original authorized jobs data environment")
    p.add_argument("--data-root", type=Path, default=ROOT)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--variants", nargs="+", choices=["primary", "natural"], default=["primary", "natural"])
    p = sub.add_parser("evaluate", help="Run frozen algorithms on an anonymous replay packet")
    p.add_argument("--packet", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--ties", type=int, default=20)
    p.add_argument("--allow-synthetic", action="store_true")
    p = sub.add_parser("synthetic-packet", help="Software validation only; not a jobs result")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--markets", type=int, default=8)
    args = ap.parse_args()
    if args.command == "preflight":
        result = preflight(args.data_root.resolve())
        text = json.dumps(result, indent=2) + "\n"
        print(text)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text)
        raise SystemExit(0 if result["status"] == "ready" else 2)
    if args.command == "export":
        export_panel(args.data_root.resolve(), args.out, args.variants)
    elif args.command == "evaluate":
        if not 0 <= args.shard < args.shards or args.ties < 0:
            raise ValueError("Invalid shard or tie count")
        evaluate_packet(args.packet, args.out, args.shard, args.shards, args.ties, args.allow_synthetic)
    elif args.command == "synthetic-packet":
        synthetic_packet(args.out, args.markets)


if __name__ == "__main__":
    main()
