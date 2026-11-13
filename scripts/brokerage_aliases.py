"""Propose the frozen brokerage alias map from the development months (protocol section 3, "Brokerages").

Step 1 (this script, default): normalize every development `source_group` (hiddenoptions.firms.normalize_name: case,
accents, punctuation, a leading "the", trailing LLC/Inc/Corp/Co/Ltd/LP and Realty/Real Estate/NYC/NY variants) and merge
names with the same key. Then write
    data/reference/brokerage_aliases_review.csv   the groups to inspect: the largest raw groups covering >= 80% of
                                                   development listings, plus the groups of a fixed seeded sample of 50
                                                   records from the rest. Each row has the proposed canonical firm, a
                                                   heuristic kind (brokerage / feed / platform / unresolved) with its
                                                   reason, loose "possible merges" to check, and blank confirmed_* columns.
    data/reference/brokerage_aliases.json          the map markets use with MarketSpec(firm_aliases=True), "reviewed": false
Step 2 (a person): fill confirmed_kind (brokerage, feed, platform or unresolved) for every sheet row, and confirmed_canonical
where a row should join another firm (its canonical id or name) or leave a proposed merge (write the row's own name).
Step 3: python scripts/brokerage_aliases.py --apply-review   rewrites the JSON from the same data plus the confirmed
columns. "reviewed" becomes true only when every sheet row has a confirmed_kind. Groups outside the sheet keep their
heuristic kind (kind_source "heuristic").

Owner listings (source_type OWNER) are excluded: each is already its own private firm.

Usage
    python scripts/brokerage_aliases.py                 # propose; refuses to overwrite a sheet that has confirmations
    python scripts/brokerage_aliases.py --apply-review
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hiddenoptions.data import DEV_MONTHS, load_listings  # noqa: E402
from hiddenoptions.firms import (ALIAS_MAP, KINDS, LEGAL_SUFFIXES, LOOSE_SUFFIXES, REALTY_SUFFIXES, _fold, _strip,  # noqa: E402
                                 canonical_id, loose_key, normalize_name)

SHEET = ROOT / "data" / "reference" / "brokerage_aliases_review.csv"
COVERAGE = 0.80
SAMPLE_RECORDS = 50
SAMPLE_SEED = 2880
PLATFORM = re.compile(r"blueground|furnish|zillow|streeteasy|apartments ?com|zumper|renthop|hotpads|leasebreak|june homes|"
                      r"sonder|airbnb|landing|corporate housing", re.I)
UNRESOLVED = re.compile(r"^(|unknown|n ?a|none|null|owner|owners|landlord|management|private|agent|broker|brokerage|"
                        r"realty|real estate|listing|listings|rentals?|test|[0-9 ]+)$", re.I)


def heuristic_kind(name: str, source_types: set[str]) -> tuple[str, str]:
    key = _strip(_fold(name), LEGAL_SUFFIXES)
    if UNRESOLVED.match(key) or len(key) < 2:
        return "unresolved", "missing or generic name"
    if PLATFORM.search(name):
        return "platform", "name matches a rental platform / furnished-operator keyword"
    if source_types == {"FEED"}:
        return "feed", "listed only through FirstMover FEED (manager or landlord feed)"
    if "PARTNER" in source_types:
        return "brokerage", "listed as a PARTNER (brokerage) source"
    return "unresolved", f"source types {sorted(source_types)}"


def development_listings() -> pd.DataFrame:
    d = pd.concat([load_listings(m) for m in DEV_MONTHS], ignore_index=True)
    return d[d.source_type != "OWNER"].assign(source_group=lambda x: x.source_group.fillna(""))


def propose(d: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    total = len(d)
    groups = (d.groupby("source_group")
              .agg(dev_listings=("listing_uid", "size"), source_types=("source_type", lambda s: ",".join(sorted(set(s)))))
              .reset_index().sort_values(["dev_listings", "source_group"], ascending=[False, True]).reset_index(drop=True))
    groups["share"] = groups.dev_listings / total
    groups["cum_share"] = groups.share.cumsum()
    groups["normalized_key"] = groups.source_group.map(normalize_name)
    groups["proposed_canonical_id"] = [canonical_id(k) if k else "" for k in groups.normalized_key]

    # Canonical firms: one per normalized key; display name = the largest raw name.
    firms: dict[str, dict] = {}
    for g in groups.itertuples():
        if not g.proposed_canonical_id:
            continue
        f = firms.setdefault(g.proposed_canonical_id, {"name": g.source_group, "raw_names": [], "dev_listings": 0, "types": set()})
        f["raw_names"].append(g.source_group)
        f["dev_listings"] += int(g.dev_listings)
        f["types"] |= set(g.source_types.split(","))
    for cid, f in firms.items():
        f["kind"], f["kind_reason"] = heuristic_kind(f["name"], f["types"])
        f["kind_source"] = "heuristic"
    groups["proposed_canonical_name"] = groups.proposed_canonical_id.map(lambda c: firms[c]["name"] if c else "")
    groups["merged_with"] = [" | ".join(n for n in firms[c]["raw_names"] if n != s) if c else ""
                             for s, c in zip(groups.source_group, groups.proposed_canonical_id)]
    groups["heuristic_kind"] = [firms[c]["kind"] if c else "unresolved" for c in groups.proposed_canonical_id]
    groups["kind_reason"] = [firms[c]["kind_reason"] if c else "missing source_group" for c in groups.proposed_canonical_id]
    loose = defaultdict(set)
    for s, c in zip(groups.source_group, groups.proposed_canonical_id):
        if c:
            loose[loose_key(s)].add(c)
    groups["possible_merges"] = [" | ".join(sorted(firms[o]["name"] for o in loose[loose_key(s)] if o != c)) if c else ""
                                 for s, c in zip(groups.source_group, groups.proposed_canonical_id)]

    # The review set: largest raw groups to >= 80% coverage, plus the groups of 50 seeded records from the rest.
    n_top = int(np.searchsorted(groups.cum_share.to_numpy(), COVERAGE) + 1)
    top = set(groups.source_group[:n_top])
    rest = d[~d.source_group.isin(top)].sort_values("listing_uid")
    picks = rest.iloc[np.random.default_rng(SAMPLE_SEED).choice(len(rest), size=min(SAMPLE_RECORDS, len(rest)), replace=False)]
    sampled = picks.source_group.value_counts()
    groups["selection"] = np.where(groups.source_group.isin(top), "top_coverage",
                                   np.where(groups.source_group.isin(sampled.index), "seeded_sample", ""))
    groups["sampled_records"] = groups.source_group.map(sampled).fillna(0).astype(int)
    sheet = groups[groups.selection != ""].copy()
    sheet["confirmed_kind"] = ""
    sheet["confirmed_canonical"] = ""
    sheet["reviewer_notes"] = ""
    sheet = sheet[["selection", "source_group", "source_types", "dev_listings", "share", "cum_share", "sampled_records",
                   "normalized_key", "proposed_canonical_id", "proposed_canonical_name", "merged_with", "heuristic_kind",
                   "kind_reason", "possible_merges", "confirmed_kind", "confirmed_canonical", "reviewer_notes"]]

    meta = {"total_listings": total, "raw_groups": len(groups), "top_groups": n_top,
            "top_share": float(groups.share[:n_top].sum()), "sample_records": len(picks), "sample_groups": int(len(sampled)),
            "sample_listings_share": float(groups.share[groups.selection == "seeded_sample"].sum()),
            "missing_source_group_listings": int((d.source_group == "").sum())}
    return {"firms": firms, "groups": groups, "meta": meta}, sheet


def build_map(state: dict, reviewed_rows: pd.DataFrame | None = None) -> dict:
    firms, groups, meta = state["firms"], state["groups"], state["meta"]
    aliases = {s: c for s, c in zip(groups.source_group, groups.proposed_canonical_id) if c}
    keys = {k: c for k, c in zip(groups.normalized_key, groups.proposed_canonical_id) if c}
    reviewed = False
    if reviewed_rows is not None:
        by_name = {f["name"].lower(): cid for cid, f in firms.items()}
        for r in reviewed_rows.itertuples():
            target = str(r.confirmed_canonical).strip()
            cid = aliases.get(r.source_group)
            if target:
                new = target if target in firms else by_name.get(target.lower(), canonical_id(normalize_name(target)) or target)
                if new not in firms:
                    firms[new] = {"name": target, "raw_names": [], "dev_listings": 0, "types": set(), "kind": "unresolved",
                                  "kind_reason": "created in review", "kind_source": "reviewed"}
                if cid and r.source_group in firms.get(cid, {}).get("raw_names", []):
                    firms[cid]["raw_names"].remove(r.source_group)
                    firms[cid]["dev_listings"] -= int(r.dev_listings)
                firms[new]["raw_names"].append(r.source_group)
                firms[new]["dev_listings"] += int(r.dev_listings)
                aliases[r.source_group] = cid = new
            kind = str(r.confirmed_kind).strip().lower()
            if kind and cid:
                if kind not in KINDS:
                    raise SystemExit(f"confirmed_kind {kind!r} for {r.source_group!r} is not one of {KINDS}")
                firms[cid].update(kind=kind, kind_source="reviewed", kind_reason="confirmed in review")
        reviewed = bool(len(reviewed_rows)) and all(str(k).strip() for k in reviewed_rows.confirmed_kind)
    firms = {c: f for c, f in firms.items() if f["raw_names"] or f["kind_source"] == "reviewed"}
    kinds_listings = defaultdict(int)
    for f in firms.values():
        kinds_listings[f["kind"]] += f["dev_listings"]
    kinds_listings["unresolved"] += meta["missing_source_group_listings"]
    # raw_names only where several raw names merge; `aliases` holds every raw name -> firm.
    out_firms = {c: {"name": f["name"], "kind": f["kind"], "kind_source": f["kind_source"], "dev_listings": f["dev_listings"],
                     **({"raw_names": sorted(f["raw_names"])} if len(f["raw_names"]) > 1 else {})} for c, f in sorted(firms.items())}
    merged = {c: f for c, f in out_firms.items() if len(f.get("raw_names", [])) > 1}
    return {
        "version": 1,
        "reviewed": reviewed,
        "note": "Proposed by scripts/brokerage_aliases.py. Used only by MarketSpec(firm_aliases=True). Owner listings keep "
                "private firms; kinds in singleton_kinds, missing and unknown names become singleton firms.",
        "months": list(DEV_MONTHS),
        "normalization": {"legal_suffixes": list(LEGAL_SUFFIXES), "realty_suffixes": list(REALTY_SUFFIXES),
                          "loose_suffixes_review_only": list(LOOSE_SUFFIXES),
                          "rule": "NFKD to ASCII, lower case, & -> and, drop . and apostrophes, other punctuation -> space, "
                                  "drop a leading 'the', then strip trailing suffixes repeatedly"},
        "singleton_kinds": ["unresolved"],
        "review": {"coverage_target": COVERAGE, "sample_records": SAMPLE_RECORDS, "sample_seed": SAMPLE_SEED,
                   "sheet": str(SHEET.relative_to(ROOT))},
        "stats": {**meta, "canonical_firms": len(out_firms), "merged_firms": len(merged),
                  "raw_names_merged": sum(len(f["raw_names"]) for f in merged.values()),
                  "listings_by_kind": dict(sorted(kinds_listings.items()))},
        "firms": out_firms,
        "aliases": dict(sorted(aliases.items())),
        "keys": dict(sorted(keys.items())),
    }


def dump_map(amap: dict) -> str:
    """JSON with one firm or alias per line: reviewable diffs without the size of a fully indented file."""
    j = lambda v: json.dumps(v, ensure_ascii=False)
    parts = []
    for k, v in amap.items():
        if isinstance(v, dict) and len(v) > 20:
            parts.append(f" {j(k)}: {{\n" + ",\n".join(f"  {j(a)}: {j(b)}" for a, b in v.items()) + "\n }")
        else:
            parts.append(f" {j(k)}: {j(v)}")
    return "{\n" + ",\n".join(parts) + "\n}\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply-review", action="store_true", help="rebuild the JSON with the sheet's confirmed_* columns")
    ap.add_argument("--force", action="store_true", help="overwrite a review sheet that already has confirmations")
    args = ap.parse_args()
    state, sheet = propose(development_listings())
    if args.apply_review:
        filled = pd.read_csv(SHEET, dtype=str, keep_default_na=False)
        filled["dev_listings"] = filled.dev_listings.astype(int)
        amap = build_map(state, filled)
    else:
        if SHEET.exists() and not args.force:
            old = pd.read_csv(SHEET, dtype=str, keep_default_na=False)
            if (old.get("confirmed_kind", pd.Series(dtype=str)).str.strip() != "").any():
                raise SystemExit(f"{SHEET} has confirmations; use --apply-review, or --force to discard them")
        SHEET.parent.mkdir(parents=True, exist_ok=True)
        sheet.to_csv(SHEET, index=False, float_format="%.5f")
        amap = build_map(state)
    ALIAS_MAP.write_text(dump_map(amap))
    s = amap["stats"]
    print(f"{s['total_listings']:,} development non-owner listings, {s['raw_groups']:,} raw source groups -> "
          f"{s['canonical_firms']:,} canonical firms ({s['merged_firms']} merges covering {s['raw_names_merged']} raw names)")
    print(f"review sheet: top {s['top_groups']} groups cover {s['top_share']:.1%}; seeded sample of {s['sample_records']} records "
          f"adds {s['sample_groups']} groups; listings by kind {s['listings_by_kind']}")
    print(f"wrote {ALIAS_MAP.relative_to(ROOT)} (reviewed={amap['reviewed']})" + ("" if args.apply_review else f" and {SHEET.relative_to(ROOT)}"))


if __name__ == "__main__":
    main()
