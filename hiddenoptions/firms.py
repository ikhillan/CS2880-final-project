"""Listing-side firms from FirstMover's `source_group`, through a frozen alias map (protocol section 3, "Brokerages").

The map lives in data/reference/brokerage_aliases.json and is written by scripts/brokerage_aliases.py. It is used only by
markets built with MarketSpec(firm_aliases=True); by default firms stay as the cleaned files give them
(`firm_id` = slug of source_group, owner listings private).

With the map:
  owner listings (source_type OWNER)   keep their private firm "owner:<unit_id>"
  source_group with a canonical firm   "firm:<canonical id>" (alias variants share one firm)
  kind in the map's singleton_kinds    "single:<unit_id>": a separate singleton firm (the protocol's rule for unresolved
  (default: unresolved), a missing     sources), so unrelated listings are never pooled into one fake firm
  source_group, or a name the map
  does not know
Names the map has not seen (e.g. a new brokerage in a held-out month) are looked up by their normalized key, so spelling
variants of a known firm still join it; anything else is unresolved.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ALIAS_MAP = ROOT / "data" / "reference" / "brokerage_aliases.json"
# Stripped from the end of a name, repeatedly, in any order (after case folding and punctuation removal).
LEGAL_SUFFIXES = ("l l c", "llc", "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "lp", "llp",
                  "pllc", "pc")
REALTY_SUFFIXES = ("real estate group", "realty group", "real estate", "realty", "nyc", "ny", "new york")
# Looser descriptors: never merged automatically; scripts/brokerage_aliases.py only lists names that would merge under them
# as "possible merges" for the reviewer (they join unrelated firms such as "Brooklyn Group" and "Brooklyn Properties").
LOOSE_SUFFIXES = ("group", "properties", "property group", "management", "management group", "residential", "international",
                  "associates", "partners", "living", "homes")
KINDS = ("brokerage", "feed", "platform", "unresolved")


def _fold(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower().replace("&", " and ")
    s = re.sub(r"[.'’]", "", s)  # "R.E." -> "re", "SERHANT." -> "serhant", "Nooklyn's" -> "nooklyns"
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s[4:] if s.startswith("the ") else s


def _strip(s: str, suffixes) -> str:
    """Strip trailing suffixes repeatedly, but never down to a stub: 'R New York' stays 'r new york' and
    '346 Realty Group LLC' stays '346 realty group' (a remainder needs 3+ characters and a letter)."""
    changed = True
    while changed:
        changed = False
        for suf in suffixes:
            if s.endswith(" " + suf):
                rest = s[: -len(suf) - 1].strip()
                if len(rest) >= 3 and re.search(r"[a-z]", rest):
                    s, changed = rest, True
    return s


def normalize_name(name) -> str:
    """Deterministic key for merging spelling variants: case, accents, punctuation, a leading 'the', and trailing legal
    and realty suffixes. 'Skyhigh Realty NYC LLC' -> 'skyhigh'; 'FIND Real Estate' and 'The Find NY' -> 'find'.
    Missing or blank names give ''."""
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    return _strip(_fold(name), LEGAL_SUFFIXES + REALTY_SUFFIXES)


def loose_key(name) -> str:
    """normalize_name plus the loose descriptors: used only to suggest merges for human review."""
    return _strip(normalize_name(name), LEGAL_SUFFIXES + REALTY_SUFFIXES + LOOSE_SUFFIXES)


def canonical_id(key: str) -> str:
    return re.sub(r"\s+", "-", key.strip())


class AliasMap:
    def __init__(self, data: dict):
        self.data = data
        encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        self.sha256 = hashlib.sha256(encoded).hexdigest()
        self.aliases: dict[str, str] = data["aliases"]
        self.keys: dict[str, str] = data["keys"]
        self.firms: dict[str, dict] = data["firms"]
        self.singleton_kinds = set(data.get("singleton_kinds", ["unresolved"]))
        self.reviewed = bool(data.get("reviewed", False))

    @classmethod
    def load(cls, path: str | Path = ALIAS_MAP) -> "AliasMap":
        return cls(json.loads(Path(path).read_text()))

    def canonical(self, source_group) -> str | None:
        if source_group is None or (isinstance(source_group, float) and np.isnan(source_group)):
            return None
        cid = self.aliases.get(str(source_group))
        return cid if cid is not None else self.keys.get(normalize_name(source_group))

    def kind(self, cid: str | None) -> str:
        return "unresolved" if cid is None or cid not in self.firms else self.firms[cid]["kind"]

    def firm_ids(self, listings: pd.DataFrame) -> np.ndarray:
        """Firm label per listing row (see the module docstring)."""
        groups = listings.source_group.astype(object).where(listings.source_group.notna(), None)
        resolved = {}
        for g in pd.unique(groups):
            cid = self.canonical(g)
            resolved[g] = None if self.kind(cid) in self.singleton_kinds else f"firm:{cid}"
        out = np.empty(len(listings), dtype=object)
        for i, (st, g, uid, fid) in enumerate(zip(listings.source_type, groups, listings.unit_id, listings.firm_id)):
            if st == "OWNER":
                out[i] = fid  # already the private "owner:<unit_id>"
            else:
                out[i] = resolved[g] if resolved[g] is not None else f"single:{uid}"
        return out


@lru_cache(maxsize=4)
def load_alias_map(path: str = str(ALIAS_MAP)) -> AliasMap:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found: run scripts/brokerage_aliases.py (firm_aliases=True needs the alias map)")
    return AliasMap.load(p)
