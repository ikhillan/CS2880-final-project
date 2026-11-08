"""E0: which feasible pairs each renter is shown.

Every policy shows each renter the same NUMBER of listings, so policies differ only in WHICH listings reach the renter:
    k  top-k, as a real recommender works: min(k, |E_b|) listings
    q  proportional: ceil(q * |E_b|) listings

Why both: sampled markets shrink feasible sets, so a lower-income renter may have 3-8 options against a higher-income renter's
45. A fixed k then shows the lower-income renter nearly everything, and comparisons of "share hidden" mostly reflect list
length. q gives each renter the same coverage. Report both.

Policies
    random    uniform among E_b: the baseline
    rent_max  the most expensive affordable listings: a no-API proxy for a recommender steered by commission
    llm       the fixed AI recommender, read from saved responses (see hiddenoptions/llm.py)
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np

from .markets import Market


def show_count(n_feasible: np.ndarray, k: int | None, q: float | None) -> np.ndarray:
    if (k is None) == (q is None):
        raise ValueError("set exactly one of k and q")
    if k is not None:
        return np.minimum(n_feasible, k)
    return np.minimum(n_feasible, np.ceil(q * n_feasible).astype(int))


def _exposure_rng(market: Market, salt: str) -> np.random.Generator:
    return np.random.default_rng([market.spec.seed, zlib.crc32(f"{market.spec.market_id}|exposure|{salt}".encode())])


def random_exposure(market: Market, k=None, q=None) -> np.ndarray:
    E = market.inst.E
    rng, e0 = _exposure_rng(market, f"random|{k}|{q}"), np.zeros_like(E)
    counts = show_count(E.sum(1), k, q)
    for b, c in enumerate(counts):
        options = np.nonzero(E[b])[0]
        if c:
            e0[b, rng.choice(options, size=int(c), replace=False)] = True
    return e0


def rent_max_exposure(market: Market, k=None, q=None) -> np.ndarray:
    E, rent = market.inst.E, market.listings.rent.to_numpy(float)
    e0, counts = np.zeros_like(E), show_count(E.sum(1), k, q)
    for b, c in enumerate(counts):
        options = np.nonzero(E[b])[0]
        e0[b, options[np.argsort(-rent[options], kind="stable")[: int(c)]]] = True
    return e0


class LlmRecommendations:
    """Saved recommender output: {market_id: {exposure: {renter_index: [listing_index, ...]}}}."""

    def __init__(self, path: str | Path):
        self.table: dict = {}
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            self.table.setdefault(r["market_id"], {}).setdefault(r["exposure"], {})[(r["renter"], r.get("rep", 0))] = r

    def exposure(self, market: Market, name: str, rep: int = 0) -> tuple[np.ndarray, dict] | None:
        """E0 from saved output, or None if this market has none. Anything outside E is dropped and counted."""
        recs = self.table.get(market.spec.market_id, {}).get(name)
        if recs is None:
            return None
        E = market.inst.E
        e0 = np.zeros_like(E)
        stats = {"renters_with_options": int((E.sum(1) > 0).sum()), "missing": 0, "refused": 0, "invalid_ids": 0, "dropped_outside_E": 0}
        for b in np.nonzero(E.sum(1) > 0)[0]:
            r = recs.get((int(b), rep))
            if r is None:
                stats["missing"] += 1
                continue
            stats["refused"] += int(r.get("status") == "refused")
            stats["invalid_ids"] += int(r.get("invalid_ids", 0))
            for s in r.get("listings", []):
                if E[b, s]:
                    e0[b, s] = True
                else:
                    stats["dropped_outside_E"] += 1
        return e0, stats


def build_exposure(market: Market, cfg: dict, llm: LlmRecommendations | None = None) -> tuple[np.ndarray, dict] | None:
    policy, k, q = cfg["policy"], cfg.get("k"), cfg.get("q")
    if policy == "random":
        return random_exposure(market, k, q), {}
    if policy == "rent_max":
        return rent_max_exposure(market, k, q), {}
    if policy == "llm":
        return None if llm is None else llm.exposure(market, cfg["name"], cfg.get("rep", 0))
    raise ValueError(f"unknown exposure policy {policy}")
