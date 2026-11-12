"""E0: which feasible pairs each renter is shown (protocol section 5).

Every policy shows each renter the same NUMBER of listings, so policies differ only in WHICH listings reach the renter:
    k  top-k: min(k, d_b) listings (primary k = 5)
    q  proportional: ceil(q * d_b) listings (second budget, q = 0.4)
Renters with d_b <= k see everything; renters with d_b = 0 see nothing.

Policies
    uniform (alias random)            k feasible listings uniformly at random: the baseline
    transparent                        top k by the renter's observable score (latent shocks removed)
    expensive_first (alias rent_max)   the k highest asking rents
    llm                                a fixed model's saved picks (Claude, or an open model run on GPUs), with the fallback
                                       rule below
    full                               every feasible pair (full visibility benchmark)

Fallback for model replies (protocol section 5): keep distinct valid IDs in reply order up to k, then fill any shortfall from
one frozen seeded uniform order over the remaining candidates. A refusal, parse failure or missing reply gets the complete
uniform fallback. The deployed policy is the model WITH this fallback; its rates are reported by group.
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np

ALIASES = {"random": "uniform", "rent_max": "expensive_first"}


def show_count(n_feasible: np.ndarray, k: int | None, q: float | None) -> np.ndarray:
    if (k is None) == (q is None):
        raise ValueError("set exactly one of k and q")
    if k is not None:
        return np.minimum(n_feasible, k)
    return np.minimum(n_feasible, np.ceil(q * n_feasible).astype(int))


def _exposure_rng(market, salt: str) -> np.random.Generator:
    return np.random.default_rng([market.spec.seed, zlib.crc32(f"{market.spec.market_id}|exposure|{salt}".encode())])


def fallback_order(market, b: int) -> np.ndarray:
    """One frozen seeded uniform order over renter b's feasible listings, shared by the uniform policy and the model fallback."""
    options = np.nonzero(market.inst.E[b])[0]
    return options[_exposure_rng(market, f"uniform-order|{b}").permutation(len(options))]


def uniform_exposure(market, k=None, q=None) -> np.ndarray:
    E = market.inst.E
    e0 = np.zeros_like(E)
    for b, c in enumerate(show_count(E.sum(1), k, q)):
        if c:
            e0[b, fallback_order(market, b)[: int(c)]] = True
    return e0


random_exposure = uniform_exposure  # older name


def transparent_exposure(market, k=None, q=None) -> np.ndarray:
    E, u = market.inst.E, market.u_observable
    e0 = np.zeros_like(E)
    for b, c in enumerate(show_count(E.sum(1), k, q)):
        options = np.nonzero(E[b])[0]
        e0[b, options[np.argsort(-u[b, options], kind="stable")[: int(c)]]] = True
    return e0


def expensive_first_exposure(market, k=None, q=None) -> np.ndarray:
    E, rent = market.inst.E, market.listings.rent_asking.to_numpy(float) if "rent_asking" in market.listings else market.listings.rent.to_numpy(float)
    e0 = np.zeros_like(E)
    for b, c in enumerate(show_count(E.sum(1), k, q)):
        options = np.nonzero(E[b])[0]
        e0[b, options[np.argsort(-rent[options], kind="stable")[: int(c)]]] = True
    return e0


rent_max_exposure = expensive_first_exposure  # older name


class LlmRecommendations:
    """Saved recommender output: {market_id: {exposure: {(renter_index, rep): row}}}. Rows come from hiddenoptions.llm."""

    def __init__(self, path: str | Path):
        self.table: dict = {}
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            self.table.setdefault(r["market_id"], {}).setdefault(r["exposure"], {})[(r["renter"], r.get("rep", 0))] = r

    def exposure(self, market, name: str, rep: int = 0, k: int | None = 5, q: float | None = None):
        """E0 from saved replies with the fallback rule, or None if this market has no saved replies for `name`."""
        recs = self.table.get(market.spec.market_id, {}).get(name)
        if recs is None:
            return None
        E = market.inst.E
        e0 = np.zeros_like(E)
        counts = show_count(E.sum(1), k, q)
        status = np.full(E.shape[0], "no_options", dtype=object)
        stats = {"renters_with_options": int((E.sum(1) > 0).sum()), "missing": 0, "refused": 0, "unparseable": 0,
                 "invalid_ids": 0, "dropped_outside_E": 0, "complete_valid": 0, "partial_fill": 0, "full_fallback": 0, "whole_set": 0}
        for b in np.nonzero(E.sum(1) > 0)[0]:
            c = int(counts[b])
            if c == E[b].sum():  # k_b = d_b: the whole set is returned directly, no model call
                e0[b, E[b]] = True
                status[b] = "whole_set"
                stats["whole_set"] += 1
                continue
            r = recs.get((int(b), rep))
            picks = []
            if r is None:
                stats["missing"] += 1
            else:
                stats["refused"] += int(r.get("status") == "refused")
                stats["unparseable"] += int(r.get("status") == "unparseable")
                stats["invalid_ids"] += int(r.get("invalid_ids", 0))
                for s in r.get("listings", []):
                    if E[b, s] and s not in picks:
                        picks.append(int(s))
                    elif not E[b, s]:
                        stats["dropped_outside_E"] += 1
                picks = picks[:c]
            n_valid = len(picks)
            for s in fallback_order(market, b):
                if len(picks) >= c:
                    break
                if s not in picks:
                    picks.append(int(s))
            e0[b, picks] = True
            status[b] = "complete_valid" if n_valid == c else ("full_fallback" if n_valid == 0 else "partial_fill")
            stats[status[b]] += 1
        stats["status_by_renter"] = status.tolist()
        return e0, stats


POLICIES = {"uniform": uniform_exposure, "transparent": transparent_exposure, "expensive_first": expensive_first_exposure}


def build_exposure(market, cfg: dict, llm: LlmRecommendations | None = None):
    policy = ALIASES.get(cfg["policy"], cfg["policy"])
    k, q = cfg.get("k"), cfg.get("q")
    if policy == "full":
        return market.inst.E.copy(), {}
    if policy == "llm":
        return None if llm is None else llm.exposure(market, cfg["name"], cfg.get("rep", 0), k, q)
    if policy not in POLICIES:
        raise ValueError(f"unknown exposure policy {cfg['policy']}")
    return POLICIES[policy](market, k, q), {}
