"""E0 for job markets: which feasible postings each seeker is shown.

Same contract as hiddenoptions/exposure.py: every policy shows a seeker min(k, |E_b|) postings (or ceil(q |E_b|)), so policies
differ only in WHICH postings reach the seeker. Slots of one posting are shown together: a policy picks postings, then lights
every slot of each pick, so exposure never splits a posting's capacity.

Policies
    random               uniform among feasible postings: the baseline (housing random_exposure, applied at posting level)
    salary_max           highest-paying feasible postings first: the mirror of rent_max
    salary_min           lowest-paying first: brackets salary-based selection from the other side
    under_leveled        lowest seniority first: showing people jobs below their level
    transparent          the deterministic part of the seeker's own score: the honest-recommender baseline
    llm                  saved recommender output (hiddenoptions/jobs/llm.py), as in housing
"""
from __future__ import annotations

import numpy as np

from hiddenoptions.exposure import LlmRecommendations, _exposure_rng, show_count
from hiddenoptions.jobs.markets import JobMarket


def _posting_level(market: JobMarket) -> tuple[np.ndarray, np.ndarray]:
    """E over (seeker, posting) and the slot -> posting map."""
    s2p = market.slot_to_posting
    n_post = int(s2p.max()) + 1
    E_post = np.zeros((market.inst.n_b, n_post), bool)
    for slot, post in enumerate(s2p):
        E_post[:, post] |= market.inst.E[:, slot]
    return E_post, s2p


def _lift(market: JobMarket, picks_post: np.ndarray) -> np.ndarray:
    """(seeker, posting) picks -> (seeker, slot) e0, restricted to E."""
    return picks_post[:, market.slot_to_posting] & market.inst.E


def _by_score(market: JobMarket, score_post: np.ndarray, k, q, salt: str) -> np.ndarray:
    """Show each seeker its top-count postings by `score_post` (higher = shown first). Ties broken by a seeded shuffle."""
    E_post, _ = _posting_level(market)
    rng = _exposure_rng(market, f"{salt}|{k}|{q}")
    counts = show_count(E_post.sum(1), k, q)
    picks = np.zeros_like(E_post)
    for b, c in enumerate(counts):
        options = np.nonzero(E_post[b])[0]
        if not c:
            continue
        options = options[rng.permutation(len(options))]  # tie-break before the stable sort
        order = np.argsort(-score_post[b, options] if score_post.ndim == 2 else -score_post[options], kind="stable")
        picks[b, options[order[: int(c)]]] = True
    return _lift(market, picks)


def random_exposure(market: JobMarket, k=None, q=None) -> np.ndarray:
    E_post, _ = _posting_level(market)
    rng, picks = _exposure_rng(market, f"random|{k}|{q}"), np.zeros_like(E_post)
    for b, c in enumerate(show_count(E_post.sum(1), k, q)):
        options = np.nonzero(E_post[b])[0]
        if c:
            picks[b, rng.choice(options, size=int(c), replace=False)] = True
    return _lift(market, picks)


def _posting_attr(market: JobMarket, col: str) -> np.ndarray:
    first_slot = np.unique(market.slot_to_posting, return_index=True)[1]
    return market.listings[col].to_numpy(float)[first_slot]


def salary_max_exposure(market, k=None, q=None):
    return _by_score(market, _posting_attr(market, "salary"), k, q, "salary_max")


def salary_min_exposure(market, k=None, q=None):
    return _by_score(market, -_posting_attr(market, "salary"), k, q, "salary_min")


def under_leveled_exposure(market, k=None, q=None):
    return _by_score(market, -_posting_attr(market, "seniority"), k, q, "under_leveled")


def transparent_exposure(market, k=None, q=None):
    """Deterministic part of the seeker score: the seeker's own ranking with the taste shocks removed.
    Requires build_market to stash it as market.score_deterministic (seeker x posting)."""
    if not hasattr(market, "score_deterministic"):
        raise ValueError("build_market must set market.score_deterministic for the transparent policy")
    return _by_score(market, market.score_deterministic, k, q, "transparent")


POLICIES = {"random": random_exposure, "salary_max": salary_max_exposure, "salary_min": salary_min_exposure,
            "under_leveled": under_leveled_exposure, "transparent": transparent_exposure}


class JobLlmRecommendations(LlmRecommendations):
    """Saved recommender picks are posting indices; lift them to slots."""

    def exposure(self, market: JobMarket, name: str, rep: int = 0):
        recs = self.table.get(market.spec.market_id, {}).get(name)
        if recs is None:
            return None
        E_post, _ = _posting_level(market)
        picks = np.zeros_like(E_post)
        stats = {"seekers_with_options": int((E_post.sum(1) > 0).sum()), "missing": 0, "refused": 0, "invalid_ids": 0, "dropped_outside_E": 0}
        for b in np.nonzero(E_post.sum(1) > 0)[0]:
            r = recs.get((int(b), rep))
            if r is None:
                stats["missing"] += 1
                continue
            stats["refused"] += int(r.get("status") == "refused")
            stats["invalid_ids"] += int(r.get("invalid_ids", 0))
            for p in r.get("listings", []):
                if E_post[b, p]:
                    picks[b, p] = True
                else:
                    stats["dropped_outside_E"] += 1
        return _lift(market, picks), stats


def build_exposure(market: JobMarket, cfg: dict, llm: JobLlmRecommendations | None = None):
    policy, k, q = cfg["policy"], cfg.get("k"), cfg.get("q")
    if policy == "llm":
        return None if llm is None else llm.exposure(market, cfg["name"], cfg.get("rep", 0))
    if policy not in POLICIES:
        raise ValueError(f"unknown exposure policy {policy}")
    return POLICIES[policy](market, k, q), {}
