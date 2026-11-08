"""Per-group outcomes for the social-impact comparison (design report section 7 and slide 23)."""
from __future__ import annotations

import numpy as np

from .audit import Oracle
from .markets import Market
from .matching import UNMATCHED


def rank_score(inst, match_b) -> np.ndarray:
    """Slide 23: 0 if unmatched, otherwise 1 - (rank - 1) / list length, with rank counted from 1."""
    out = np.zeros(inst.n_b)
    for b in np.nonzero(match_b >= 0)[0]:
        out[b] = 1 - inst.rank_b[b, match_b[b]] / inst.list_len_b[b]
    return out


def _renter_ranks(inst, match_b) -> np.ndarray:
    r = np.full(inst.n_b, UNMATCHED, dtype=np.int64)
    bs = np.nonzero(match_b >= 0)[0]
    r[bs] = inst.rank_b[bs, match_b[bs]]
    return r


def group_masks(market: Market) -> dict[str, np.ndarray]:
    r = market.renters
    return {
        "all": np.ones(len(r), bool),
        "lower_income": r.lower_income.to_numpy(bool),
        "higher_income": ~r.lower_income.to_numpy(bool),
        "children": r.has_children.to_numpy(bool),
        "no_children": ~r.has_children.to_numpy(bool),
    }


def outcomes(market: Market, oracle: Oracle, x: np.ndarray) -> dict:
    """Outcomes after plan x, compared with the provisional DA on E0 (no audits)."""
    inst, e0 = market.inst, oracle.e0
    none = np.zeros(inst.n_firms, bool)
    m0, s0, block0 = oracle.evaluate(none)
    m1, s1, block1 = oracle.evaluate(x)
    feasible, shown, after = inst.E.sum(1), e0.sum(1), inst.visible(x, e0).sum(1)
    rank0, rank1 = _renter_ranks(inst, m0), _renter_ranks(inst, m1)
    score0, score1 = rank_score(inst, m0), rank_score(inst, m1)
    rent, income = market.listings.rent.to_numpy(float), market.renters.income_month.to_numpy(float)
    has_options = feasible > 0

    def mean(a, mask):
        return float(np.mean(a[mask])) if mask.any() else None

    out = {}
    for name, g in group_masks(market).items():
        o = g & has_options
        matched1 = g & (m1 >= 0)
        out[name] = {
            "n": int(g.sum()),
            "no_feasible_listing": mean(~has_options, g),  # structural: never attributed to information
            "feasible": mean(feasible, g),
            "shown": mean(shown, g),
            "hidden_count": mean(feasible - shown, g),
            "hidden_share": mean(1 - shown / np.maximum(feasible, 1), o),
            "access_after": mean(after / np.maximum(feasible, 1), o),
            "hidden_blockers_before": mean(block0.sum(1), g),
            "match_before": mean(m0 >= 0, g),
            "match_after": mean(m1 >= 0, g),
            "unmatched_despite_options_before": mean(m0 < 0, o),
            "unmatched_despite_options_after": mean(m1 < 0, o),
            "rank_score_before": mean(score0, g),
            "rank_score_after": mean(score1, g),
            "rank_score_change": mean(score1 - score0, g),
            "worse_off": mean(rank1 > rank0, g),
            "better_off": mean(rank1 < rank0, g),
            "lost_match": mean((m0 >= 0) & (m1 < 0), g),
            "rent_after": float(np.mean(rent[m1[matched1]])) if matched1.any() else None,
            "rent_burden_after": float(np.mean(12 * rent[m1[matched1]] / np.maximum(income[matched1], 1))) if matched1.any() else None,
        }
    srank0 = np.where(s0 >= 0, 1, 0)
    out["listings"] = {"n": inst.n_s, "filled_before": float(srank0.mean()), "filled_after": float((s1 >= 0).mean())}
    out["n_firms_audited_renter_side"] = int(x[: market.n_firms_renter].sum())
    out["n_firms_audited_listing_side"] = int(x[market.n_firms_renter:].sum())
    return out


def nonmonotone_additions(inst, oracle: Oracle, x: np.ndarray) -> dict:
    """From a passing plan, how many single extra audits break stability? (slide 8: more audits can make things worse)"""
    y, broke, tried = x.copy(), 0, 0
    for f in np.nonzero(~x)[0]:
        y[f] = True
        tried += 1
        broke += int(not oracle.stable(y))
        y[f] = False
    return {"tried": tried, "broke": broke}


def plan_spread(market: Market, oracle: Oracle, plans: list[np.ndarray]) -> dict:
    """Do equally cheap plans favor different groups? (slide 24)"""
    inst, masks = market.inst, group_masks(market)
    low, high = masks["lower_income"], masks["higher_income"]
    score0 = rank_score(inst, oracle.evaluate(np.zeros(inst.n_firms, bool))[0])
    allocations, low_change, high_change, low_worse = set(), [], [], []
    rank0 = _renter_ranks(inst, oracle.evaluate(np.zeros(inst.n_firms, bool))[0])
    for x in plans:
        m = oracle.evaluate(x)[0]
        allocations.add(m.tobytes())
        change = rank_score(inst, m) - score0
        low_change.append(float(change[low].mean()) if low.any() else 0.0)
        high_change.append(float(change[high].mean()) if high.any() else 0.0)
        low_worse.append(float((_renter_ranks(inst, m)[low] > rank0[low]).mean()) if low.any() else 0.0)
    return {
        "n_plans": len(plans),
        "n_allocations": len(allocations),
        "lower_income_change_range": [min(low_change), max(low_change)] if plans else None,
        "higher_income_change_range": [min(high_change), max(high_change)] if plans else None,
        "lower_income_worse_range": [min(low_worse), max(low_worse)] if plans else None,
    }
