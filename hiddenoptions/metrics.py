"""Per-group outcomes for the comparisons in Todd's protocol (sections 2, 3 and 9).

Conventions
    rank           full-list rank, best = 1; an unmatched renter has rank degree + 1 (the outside option is worst)
    M0             DA on the shown pairs (no audits); the common reference for every repair of that policy
    M_E            DA on all feasible pairs (full disclosure); every renter's best stable assignment
    Y_b            1 if renter b has a hidden blocking pair under M0 (a listing not shown that would prefer b, and b prefers it)
    can_hide       renter has more feasible listings than shown (d_b > k_b); only these renters can have a hidden option of their own
    unavoidable    worse off under M_E than under M0: loses under every successful plan
    severity       rank under the plan minus rank under M_E (never negative for a successful plan)
    scarcity       four categories that partition renters (protocol section 3): no feasible listing in the whole cohort; options
                   in the cohort but none in the draw; options in the draw but unmatched under M_E; matched under M_E
Zero-option renters stay in every denominator with Y = 0, and their prevalence is reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .matching import UNMATCHED, blocking_mask, deferred_acceptance


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


def full_list_rank(inst, match_b) -> np.ndarray:
    """1-based rank among feasible listings; degree + 1 when unmatched."""
    r = inst.list_len_b.astype(np.int64) + 1
    bs = np.nonzero(match_b >= 0)[0]
    r[bs] = inst.rank_b[bs, match_b[bs]] + 1
    return r


def group_masks(market) -> dict[str, np.ndarray]:
    r = market.renters
    lower = r.lower_income.to_numpy(bool)
    children = r.has_children.to_numpy(bool) if "has_children" in r else np.zeros(len(r), bool)
    return {"all": np.ones(len(r), bool), "lower_income": lower, "higher_income": ~lower, "children": children, "no_children": ~children}


def price_percentile(market, e0: np.ndarray) -> np.ndarray:
    """Per renter: mean within-candidate-set rent percentile of the shown listings, (ascending midrank - 1) / (d - 1).

    A singleton candidate set scores 0.5; renters with nothing shown get NaN. Uniform shortlists average exactly 0.5.
    """
    E = market.inst.E
    rent = (market.listings.rent_asking if "rent_asking" in market.listings else market.listings.rent).to_numpy(float)
    out = np.full(E.shape[0], np.nan)
    for b in range(E.shape[0]):
        options = np.nonzero(E[b])[0]
        shown = e0[b, options]
        if not shown.any():
            continue
        if len(options) == 1:
            out[b] = 0.5
            continue
        mid = pd.Series(rent[options]).rank(method="average").to_numpy()
        out[b] = float(np.mean((mid[shown] - 1) / (len(options) - 1)))
    return out


def outcomes(market, oracle, x: np.ndarray) -> dict:
    """Outcomes after plan x, compared with M0 (no audits) and M_E (full disclosure). Works for housing and jobs markets."""
    inst, e0 = market.inst, oracle.e0
    none, full = np.zeros(inst.n_firms, bool), np.ones(inst.n_firms, bool)
    m0, s0, block0 = oracle.evaluate(none)
    mE, _, _ = oracle.evaluate(full)
    m1, s1, _ = oracle.evaluate(x)
    feasible, shown, after = inst.E.sum(1), e0.sum(1), inst.visible(x, e0).sum(1)
    rank0, rank1 = _renter_ranks(inst, m0), _renter_ranks(inst, m1)
    full0, fullE, full1 = full_list_rank(inst, m0), full_list_rank(inst, mE), full_list_rank(inst, m1)
    score0, score1 = rank_score(inst, m0), rank_score(inst, m1)
    Y = block0.any(1)
    can_hide = feasible > shown
    unavoidable = fullE > full0
    severity = full1 - fullE
    pct = price_percentile(market, e0)
    rent = market.listings.rent.to_numpy(float)
    income_col = "income" if "income" in market.renters else "income_month"
    income = market.renters[income_col].to_numpy(float)
    has_options = feasible > 0
    cohort_options = market.renters.n_feasible_cell.to_numpy() > 0 if "n_feasible_cell" in market.renters else None

    def mean(a, mask):
        return float(np.mean(a[mask])) if mask.any() else None

    out = {}
    for name, g in group_masks(market).items():
        o = g & has_options
        h = g & can_hide
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
            "blocker_count": int((Y & g).sum()),
            "blocker_rate": mean(Y, g),  # Y_b, zero-option renters included as 0
            "can_hide_count": int(h.sum()),
            "can_hide_share": mean(can_hide, g),  # q
            "blocker_rate_given_can_hide": mean(Y, h),  # undefined (None) when q = 0
            "price_percentile_shown": float(np.nanmean(pct[g])) if np.isfinite(pct[g]).any() else None,
            "match_before": mean(m0 >= 0, g),
            "match_after": mean(m1 >= 0, g),
            "match_full_disclosure": mean(mE >= 0, g),
            "top3_before": mean((m0 >= 0) & (full0 <= 3), g),
            "prefer_full_to_initial": mean(fullE < full0, g),
            "unmatched_despite_options_before": mean(m0 < 0, o),
            "unmatched_despite_options_after": mean(m1 < 0, o),
            "rank_score_before": mean(score0, g),
            "rank_score_after": mean(score1, g),
            "rank_score_change": mean(score1 - score0, g),
            "worse_off": mean(rank1 > rank0, g),
            "better_off": mean(rank1 < rank0, g),
            "lost_match": mean((m0 >= 0) & (m1 < 0), g),
            "unavoidable_loss": mean(unavoidable, g),
            "severity_positive": mean(severity > 0, g),
            "severity_mean": mean(severity, g),
            "severity_positive_unavoidable": mean(severity > 0, g & unavoidable),
            "severity_mean_unavoidable": mean(severity, g & unavoidable),
            "rent_after": float(np.mean(rent[m1[matched1]])) if matched1.any() else None,
            "rent_burden_after": float(np.mean(12 * rent[m1[matched1]] / np.maximum(income[matched1], 1))) if matched1.any() else None,
            "scarcity_no_option_in_cohort": mean(~cohort_options, g) if cohort_options is not None else None,
            "scarcity_none_in_draw": mean(cohort_options & ~has_options, g) if cohort_options is not None else None,
            "scarcity_unmatched_full": mean(has_options & (mE < 0), g),
            "scarcity_matched_full": mean(mE >= 0, g),
        }
    out["listings"] = {"n": inst.n_s, "filled_before": float((s0 >= 0).mean()), "filled_after": float((s1 >= 0).mean())}
    out["n_firms_audited_renter_side"] = int(x[: market.n_firms_renter].sum())
    out["n_firms_audited_listing_side"] = int(x[market.n_firms_renter:].sum())
    return out


def group_switch(market, e0_a: np.ndarray, e0_b: np.ndarray) -> dict:
    """Four DA runs: lower-income renters see A or B, higher-income renters see A or B (protocol section 2).

    Keys are 'LH' with each letter 'A' or 'B' (the policy shown to that group). Each entry holds hidden-blocker counts and group
    sizes, so the analysis can form standardized gaps Z_ij and the decomposition S_L + S_H = T with identical denominators.
    """
    inst = market.inst
    lower = market.renters.lower_income.to_numpy(bool)
    out = {}
    for low_policy, high_policy in (("A", "A"), ("B", "A"), ("A", "B"), ("B", "B")):
        e0 = np.where(lower[:, None], e0_a if low_policy == "A" else e0_b, e0_a if high_policy == "A" else e0_b)
        mb, ms = deferred_acceptance(inst, e0)
        Y = blocking_mask(inst, mb, ms).any(1)
        out[low_policy + high_policy] = {"lower_count": int((Y & lower).sum()), "lower_n": int(lower.sum()),
                                         "higher_count": int((Y & ~lower).sum()), "higher_n": int((~lower).sum())}
    return out


def nonmonotone_additions(inst, oracle, x: np.ndarray) -> dict:
    """From a passing plan, how many single extra audits break stability? (slide 8: more audits can make things worse)"""
    y, broke, tried = x.copy(), 0, 0
    for f in np.nonzero(~x)[0]:
        y[f] = True
        tried += 1
        broke += int(not oracle.stable(y))
        y[f] = False
    return {"tried": tried, "broke": broke}


def plan_spread(market, oracle, plans: list[np.ndarray]) -> dict:
    """Do equally cheap plans favor different groups? (slide 24 and protocol section 9)"""
    inst, masks = market.inst, group_masks(market)
    low, high = masks["lower_income"], masks["higher_income"]
    m0 = oracle.evaluate(np.zeros(inst.n_firms, bool))[0]
    fullE = full_list_rank(inst, oracle.evaluate(np.ones(inst.n_firms, bool))[0])
    score0, rank0 = rank_score(inst, m0), _renter_ranks(inst, m0)
    allocations, low_change, high_change, low_worse, low_severity = set(), [], [], [], []
    for x in plans:
        m = oracle.evaluate(x)[0]
        allocations.add(m.tobytes())
        change = rank_score(inst, m) - score0
        low_change.append(float(change[low].mean()) if low.any() else 0.0)
        high_change.append(float(change[high].mean()) if high.any() else 0.0)
        low_worse.append(float((_renter_ranks(inst, m)[low] > rank0[low]).mean()) if low.any() else 0.0)
        low_severity.append(float((full_list_rank(inst, m)[low] - fullE[low] > 0).mean()) if low.any() else 0.0)
    rng = lambda v: [min(v), max(v)] if plans else None
    return {"n_plans": len(plans), "n_allocations": len(allocations), "lower_income_change_range": rng(low_change),
            "higher_income_change_range": rng(high_change), "lower_income_worse_range": rng(low_worse),
            "lower_income_severity_positive_range": rng(low_severity)}
