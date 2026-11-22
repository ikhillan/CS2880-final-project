"""Evaluation and paired analysis for the jobs anchor follow-up.

Policies must be built before calling this module: realized preferences and
full-information matches are evaluation information, not recommender inputs.
All ranks below refer to distinct postings, not interchangeable capacity slots.

``evaluate`` and ``transition`` return ``{group: {metric: {numerator,
denominator}}}``. Counts remain integers. The normalized-rank numerator is a
sum of individual fractional costs; keeping its denominator as the person
count avoids accidentally weighting markets by their feasible-set sizes.

Analysis consumes long rows with window, variant, market, seed, arm, group,
metric, numerator and denominator. A transition table may also have baseline.
Within each seed we pool numerators and denominators across its markets. The
point estimand is the equally weighted mean of these seed rates, matching the
existing jobs paired comparisons. Bootstrap draws resample whole seed blocks
and are shared across arms, metrics and groups. They do not represent sampling
uncertainty about the real NYC population or validate the preference model.

Undefined denominators remain NaN. An incomplete seed panel or an undefined
seed rate produces an undefined main estimate and interval, with the available
seed mean and support counts reported separately. No absent group is imputed
as a zero outcome. Optional simultaneous intervals use the bootstrap maximum
absolute centered deviation over the requested contrast family. These are
approximate, post-hoc sensitivity intervals, not finite-sample guarantees.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def _posting_context(inst, slot_to_posting):
    eligible = np.asarray(inst.E, dtype=bool)
    ranks = np.asarray(inst.rank_b)
    if eligible.ndim != 2 or ranks.shape != eligible.shape:
        raise ValueError("Eligibility and applicant ranks must be aligned matrices")
    raw = np.asarray(slot_to_posting)
    if raw.shape != (eligible.shape[1],) or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("slot_to_posting must have one integer identifier per slot")
    if (raw < 0).any():
        raise ValueError("Posting identifiers must be nonnegative")
    # Renumbering also supports nonconsecutive posting identifiers.
    _, slots = np.unique(raw, return_inverse=True)
    n, p = eligible.shape[0], len(np.unique(raw))
    post_e = np.zeros((n, p), dtype=bool)
    post_ranks = np.full((n, p), -1, dtype=np.int64)
    for i in range(n):
        ordered_slots = np.flatnonzero(eligible[i])
        ordered_slots = ordered_slots[np.argsort(ranks[i, ordered_slots], kind="stable")]
        ordered_posts = list(dict.fromkeys(slots[ordered_slots].tolist()))
        post_e[i, ordered_posts] = True
        post_ranks[i, ordered_posts] = np.arange(len(ordered_posts))
    return eligible, slots, post_e, post_ranks, post_e.sum(axis=1)


def _match(match, eligible, slots, post_ranks, degree):
    match = np.asarray(match)
    if match.shape != (eligible.shape[0],) or not np.issubdtype(match.dtype, np.integer):
        raise ValueError("A matching must have one integer slot or -1 per applicant")
    if (match < -1).any() or (match >= eligible.shape[1]).any():
        raise ValueError("Matching contains an invalid slot")
    placed = match >= 0
    ids = np.flatnonzero(placed)
    if len(np.unique(match[placed])) != int(placed.sum()):
        raise ValueError("Two applicants occupy the same capacity slot")
    if not eligible[ids, match[ids]].all():
        raise ValueError("Matching includes an infeasible pair")
    posting = np.full(len(match), -1, dtype=np.int64)
    posting[placed] = slots[match[placed]]
    # Every feasible posting is better than being unmatched. For degree zero,
    # the ordinal code is zero but normalized unmatched cost is still one.
    rank = degree.copy()
    rank[ids] = post_ranks[ids, posting[ids]]
    cost = np.ones(len(match), dtype=float)
    cost[ids] = rank[ids] / degree[ids]
    return placed, posting, rank, cost


def _groups(groups, n, degree):
    if not isinstance(groups, Mapping):
        raise TypeError("groups must map names to Boolean applicant masks")
    result = {"all": np.ones(n, dtype=bool)}
    for name, value in groups.items():
        mask = np.asarray(value)
        if mask.shape != (n,) or mask.dtype != np.dtype(bool):
            raise ValueError(f"Group {name!r} must be a Boolean mask of length {n}")
        if name == "all" and not mask.all():
            raise ValueError("The all group must contain every applicant")
        result[str(name)] = mask
    bins = {"degree_0": degree == 0, "degree_1_5": (degree >= 1) & (degree <= 5),
            "degree_6_10": (degree >= 6) & (degree <= 10), "degree_11_plus": degree >= 11}
    if set(bins) & set(groups):
        raise ValueError("Degree-bin group names are reserved")
    result.update(bins)
    return result


def _ratio(numerator, denominator):
    return {"numerator": numerator, "denominator": int(denominator)}


def evaluate(inst, shown, match, full_match, groups, slot_to_posting):
    """Return policy outcomes, including zero-option applicants in placement.

    coverage_conditional conditions on a full-information match; coverage_population
    has the whole group as denominator. top_choice_included conditions on at
    least one feasible posting. normalized_rank is a cost: best posting = 0,
    posting rank r (zero based) = r / degree, unmatched = 1. It is a descriptive
    ordinal score, not cardinal welfare. matched_rank is one-based and only
    describes the selected subset that obtains a posting.
    """
    eligible, slots, post_e, post_ranks, degree = _posting_context(inst, slot_to_posting)
    shown = np.asarray(shown, dtype=bool)
    if shown.shape != eligible.shape or (shown & ~eligible).any():
        raise ValueError("The displayed graph must be a subset of eligibility")
    placed, posting, rank, cost = _match(match, eligible, slots, post_ranks, degree)
    has_full, full_post, _, _ = _match(full_match, eligible, slots, post_ranks, degree)
    shown_post = np.zeros_like(post_e)
    for slot, post in enumerate(slots):
        shown_post[:, post] |= shown[:, slot]
    ids = np.flatnonzero(placed)
    if not shown[ids, np.asarray(match)[ids]].all():
        raise ValueError("An assignment uses a capacity slot absent from the menu")
    covered = np.zeros(len(degree), dtype=bool)
    ids = np.flatnonzero(has_full)
    covered[ids] = shown_post[ids, full_post[ids]]
    top_shown = ((post_ranks == 0) & shown_post).any(axis=1)
    out = {}
    for group, mask in _groups(groups, len(degree), degree).items():
        n = int(mask.sum())
        has_options = mask & (degree > 0)
        matched = mask & placed
        out[group] = {
            "group_count": _ratio(n, 1),
            "placed": _ratio(int(matched.sum()), n),
            "coverage_conditional": _ratio(int((mask & covered).sum()), int((mask & has_full).sum())),
            "coverage_population": _ratio(int((mask & covered).sum()), n),
            "full_match_share": _ratio(int((mask & has_full).sum()), n),
            "top_choice_included": _ratio(int((mask & top_shown).sum()), int(has_options.sum())),
            "top_choice_included_population": _ratio(int((mask & top_shown).sum()), n),
            "eligible_share": _ratio(int(has_options.sum()), n),
            "feasible_degree": _ratio(int(degree[mask].sum()), n),
            "normalized_rank": _ratio(float(cost[mask].sum()), n),
            "normalized_rank_eligible": _ratio(float(cost[has_options].sum()), int(has_options.sum())),
            "matched_rank": _ratio(int((rank[matched] + 1).sum()), int(matched.sum())),
        }
    return out


def transition(inst, before_match, after_match, groups, slot_to_posting):
    """Paired person-level transitions relative to a named comparison policy.

    gross_exits and gross_entrants use population denominators; exit_rate and
    entry_rate use baseline placed and baseline unplaced denominators. Same
    job ignores swaps between interchangeable slots of one posting. Rank
    worsening includes losing a seat; worse_among_retained additionally
    conditions on being matched under both policies. Both-unmatched people
    are neutral in rank comparisons, including people with no feasible option.
    """
    eligible, slots, _, post_ranks, degree = _posting_context(inst, slot_to_posting)
    bp, bj, br, bc = _match(before_match, eligible, slots, post_ranks, degree)
    ap, aj, ar, ac = _match(after_match, eligible, slots, post_ranks, degree)
    exits, entrants, retained = bp & ~ap, ~bp & ap, bp & ap
    same = retained & (bj == aj)
    switch = retained & (bj != aj)
    worse, better = ar > br, ar < br
    if int(entrants.sum() - exits.sum()) != int(ap.sum() - bp.sum()):
        raise AssertionError("Placement transitions do not conserve headcount")
    out = {}
    for group, mask in _groups(groups, len(degree), degree).items():
        n, before_n = int(mask.sum()), int((mask & bp).sum())
        retained_n = int((mask & retained).sum())
        ex, en = int((mask & exits).sum()), int((mask & entrants).sum())
        out[group] = {
            "group_count": _ratio(n, 1),
            "baseline_placed": _ratio(before_n, n),
            "after_placed": _ratio(int((mask & ap).sum()), n),
            "placement_change": _ratio(en - ex, n),
            "gross_exits": _ratio(ex, n),
            "gross_entrants": _ratio(en, n),
            "exit_rate": _ratio(ex, before_n),
            "entry_rate": _ratio(en, n - before_n),
            "retained": _ratio(retained_n, n),
            "same_job": _ratio(int((mask & same).sum()), n),
            "job_switch": _ratio(int((mask & switch).sum()), n),
            "both_unmatched": _ratio(int((mask & ~bp & ~ap).sum()), n),
            "rank_worse": _ratio(int((mask & worse).sum()), n),
            "rank_better": _ratio(int((mask & better).sum()), n),
            "rank_unchanged": _ratio(int((mask & (ar == br)).sum()), n),
            "worse_among_retained": _ratio(int((mask & worse & retained).sum()), retained_n),
            "switch_among_retained": _ratio(int((mask & switch).sum()), retained_n),
            "normalized_rank_change": _ratio(float((ac[mask] - bc[mask]).sum()), n),
        }
    return out


def _frame(rows):
    frame = pd.DataFrame(rows).copy()
    needed = {"window", "variant", "market", "seed", "arm", "group", "metric", "numerator", "denominator"}
    if missing := needed - set(frame.columns):
        raise ValueError(f"Missing long-table columns: {sorted(missing)}")
    if frame.empty:
        return frame
    identifiers = [c for c in frame.columns if c in needed - {"numerator", "denominator"} or c in ("baseline", "tie")]
    if frame[identifiers].isna().any().any():
        raise ValueError("Long-table identifiers must not be missing")
    if frame.duplicated(identifiers).any():
        raise ValueError("Duplicate market/arm/group/metric records")
    for name in ("numerator", "denominator"):
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        if not np.isfinite(frame[name].to_numpy(float)).all():
            raise ValueError(f"{name} must be finite; use numerator=denominator=0 for an absent group")
    if (frame.denominator < 0).any():
        raise ValueError("Denominators must be nonnegative")
    if ((frame.denominator == 0) & (frame.numerator != 0)).any():
        raise ValueError("A zero denominator must have a zero numerator")
    return frame


def _plan(frame, reps, rng_seed):
    if reps < 1:
        raise ValueError("reps must be positive")
    rng, cache, plans = np.random.default_rng(rng_seed), {}, {}
    for key, block in frame.groupby(["window", "variant"], sort=True):
        seeds = tuple(sorted(block.seed.unique()))
        if seeds not in cache:
            cache[seeds] = rng.integers(0, len(seeds), size=(reps, len(seeds)))
        plans[key] = (seeds, cache[seeds])
    return plans


def _rates(frame, seeds):
    pooled = frame.groupby("seed")[["numerator", "denominator"]].sum().reindex(seeds)
    return np.divide(pooled.numerator.to_numpy(float), pooled.denominator.to_numpy(float),
                     out=np.full(len(seeds), np.nan), where=pooled.denominator.to_numpy(float) > 0)


def _summary(values, draws, expected_seeds, alpha, complete_pairs=True, complete_markets=True):
    finite = np.isfinite(values)
    complete = bool(finite.all() and (expected_seeds is None or len(values) == expected_seeds)
                    and complete_pairs and complete_markets)
    boot = values[draws].mean(axis=1)
    valid_boot = np.isfinite(boot)
    ci_ok = complete and len(values) >= 2
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2]) if ci_ok else (np.nan, np.nan)
    reasons = []
    if expected_seeds is not None and len(values) != expected_seeds:
        reasons.append("incomplete_seed_panel")
    if not finite.all():
        reasons.append("undefined_seed_rate")
    if not complete_pairs:
        reasons.append("unpaired_markets")
    if not complete_markets:
        reasons.append("incomplete_market_panel")
    if len(values) < 2:
        reasons.append("fewer_than_two_seeds")
    return {
        "mean": float(values.mean()) if complete else np.nan,
        "lo": float(lo), "hi": float(hi),
        "available_seed_mean": float(values[finite].mean()) if finite.any() else np.nan,
        "seeds_total": len(values), "seeds_available": int(finite.sum()),
        "expected_seeds": expected_seeds,
        "valid_bootstrap_draws": int(valid_boot.sum()), "bootstrap_draws": len(draws),
        "status": ";".join(reasons) or "ok", "estimand": "mean_of_seed_rates",
    }, boot, ci_ok


def _market_universe(frame):
    return {key: set(zip(part.seed, part.market))
            for key, part in frame.groupby(["window", "variant"], sort=True)}


def _optional_keys(frame):
    return [c for c in ("baseline", "tie") if c in frame]


def aggregate_levels(rows, *, reps=10_000, rng_seed=0, expected_seeds=20, alpha=.05):
    """Tidy levels for policy or transition long rows; returned rates are not pp.

    baseline and tie are retained as grouping columns when present. Tie draws
    are sensitivity scenarios, not additional independent seeds. Empty
    denominators and incomplete support are disclosed rather than dropped.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie between zero and one")
    frame = _frame(rows)
    if frame.empty:
        return pd.DataFrame()
    plans = _plan(frame, reps, rng_seed)
    universe = _market_universe(frame)
    keys = ["window", "variant", "arm"] + _optional_keys(frame) + ["group", "metric"]
    result = []
    for key, part in frame.groupby(keys, sort=True):
        seeds, draws = plans[key[:2]]
        expected = universe[key[:2]]
        missing = len(expected - set(zip(part.seed, part.market)))
        summary, _, _ = _summary(_rates(part, seeds), draws, expected_seeds, alpha,
                                 complete_markets=not missing)
        result.append({**dict(zip(keys, key)), **summary,
                       "markets": len(part), "markets_expected": len(expected), "missing_markets": missing,
                       "numerator_total": part.numerator.sum(),
                       "denominator_total": part.denominator.sum()})
    return pd.DataFrame(result)


def paired_contrasts(rows, comparisons: Sequence[tuple[str, str]], *, metrics=None, groups=None,
                     reps=10_000, rng_seed=0, expected_seeds=20, alpha=.05,
                     simultaneous=True, strict_pairs=True):
    """Return arm-minus-comparator contrasts on identical market records.

    Select primary metrics/groups explicitly: four arm comparisons x
    coverage_conditional/placed x group all gives an eight-contrast family.
    Simultaneous intervals use one shared bootstrap critical value for the
    maximum absolute centered deviation across ALL requested contrasts in a
    window/variant/tie. A missing endpoint makes that entire simultaneous
    family unavailable. Tie draws must not be treated as independent seeds.
    Pointwise lo/hi remain available. Defaults reject unpaired market records;
    strict_pairs=False reports an undefined estimate instead of dropping them.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie between zero and one")
    comparisons = [tuple(pair) for pair in comparisons]
    if any(len(pair) != 2 or pair[0] == pair[1] for pair in comparisons):
        raise ValueError("Each comparison needs two distinct arm names")
    if len(set(comparisons)) != len(comparisons):
        raise ValueError("Repeated arm comparisons would duplicate the inferential family")
    frame = _frame(rows)
    if frame.empty:
        return pd.DataFrame()
    plans = _plan(frame, reps, rng_seed)
    universe = _market_universe(frame)
    # Requested primary endpoints must not vanish and thereby shrink the
    # simultaneous family. Check within each scenario before filtering.
    scenario_keys = ["window", "variant"] + _optional_keys(frame)
    for scenario, part in frame.groupby(scenario_keys, sort=True):
        wanted_metrics = list(metrics) if metrics is not None else sorted(part.metric.unique())
        wanted_groups = list(groups) if groups is not None else sorted(part.group.unique())
        present = set(zip(part.group, part.metric))
        missing = {(g, m) for g in wanted_groups for m in wanted_metrics} - present
        if missing:
            raise ValueError(f"Missing requested endpoints in {scenario}: {sorted(missing)}")
    if metrics is not None:
        frame = frame[frame.metric.isin(metrics)]
    if groups is not None:
        frame = frame[frame.group.isin(groups)]
    keys = ["window", "variant"] + _optional_keys(frame) + ["group", "metric"]
    result, bootstraps = [], []
    for key, part in frame.groupby(keys, sort=True):
        seeds, draws = plans[key[:2]]
        for arm, comparator in comparisons:
            a, b = part[part.arm == arm], part[part.arm == comparator]
            joined = a.merge(b, on=["seed", "market"], how="outer", suffixes=("_a", "_b"),
                             indicator=True, validate="one_to_one")
            paired = joined._merge.eq("both")
            unpaired = int((~paired).sum())
            if (unpaired or a.empty or b.empty) and strict_pairs:
                raise ValueError(f"Incomplete paired markets for {key}: {arm} vs {comparator}")
            common = joined.loc[paired, ["seed", "market"]]
            missing = len(universe[key[:2]] - set(zip(common.seed, common.market)))
            if missing and strict_pairs:
                raise ValueError(f"Incomplete paired markets for {key}: {missing} absent from common panel")
            aa = a.merge(common, on=["seed", "market"], validate="one_to_one")
            bb = b.merge(common, on=["seed", "market"], validate="one_to_one")
            values = _rates(aa, seeds) - _rates(bb, seeds)
            summary, boot, ci_ok = _summary(values, draws, expected_seeds, alpha,
                                           complete_pairs=not unpaired and not a.empty and not b.empty,
                                           complete_markets=not missing)
            result.append({**dict(zip(keys, key)), "arm": arm, "comparator": comparator,
                           **summary, "paired_markets": int(paired.sum()), "unpaired_markets": unpaired,
                           "markets_expected": len(universe[key[:2]]), "missing_markets": missing})
            bootstraps.append((boot, ci_ok))
    table = pd.DataFrame(result)
    if simultaneous and not table.empty:
        table["simultaneous_lo"] = np.nan
        table["simultaneous_hi"] = np.nan
        family_keys = ["window", "variant"] + (["tie"] if "tie" in table else [])
        for _, family in table.groupby(family_keys, sort=True):
            size = len(family)
            table.loc[family.index, "family_size"] = size
            complete = all(bootstraps[index][1] for index in family.index)
            table.loc[family.index, "simultaneous_family_complete"] = complete
            if complete:
                deviations = np.stack([np.abs(bootstraps[index][0] - table.loc[index, "mean"])
                                       for index in family.index], axis=1)
                critical = float(np.quantile(deviations.max(axis=1), 1 - alpha))
                table.loc[family.index, "simultaneous_critical_value"] = critical
                table.loc[family.index, "simultaneous_lo"] = family["mean"] - critical
                table.loc[family.index, "simultaneous_hi"] = family["mean"] + critical
        table["simultaneous_method"] = "centered_max_absolute_deviation_bootstrap"
        table["family_confidence_level"] = 1 - alpha
        table["inference_scope"] = "post_hoc_simulation_sensitivity_not_confirmatory"
    return table
