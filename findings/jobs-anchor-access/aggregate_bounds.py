"""Bounds from released jobs aggregate levels; no new allocation is simulated.

Run from any directory: python3 findings/jobs-anchor-access/aggregate_bounds.py
Only the standard library is required. The tiny self-checks use invented
binary outcomes/preferences, not the unavailable jobs data.
"""
from __future__ import annotations

import ast
import csv
from decimal import Decimal, getcontext
import hashlib
from itertools import permutations, product
import json
from pathlib import Path

getcontext().prec = 45
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GROUPS = ("all", "lower_wage", "higher_wage")


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def count_from_rate(value, denominator, percent=False):
    count = Decimal(str(value)) * denominator / (100 if percent else 1)
    rounded = int(count.to_integral_value())
    if abs(count - rounded) > Decimal("0.000001"):
        raise AssertionError(f"Published level does not reconstruct a count: {value} x {denominator}")
    if not 0 <= rounded <= denominator:
        raise AssertionError("Count outside its population")
    return rounded


def signed_count(value, denominator, percent=False):
    value = Decimal(str(value))
    sign = -1 if value < 0 else 1
    return sign * count_from_rate(abs(value), denominator, percent)


def percent(numerator, denominator):
    return "" if not denominator else f"{100 * numerator / denominator:.9f}"


def binary_exit_bounds(n, before, after):
    return max(0, before - after), min(before, n - after)


def coverage_bounds(full, hits, change_budget):
    return max(0, hits - change_budget), min(full, hits + change_budget)


def default_lower_share():
    # Read the dataclass default without importing the data-loading module.
    tree = ast.parse((ROOT / "hiddenoptions/jobs/markets.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "JobMarketSpec":
            for member in node.body:
                if isinstance(member, ast.AnnAssign) and isinstance(member.target, ast.Name) and member.target.id == "lower_share":
                    return float(ast.literal_eval(member.value))
    raise AssertionError("Cannot establish the archived group sampling ratio")


def _tiny_da(preferences, priorities, menus):
    n = len(preferences)
    holders, matched, next_choice = [-1] * n, [-1] * n, [0] * n
    free = list(range(n))
    while free:
        i = free.pop()
        while next_choice[i] < n:
            j = preferences[i][next_choice[i]]
            next_choice[i] += 1
            if not menus[i][j]:
                continue
            old = holders[j]
            if old < 0 or priorities[j].index(i) < priorities[j].index(old):
                holders[j], matched[i] = i, j
                if old >= 0:
                    matched[old] = -1
                    free.append(old)
                break
    return matched


def self_checks():
    # Binary coupling intervals are sharp when only the two margins are fixed.
    for n in range(1, 6):
        seen = {}
        for before in product([0, 1], repeat=n):
            for after in product([0, 1], repeat=n):
                exits = sum(a and not b for a, b in zip(before, after))
                seen.setdefault((sum(before), sum(after)), set()).add(exits)
        for (before, after), values in seen.items():
            low, high = binary_exit_bounds(n, before, after)
            assert values == set(range(low, high + 1))
    # All two-person strict preference/priority orders and all single-row edits.
    orders = list(permutations(range(2)))
    row_menus = list(product([False, True], repeat=2))
    transitions = 0
    for preferences in product(orders, repeat=2):
        for priorities in product(orders, repeat=2):
            for menus in product(row_menus, repeat=2):
                before = _tiny_da(preferences, priorities, menus)
                for i in range(2):
                    for replacement in row_menus:
                        changed = list(menus)
                        changed[i] = replacement
                        after = _tiny_da(preferences, priorities, changed)
                        assert sum(a >= 0 and b < 0 for a, b in zip(before, after)) <= 1
                        assert sum(a < 0 and b >= 0 for a, b in zip(before, after)) <= 1
                        transitions += 1
    return {"binary_population_sizes": [1, 2, 3, 4, 5], "invented_two_person_single_row_transitions": transitions}


def build():
    sources = set()
    result = []
    for window, suffix, config_name in [
        ("frozen", "", "configs/jobs_frozen.json"),
        ("replication", "replication", "configs/jobs_replication.json"),
    ]:
        folder = ROOT / "findings/jobs-gpt-frozen" / suffix
        config_path = ROOT / config_name
        cfg = json.loads(config_path.read_text())
        sources.add(config_path)
        primary = next(v for v in cfg["variants"] if v["label"] == "primary")
        settings = {**cfg["market"], **primary}
        assert settings["design"] == "stratified"
        assert len(cfg["grid"]["sizes"]) == 1
        n = int(cfg["grid"]["sizes"][0])
        lower = round(n * float(settings.get("lower_share", default_lower_share())))
        group_size = {"all": n, "lower_wage": lower, "higher_wage": n - lower}
        assert n == 24 and lower == 12, "This release uses equal-sized 12-person groups"
        paths = {name: folder / name for name in ("anchor.csv", "accuracy.csv", "capacity_levels.csv")}
        sources.update(paths.values())
        anchors = [r for r in read_csv(paths["anchor.csv"]) if r["variant"] == "primary"]
        accuracy = {(r["rule"], r["group"]): r for r in read_csv(paths["accuracy.csv"])}
        levels = {r["rule"]: r for r in read_csv(paths["capacity_levels.csv"]) if r["variant"] == "primary"}
        expected_markets = len(cfg["grid"]["cells"]) * len(cfg["grid"]["months"]) * int(cfg["grid"]["seeds"])
        for a in anchors:
            rule = a["rule"]
            markets = int(levels[rule]["markets"])
            assert markets == expected_markets
            total_n = markets * n
            changed = count_from_rate(a["lists_changed"], total_n, percent=True)
            counts = {}
            for group in GROUPS:
                observed = accuracy[rule, group]
                size = markets * group_size[group]
                before = count_from_rate(observed["placed"], size)
                full = count_from_rate(observed["full_match_share"], size)
                hits = count_from_rate(observed["accurate"], full)
                gain_key = {"all": "gain", "lower_wage": "gain_lower", "higher_wage": "gain_higher"}[group]
                delta = signed_count(a[gain_key], size, percent=True)
                after = before + delta
                assert 0 <= after <= size
                lo, hi = binary_exit_bounds(size, before, after)
                counts[group] = dict(n=size, before=before, after=after, full=full, hits=hits,
                                     delta=delta, lo=lo, hi=hi)
            all_counts = counts["all"]
            for field in ("n", "before", "after", "full", "hits", "delta"):
                assert all_counts[field] == counts["lower_wage"][field] + counts["higher_wage"][field]
            assert count_from_rate(a["placed"], total_n, True) == all_counts["before"]
            assert count_from_rate(a["anchor_placed"], total_n, True) == all_counts["after"]
            assert count_from_rate(levels[rule]["placed"], total_n, True) == all_counts["before"]
            assert count_from_rate(levels[rule]["full_placed"], total_n, True) == all_counts["full"]
            strat_lo = sum(counts[g]["lo"] for g in GROUPS[1:])
            strat_hi = sum(counts[g]["hi"] for g in GROUPS[1:])
            # One changed row produces at most one exit AND at most one entry.
            da_lo = strat_lo
            da_hi = min(strat_hi, changed - max(all_counts["delta"], 0))
            assert da_lo <= da_hi
            for group in GROUPS:
                c = counts[group]
                if group == "all":
                    lower_bound, upper_bound = da_lo, da_hi
                    group_lo, group_hi = strat_lo, strat_hi
                else:
                    other = "higher_wage" if group == "lower_wage" else "lower_wage"
                    lower_bound = max(c["lo"], da_lo - counts[other]["hi"])
                    upper_bound = min(c["hi"], da_hi - counts[other]["lo"])
                    group_lo, group_hi = c["lo"], c["hi"]
                assert lower_bound <= upper_bound
                # Group edit counts are unavailable. Total edits is a valid
                # marginal budget, but cannot be spent twice across groups.
                change_budget = min(changed, c["n"])
                h_lo, h_hi = coverage_bounds(c["full"], c["hits"], change_budget)
                result.append({
                    "window": window, "variant": "primary", "rule": rule, "group": group,
                    "markets": markets, "observations": c["n"], "group_size_per_market": group_size[group],
                    "baseline_placed_count": c["before"], "anchored_placed_count": c["after"],
                    "net_placement_change_count": c["delta"], "changed_menus_all_groups": changed,
                    "baseline_placement_pct": percent(c["before"], c["n"]),
                    "anchored_placement_pct": percent(c["after"], c["n"]),
                    "binary_exit_lower_count": c["lo"], "binary_exit_upper_count": c["hi"],
                    "group_margin_exit_lower_count": group_lo, "group_margin_exit_upper_count": group_hi,
                    "da_exit_lower_count": lower_bound, "da_exit_upper_count": upper_bound,
                    "da_exit_lower_pct": percent(lower_bound, c["n"]), "da_exit_upper_pct": percent(upper_bound, c["n"]),
                    "da_entry_lower_count": lower_bound + c["delta"], "da_entry_upper_count": upper_bound + c["delta"],
                    "da_entry_lower_pct": percent(lower_bound + c["delta"], c["n"]),
                    "da_entry_upper_pct": percent(upper_bound + c["delta"], c["n"]),
                    "da_exit_recipient_upper_pct": percent(upper_bound, c["before"]),
                    "full_match_count": c["full"], "baseline_coverage_hit_count": c["hits"],
                    "coverage_change_budget_upper_count": change_budget,
                    "baseline_conditional_coverage_pct": percent(c["hits"], c["full"]),
                    "anchored_coverage_hit_lower_count": h_lo, "anchored_coverage_hit_upper_count": h_hi,
                    "anchored_conditional_coverage_lower_pct": percent(h_lo, c["full"]),
                    "anchored_conditional_coverage_upper_pct": percent(h_hi, c["full"]),
                    "conditional_coverage_change_lower_pp": percent(h_lo - c["hits"], c["full"]),
                    "conditional_coverage_change_upper_pp": percent(h_hi - c["hits"], c["full"]),
                    "unconditional_coverage_change_lower_pp": percent(h_lo - c["hits"], c["n"]),
                    "unconditional_coverage_change_upper_pp": percent(h_hi - c["hits"], c["n"]),
                })
    sources.update(ROOT / p for p in ("hiddenoptions/jobs/markets.py", "experiments_jobs/accuracy.py", "experiments_jobs/capacity_anchor.py"))
    return result, sources


def main():
    checks = self_checks()
    rows, sources = build()
    with (HERE / "bounds.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "description": "Arithmetic outer bounds from published aggregate levels; not a new jobs replay or confidence intervals.",
        "coverage_estimand": "pooled full-information recipients; not mean per-seed conditional coverage",
        "sharpness": "Binary margin bounds are sharp without mechanism restrictions. Joint DA/anchor sharpness is not claimed.",
        "checks": checks,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)},
    }
    (HERE / "aggregate_bounds_sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Rebuilt {len(rows)} aggregate-bound rows; all count, group and tiny-instance checks passed.")
    for r in rows:
        if r["rule"] == "GPT (5 jobs)" and r["group"] == "all":
            print(f"{r['window']}: exits {r['da_exit_lower_count']}..{r['da_exit_upper_count']} "
                  f"({r['da_exit_lower_pct']}..{r['da_exit_upper_pct']}% of observations); "
                  f"pooled conditional coverage {r['anchored_conditional_coverage_lower_pct']}.."
                  f"{r['anchored_conditional_coverage_upper_pct']}%")


if __name__ == "__main__":
    main()
