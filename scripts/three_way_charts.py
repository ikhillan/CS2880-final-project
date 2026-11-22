"""Regenerate the three-setting figures from released aggregate results.

Chile comes from findings/chile/rebuild.py's verified four-cohort tables.
NYC rental coverage and its intervals come from gpt-luna-rentals/accuracy.txt;
that released text rounds percentage points to one decimal place. Rental
placement and subgroup placement use the seed-block intervals in
rental-capacity-gpt/*/capacity_decomposition.csv. Jobs use
jobs-gpt-frozen/{,replication/}accuracy_pairs.csv, including paired intervals.

Run from any directory. The optional positional argument selects an output
folder. The default is the repository's findings directory. No individual
records, model calls, or new statistical resampling are used.
"""
import argparse
import csv
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FINDINGS = Path(__file__).resolve().parents[1] / "findings"
RENTAL_WINDOWS = (
    ("stage1_pilot", "Development\n2026"),
    ("heldout_primary", "Held out\n2026"),
    ("heldout_replication", "Replication\n2025"),
)
JOB_WINDOWS = (("", "Frozen"), ("replication", "Replication"))
BLUE, ORANGE, LIGHT, GREY = "#2E6FBF", "#C8571B", "#E9A27C", "#555555"


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def select(rows, **criteria):
    matches = [row for row in rows if all(row[key] == value for key, value in criteria.items())]
    if len(matches) != 1:
        raise ValueError(f"Expected one aggregate row for {criteria}, found {len(matches)}")
    return matches[0]


def estimate(value, lower=None, upper=None):
    point = float(value)
    if lower is None:
        return point, point, point
    low, high = float(lower), float(upper)
    if not low <= point <= high:
        raise ValueError(f"Interval [{low}, {high}] excludes the reported estimate {point}")
    return point, low, high


def jobs_change(row, metric):
    key = "d_" + metric
    return estimate(*(100 * float(row[key + suffix]) for suffix in ("", "_lo", "_hi")))


def load_chile():
    cohorts = read_csv(FINDINGS / "chile/cohort_comparison.csv")
    groups = read_csv(FINDINGS / "chile/group_comparison.csv")
    labels, coverage, placement = [], [], []
    by_group = [[], []]
    for year in ("2022", "2023", "2024", "2025"):
        labels.append(year)
        cohort = select(cohorts, year=year, comparison="rank_vs_blind")
        coverage.append(estimate(cohort["hit_change_pp"]))
        placement.append(estimate(cohort["placement_change_pp"]))
        for target, group in zip(by_group, ("no_initial_offer", "initial_offer")):
            row = select(groups, year=year, comparison="rank_vs_blind", group=group)
            target.append(estimate(row["placement_change_pp"]))
    return labels, coverage, placement, by_group


def load_rentals():
    # This text is the released source for the paired accuracy intervals.
    # Require one complete GPT record for each enriched-design window.
    number = r"[+-]?\d+(?:\.\d+)?"
    pattern = re.compile(
        rf"gpt_k5:.*?\(d acc ({number}) \[({number}),({number})\], "
        rf"d placed ({number}) \[({number}),({number})\]\)"
    )
    lines = (FINDINGS / "gpt-luna-rentals/accuracy.txt").read_text().splitlines()
    labels, coverage, placement = [], [], []
    by_group = [[], []]
    for window, label in RENTAL_WINDOWS:
        matching = [line for line in lines if line.split()[:2] == [window, "enriched"]]
        if len(matching) != 1 or (match := pattern.search(matching[0])) is None:
            raise ValueError(f"Missing or ambiguous rental accuracy record for {window}")
        values = match.groups()
        coverage.append(estimate(*values[:3]))
        rounded_placement = float(values[3])
        labels.append(label)
        decomposition = read_csv(FINDINGS / "rental-capacity-gpt" / window / "capacity_decomposition.csv")
        paired = select(decomposition, variant="enriched", rule="GPT (5)", baseline="Random (5)")
        placement.append(estimate(*(paired["d_placed" + suffix] for suffix in ("", "_lo", "_hi"))))
        table = read_csv(FINDINGS / "rental-capacity-gpt" / window / "capacity_levels.csv")
        baseline = select(table, variant="enriched", rule="Random (5)")
        policy = select(table, variant="enriched", rule="GPT (5)")
        for target, field in zip(by_group, ("placed_higher", "placed_lower")):
            key = "d_" + field
            target.append(estimate(*(paired[key + suffix] for suffix in ("", "_lo", "_hi"))))
            if abs(target[-1][0] - (float(policy[field]) - float(baseline[field]))) > 1e-9:
                raise ValueError(f"Rental subgroup sources disagree for {window} {field}")
        # Independently released placement levels agree to the text's precision.
        delta = float(policy["placed"]) - float(baseline["placed"])
        if abs(delta - rounded_placement) > 0.051 or abs(delta - placement[-1][0]) > 1e-9:
            raise ValueError(f"Rental placement sources disagree for {window}")
    return labels, coverage, placement, by_group


def load_jobs():
    labels, coverage, placement = [], [], []
    by_group = [[], []]
    for window, label in JOB_WINDOWS:
        table = read_csv(FINDINGS / "jobs-gpt-frozen" / window / "accuracy_pairs.csv")
        arms = {"rule": "GPT (5 jobs)", "baseline": "Random (5 jobs)"}
        row = select(table, group="all", **arms)
        labels.append(label)
        coverage.append(jobs_change(row, "accurate"))
        placement.append(jobs_change(row, "placed"))
        for target, group in zip(by_group, ("lower_wage", "higher_wage")):
            target.append(jobs_change(select(table, group=group, **arms), "placed"))
    return labels, coverage, placement, by_group


def bars(ax, positions, values, width, color, name, precision):
    points, lows, highs = zip(*values)
    ax.bar(positions, points, width, color=color, label=name, zorder=2)
    if any(low != high for low, high in zip(lows, highs)):
        errors = [[point - low for point, low in zip(points, lows)],
                  [high - point for point, high in zip(points, highs)]]
        ax.errorbar(positions, points, yerr=errors, fmt="none", color="#333333",
                    capsize=3, linewidth=1, zorder=3)
    for x, (point, low, high) in zip(positions, values):
        above = point >= 0
        anchor = high if above else low
        ax.annotate(f"{point:+.{precision}f}", (x, anchor),
                    xytext=(0, 5 if above else -5), textcoords="offset points",
                    ha="center", va="bottom" if above else "top", fontsize=9.5)


def finish_axes(ax, labels, title, comparison, values):
    ax.axhline(0, color=GREY, linewidth=0.9, zorder=1)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=35)
    ax.text(0, 1.06, comparison, transform=ax.transAxes, fontsize=10, color=GREY)
    ax.set_ylabel("Change, percentage points", fontsize=10)
    low = min(0, *(value[1] for series in values for value in series))
    high = max(0, *(value[2] for series in values for value in series))
    span = max(high - low, 0.01)
    ax.set_ylim(low - 0.22 * span, high + 0.22 * span)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.grid(axis="y", alpha=0.18, zorder=0)


def draw_comparison(output, studies):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
    for ax, (title, comparison, data) in zip(axes, studies):
        labels, coverage, placement, _ = data
        width = 0.34
        x = list(range(len(labels)))
        precision = 2 if title.startswith("Chile") else 1
        bars(ax, [v - width / 2 for v in x], coverage, width, BLUE,
             "Reference-option coverage", precision)
        bars(ax, [v + width / 2 for v in x], placement, width, ORANGE,
             "Placed", precision)
        finish_axes(ax, labels, title, comparison, [coverage, placement])
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper left", bbox_to_anchor=(0.025, 0.92),
               frameon=False, ncol=2, fontsize=11)
    fig.suptitle("Shortlist coverage and placement under each policy comparison",
                 x=0.03, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.03, 0.11,
             "Chile coverage targets the observed April school; placements are counterfactual replays. Both measures use register-linked applicants.\n"
             "NYC coverage targets the full-information partner among people with such a match; placement uses everyone. Preferences and allocations are simulated.\n"
             "Panels have separate scales. NYC whiskers are reported 95% intervals. Rental coverage resamples markets; all other intervals resample seed blocks.",
             ha="left", va="top", fontsize=9, color=GREY, linespacing=1.5)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.73, bottom=0.24, wspace=0.35)
    fig.savefig(output / "three_way_accuracy_placement.png", dpi=170)
    plt.close(fig)


def draw_groups(output, studies):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
    names = (("No initial offer", "Had an initial offer"),
             ("Higher income", "Lower income"),
             ("Lower wage", "Higher wage"))
    for ax, (title, comparison, data), group_names in zip(axes, studies, names):
        labels, _, _, groups = data
        x = list(range(len(labels)))
        width = 0.34
        precision = 2 if title.startswith("Chile") else 1
        for offset, values, color, name in zip((-width / 2, width / 2), groups,
                                               (ORANGE, LIGHT), group_names):
            bars(ax, [v + offset for v in x], values, width, color, name, precision)
        finish_axes(ax, labels, title, comparison, groups)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", y=1.34, pad=0)
        ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.12),
                  frameon=False, ncol=1, fontsize=9.5, borderpad=0)
    fig.suptitle("Which groups lose placements?",
                 x=0.03, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.03, 0.11,
             "Chile groups use historical initial-offer status among register-linked applicants. NYC groups use the study's income or wage definition.\n"
             "Each difference uses its own group's denominator; panel scales differ. Chile placements are counterfactual replays; NYC outcomes are simulated.\n"
             "NYC whiskers are reported 95% seed-block intervals. Chile counts are exact for these cohorts.",
             ha="left", va="top", fontsize=9, color=GREY, linespacing=1.5)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.69, bottom=0.24, wspace=0.35)
    fig.savefig(output / "three_way_who_loses.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=FINDINGS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    studies = [
        ("Chilean school admissions", "Rank-informed minus rank-blind", load_chile()),
        ("NYC rentals · enriched design", "GPT minus random lists", load_rentals()),
        ("NYC jobs · primary design", "GPT minus random lists", load_jobs()),
    ]
    draw_comparison(args.output, studies)
    draw_groups(args.output, studies)
    print(f"Wrote the two three-setting charts to {args.output}")


if __name__ == "__main__":
    main()
