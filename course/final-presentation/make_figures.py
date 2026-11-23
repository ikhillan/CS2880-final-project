"""Generate presentation figures and tables from authenticated aggregate rows."""
import csv
import hashlib
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "figures"
BLUE, RUST, GREY = "#2868AE", "#B04B2C", "#626A71"
sources = {}


def rows(relative):
    path = REPO / relative
    sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def source_text(relative):
    path = REPO / relative
    content = path.read_bytes()
    sources[relative] = hashlib.sha256(content).hexdigest()
    return content.decode()


def one(table, **keys):
    matches = [r for r in table if all(r[k] == v for k, v in keys.items())]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for {keys}, got {len(matches)}")
    return matches[0]


def clean(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(OUT / Path(name).with_suffix(".png"), bbox_inches="tight", dpi=170)
    plt.close(fig)


def rentals():
    # Both metrics use the paired market-bootstrap intervals released here.
    # The source rounds all percentage-point estimates to one decimal place.
    lines = source_text("findings/gpt-luna-rentals/accuracy.txt").splitlines()
    number = r"[+-]?\d+(?:\.\d+)?"
    pattern = re.compile(
        rf"gpt_k5:.*?\(d acc ({number}) \[({number}),({number})\], "
        rf"d placed ({number}) \[({number}),({number})\]\)"
    )
    windows = (("stage1_pilot", "Jun–Aug 2026"),
               ("heldout_primary", "Sep–Oct 2026"),
               ("heldout_replication", "Jun–Aug 2025"))
    fig, ax = plt.subplots(figsize=(9.3, 2.8))
    for y, (window, _) in zip((2, 1, 0), windows):
        matching = [line for line in lines if line.split()[:2] == [window, "enriched"]]
        if len(matching) != 1 or (match := pattern.search(matching[0])) is None:
            raise ValueError(f"Missing or ambiguous rental accuracy record for {window}")
        values = list(map(float, match.groups()))
        for (point, low, high), color, label in (
            (values[:3], BLUE, "Reference-option coverage"),
            (values[3:], RUST, "Placement"),
        ):
            if not low <= point <= high:
                raise ValueError(f"Rental interval excludes estimate for {window} {label}")
            ax.errorbar(point, y, xerr=[[point-low], [high-point]], fmt="o",
                        color=color, ms=7, capsize=4, lw=1.5,
                        label=label if y == 2 else None, zorder=3)
            ax.text(point, y+.19, f"{point:+.1f}", ha="center", va="bottom",
                    color=color, fontsize=13)
    ax.axvline(0, color=GREY, ls="--", lw=1)
    ax.set_yticks([2, 1, 0], [label for _, label in windows])
    ax.set_ylim(-.45, 2.55)
    ax.set_xlim(-5.5, 10.5)
    ax.set_xticks([-4, 0, 4, 8])
    ax.set_xlabel("Difference from random menus (percentage points; 0 = random)", fontsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.23), ncol=2, frameon=False,
              fontsize=12)
    ax.xaxis.grid(True, color="#eeeeee")
    clean(ax)
    save(fig, "rentals.pdf")


def anchors():
    fig, ax = plt.subplots(figsize=(9.3, 2.2))
    for y, suffix in enumerate(("replication/", "")):
        base = "findings/jobs-gpt-frozen/" + suffix
        pair = one(rows(base + "accuracy_pairs.csv"), rule="GPT (5 jobs)",
                   baseline="Random (5 jobs)", group="all")
        anchor = one(rows(base + "anchor.csv"), variant="primary", rule="GPT (5 jobs)")
        random_anchor = one(rows(base + "anchor.csv"), variant="primary", rule="Random (5 jobs)")
        if suffix == "replication/":
            rates = {"jobsBefore": anchor["placed"], "jobsAfter": anchor["anchor_placed"],
                     "jobsRandomAfter": random_anchor["anchor_placed"]}
            (HERE / "data/jobs-main-macros.tex").write_text("".join(
                f"\\newcommand{{\\{name}}}{{{float(value):.1f}}}\n" for name, value in rates.items()))
        before = [100 * float(pair["d_placed" + s]) for s in ("", "_lo", "_hi")]
        after = [float(anchor["anchored_vs_random" + s]) for s in ("", "_lo", "_hi")]
        assert abs(after[0] - before[0] - float(anchor["gain"])) < 1e-8
        ax.plot([before[0], after[0]], [y, y], color="#CCD3D9", lw=3, zorder=1)
        for values, color, label in ((before, RUST, "GPT"), (after, BLUE, "GPT + anchor")):
            p, lo, hi = values
            ax.errorbar(p, y, xerr=[[p-lo], [hi-p]], fmt="o", color=color, ms=7,
                        capsize=4, lw=1.4, label=label if y == 1 else None, zorder=3)
            ax.text(p, y-.22, f"{p:.1f}", ha="center", color=color, fontsize=12)
        ax.text((before[0]+after[0])/2, y+.18, f"+{float(anchor['gain']):.1f} pp",
                ha="center", fontsize=12)
        p, lo, hi = [float(random_anchor["gain" + suffix]) for suffix in ("", "_lo", "_hi")]
        ax.errorbar(p, y, xerr=[[p-lo], [hi-p]], fmt="D", color=GREY, ms=6,
                    capsize=4, lw=1.4, label="Random + anchor" if y == 1 else None, zorder=3)
        ax.text(p, y-.22, f"+{p:.1f}", ha="center", color=GREY, fontsize=12)
    ax.axvline(0, color=GREY, ls="--", lw=1)
    ax.set_yticks([0, 1], ["Replication", "Frozen"])
    ax.set_ylim(-.55, 1.6)
    ax.set_xlim(-21, 2)
    ax.set_xticks([-20, -15, -10, -5, 0])
    ax.set_xlabel("Placement difference (pp; 0 = random)", fontsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.25), ncol=3, frameon=False)
    clean(ax)
    save(fig, "jobs-anchor.pdf")


def incidence():
    table = rows("findings/chile/group_comparison.csv")
    fig, ax = plt.subplots(figsize=(9.3, 2.5))
    labels = []
    for y, group, name, color in (
        (1, "no_initial_offer", "No initial offer", RUST),
        (0, "initial_offer", "Had an initial offer", GREY),
    ):
        row = one(table, year="total", comparison="rank_vs_blind", group=group)
        n = int(row["linked_applicant_years"])
        change = int(row["rank_placed"]) - int(row["baseline_placed"])
        if change != int(row["placement_change"]):
            raise ValueError(f"Chile placement accounting disagrees for {group}")
        value = 100 * change / n
        if abs(value - float(row["placement_change_pp"])) > 1e-8:
            raise ValueError(f"Chile placement rate disagrees for {group}")
        labels.append(f"{name}\nn = {n:,}")
        ax.barh(y, value, color=color, height=.32, zorder=2)
        ax.text(value-.025, y, f"{value:.3f}".replace("-", "−"),
                va="center", ha="right", fontsize=15, color=color)
    ax.axvline(0, color=GREY, lw=1)
    ax.set_yticks([1, 0], labels, fontsize=13)
    ax.set_xlim(-1.05, .10)
    ax.set_ylim(-.5, 1.5)
    ax.set_xticks([-1, -.75, -.5, -.25, 0])
    ax.set_xlabel("Within-group placement change: rank-informed − rank-blind (pp)", fontsize=12)
    ax.xaxis.grid(True, color="#eeeeee")
    clean(ax)
    save(fig, "chile-incidence.pdf")


def retention():
    total = one(rows("findings/chile/retention_bounds.csv"), year="total")
    exact, lower, upper = [int(total[k]) for k in
                          ("exact_contract_gain", "attained_retention_gain", "retention_upper")]
    fig, ax = plt.subplots(figsize=(9.5, 2.9))
    ax.scatter([exact], [1], color=GREY, s=65, zorder=3)
    ax.text(exact, 1.19, f"{exact:,} exact", ha="center", fontsize=14)
    ax.plot([lower, upper], [0, 0], lw=4, color="#ABC6E4")
    ax.scatter([lower], [0], color=BLUE, s=65, zorder=3)
    ax.plot([upper, upper], [-.08, .08], lw=2, color=BLUE)
    ax.text(lower, -.22, f"{lower:,} achieved", ha="center", color=BLUE, fontsize=14)
    ax.text(upper, -.22, f"{upper:,} upper bound", ha="center", color=BLUE, fontsize=14)
    ax.text((lower+upper)/2, .17, "Remaining interval for the optimum", ha="center", fontsize=12)
    ax.set_yticks([1, 0], ["Same course\nand seat type", "Seat retained\nschool may change"])
    ax.set_xlim(0, upper*1.12)
    ax.set_ylim(-.5, 1.6)
    ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
    ax.set_xlabel("Additional historical-school hits", fontsize=13)
    ax.xaxis.grid(True, color="#eeeeee")
    clean(ax)
    save(fig, "retention.pdf")


def access_tables():
    """Generate the access audit from released rows, with explicit comparators."""
    base = "findings/rental-anchor-access/"
    primary = rows(base + "primary.csv")
    transitions = rows(base + "transitions.csv")
    contrasts = rows(base + "contrasts.csv")
    levels = rows(base + "levels.csv")
    data = HERE / "data"
    data.mkdir(exist_ok=True)
    pairs = [("anchor_gpt", "gpt", r"GPT $\to$ Anchor"),
             ("balanced_gpt", "anchor_gpt", r"Anchor $\to$ Balanced")]

    def delta(window, variant, arm, comparator, metric):
        r = one(primary, window=window, variant=variant, group="all",
                arm=arm, comparator=comparator, metric=metric)
        assert r["status"] == "ok" and r["simultaneous_family_complete"] == "True"
        return [100*float(r[k]) for k in ("mean", "simultaneous_lo", "simultaneous_hi")]

    def exits(window, variant, arm, comparator):
        r = one(transitions, window=window, variant=variant, group="all", arm=arm,
                baseline=comparator, metric="gross_exits")
        count = float(r["numerator_total"])
        assert count.is_integer()
        return int(count)

    main = [r"\begin{tabularx}{\textwidth}{@{}Xr@{\hspace{5mm}}r@{\hspace{5mm}}r@{}}\toprule",
            r"Change & Coverage & Placed & Exits\\\midrule"]
    for arm, comparator, label in pairs:
        c = delta("heldout_primary", "enriched", arm, comparator, "coverage_market_mean")[0]
        p = delta("heldout_primary", "enriched", arm, comparator, "placed")[0]
        x = exits("heldout_primary", "enriched", arm, comparator)
        main.append(f"{label} & ${c:+.1f}$ pp & ${p:+.1f}$ pp & {x} " + r"\\")
    main.append(r"\bottomrule\end{tabularx}")
    (data / "rental-access-main.tex").write_text("\n".join(main)+"\n")

    for arm, comparator, _ in pairs:
        table = [r"\begin{tabular}{@{}llrrr@{}}\toprule",
                 r"Design & Window & $\Delta$ coverage & $\Delta$ placement & Exits\\\midrule"]
        for variant, design in (("primary", "Ordinary"), ("enriched", "Enriched")):
            for window, name in (("stage1_pilot", "Development"),
                                 ("heldout_primary", "Held out"),
                                 ("heldout_replication", "Replication")):
                vals = [delta(window, variant, arm, comparator, m)
                        for m in ("coverage_market_mean", "placed")]
                formatted = [f"${p:+.2f}$ [{lo:.2f}, {hi:.2f}]" for p, lo, hi in vals]
                table.append(" & ".join([design, name, *formatted,
                             str(exits(window, variant, arm, comparator))])+r"\\")
        table.append(r"\bottomrule\end{tabular}")
        (data / f"rental-access-{arm}.tex").write_text("\n".join(table)+"\n")

    controls = [r"\begin{tabular}{@{}lrr@{}}\toprule",
                r"Policy & Coverage (\%) & Placement (\%)\\\midrule"]
    for arm, label in (("random", "Random"), ("gpt", "GPT"), ("score", "Model-informed score"),
                       ("anchor_random", "Anchor + random"), ("anchor_gpt", "Anchor + GPT"),
                       ("anchor_score", "Anchor + score"), ("balanced_gpt", "Balanced + GPT"),
                       ("balanced_score", "Balanced + score"), ("score_anchor_gpt", "Score anchor + GPT")):
        vals = [100*float(one(levels, window="heldout_primary", variant="enriched",
                             group="all", arm=arm, metric=m)["mean"])
                for m in ("coverage_market_mean", "placed")]
        controls.append(f"{label} & {vals[0]:.2f} & {vals[1]:.2f} " + r"\\")
    controls.append(r"\bottomrule\end{tabular}")
    (data / "rental-access-controls.tex").write_text("\n".join(controls)+"\n")

    spill = one(contrasts, window="heldout_primary", variant="enriched", group="degree_1_5",
                arm="balanced_gpt", comparator="anchor_gpt", metric="placed")
    assert float(one(contrasts, window="heldout_primary", variant="enriched", group="degree_1_5",
                     arm="balanced_gpt", comparator="anchor_gpt", metric="coverage_market_mean")["mean"]) == 0
    macros = {"unchangedMenuGain": f"{100*float(spill['mean']):.1f}"}
    for arm, name in (("anchor_gpt", "unchangedBefore"), ("balanced_gpt", "unchangedAfter")):
        r = one(levels, window="heldout_primary", variant="enriched", group="degree_1_5",
                arm=arm, metric="placed")
        macros[name] = f"{100*float(r['mean']):.1f}"
    subgroup_counts = {}
    for metric, name in (("baseline_placed", "subgroupBeforeCount"),
                         ("after_placed", "subgroupAfterCount"),
                         ("gross_entrants", "subgroupEntrants"),
                         ("gross_exits", "subgroupExits")):
        r = one(transitions, window="heldout_primary", variant="enriched",
                group="degree_1_5", arm="balanced_gpt", baseline="anchor_gpt",
                metric=metric)
        count = float(r["numerator_total"])
        assert r["status"] == "ok" and count.is_integer() and count >= 0
        subgroup_counts[name] = int(count)
        macros[name] = str(int(count))
    assert (subgroup_counts["subgroupBeforeCount"]
            - subgroup_counts["subgroupExits"]
            + subgroup_counts["subgroupEntrants"]
            == subgroup_counts["subgroupAfterCount"])
    # Paired seed-rate contrasts for the same subgroup used in the headline.
    # Counts are pooled appearances; the rate columns weight the 20 seeds equally.
    subgroup_rows = []
    for variant, design in (("primary", "Ordinary"), ("enriched", "Enriched")):
        gains = []
        for window, name, macro in (("stage1_pilot", "Jun--Aug 2026", "subgroupGainDev"),
                                    ("heldout_primary", "Sep--Oct 2026", "subgroupGainPrimary"),
                                    ("heldout_replication", "Jun--Aug 2025", "subgroupGainRep")):
            cell = dict(window=window, variant=variant, group="degree_1_5", arm="balanced_gpt")
            paired = one(contrasts, **cell, comparator="anchor_gpt", metric="placed")
            assert paired["status"] == "ok" and paired["estimand"] == "mean_of_seed_rates"
            assert paired["seeds_available"] == paired["expected_seeds"] == "20"
            assert paired["paired_markets"] == paired["markets_expected"]
            assert paired["unpaired_markets"] == paired["missing_markets"] == "0"
            assert paired["valid_bootstrap_draws"] == paired["bootstrap_draws"] == "10000"
            before = one(transitions, **cell, baseline="anchor_gpt", metric="baseline_placed")
            after = one(transitions, **cell, baseline="anchor_gpt", metric="after_placed")
            assert before["denominator_total"] == after["denominator_total"]
            n = float(before["denominator_total"])
            assert n.is_integer() and n > 0
            gain, low, high = [100 * float(paired[k]) for k in ("mean", "lo", "hi")]
            old, new = [100 * float(row["mean"]) for row in (before, after)]
            assert abs(gain - (new - old)) < 1e-8
            gains.append(gain)
            if variant == "enriched":
                macros[macro] = f"{gain:.1f}"
            subgroup_rows.append(f"{design} & {name} & {int(n):,} & {old:.1f} & {new:.1f} & "
                                 f"$+{gain:.1f}$ [{low:.1f}, {high:.1f}]" + r"\\")
        macros[f"{design.lower()}FixedGainLow"] = f"{min(gains):.1f}"
        macros[f"{design.lower()}FixedGainHigh"] = f"{max(gains):.1f}"
    macros["fixedMenuPanelRows"] = "\n".join(subgroup_rows)
    for arm, comparator, prefix in (("anchor_gpt", "gpt", "anchor"),
                                    ("balanced_gpt", "anchor_gpt", "balanced")):
        for metric, suffix in (("gross_exits", "Exits"), ("gross_entrants", "Entrants")):
            r = one(transitions, window="heldout_primary", variant="enriched", group="all",
                    arm=arm, baseline=comparator, metric=metric)
            macros[prefix+suffix] = str(int(float(r["numerator_total"])))
        macros[prefix+"Net"] = str(int(macros[prefix+"Entrants"])-int(macros[prefix+"Exits"]))
    (data / "rental-access-macros.tex").write_text("".join(
        f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))


def policy_comparison():
    """A four-policy comparison, without claiming a fitted frontier."""
    table = rows("findings/rental-anchor-access/levels.csv")
    fig, ax = plt.subplots(figsize=(9.3, 2.7))
    for arm, label, color, dx, dy in (
        ("random", "Random", GREY, .3, -.2),
        ("gpt", "GPT", RUST, .28, -.2),
        ("anchor_gpt", "One anchor + GPT", BLUE, .28, .05),
        ("balanced_gpt", "Balanced + GPT", "#1A7D78", .3, -.2),
    ):
        x, y = [100*float(one(table, window="heldout_primary", variant="enriched",
                              group="all", arm=arm, metric=m)["mean"])
                for m in ("coverage_market_mean", "placed")]
        ax.scatter([x], [y], color=color, s=62, zorder=3)
        ax.text(x+dx, y+dy, label, color=color, fontsize=13, va="center")
    ax.set_xlabel("Reference-option coverage (%)", fontsize=13)
    ax.set_ylabel("Placed (%)", fontsize=13)
    ax.set_xlim(14.4, 28.5)
    ax.set_ylim(59.5, 83)
    ax.set_xticks([16, 20, 24, 28])
    ax.set_yticks([60, 65, 70, 75, 80])
    ax.grid(True, color="#e9edef", lw=.8)
    clean(ax)
    save(fig, "rental-policies.pdf")


def chile_tables():
    data = HERE / "data"
    total = one(rows("findings/chile/retention_bounds.csv"), year="total")
    prospective = rows("findings/chile/prospective.csv")
    macros = {"exactGain": total["exact_contract_gain"],
              "retentionGain": total["attained_retention_gain"],
              "retentionUpper": f"{int(total['retention_upper']):,}"}
    for policy, prefix in (("structural", "structural"), ("learned", "learned")):
        c = one(prospective, year="total", rule=policy, lottery="course")
        f = one(prospective, year="total", rule=policy, lottery="family")
        assert c["net_historical_hit_change"] == f["net_historical_hit_change"]
        macros[prefix+"Gain"] = c["net_historical_hit_change"]
        macros[prefix+"CourseExits"] = c["incumbent_exit_events"]
        macros[prefix+"FamilyExits"] = f["incumbent_exit_events"]
    field = one(rows("findings/chile/field_cohort.csv"), year="2021")
    linked = int(field["linked_applicant_years"])
    assert 0 < linked <= int(field["entry_grade_applicant_years"])
    assert 0 <= int(field["changed_menu_applicant_years"]) <= linked
    total_change = int(field["simulated_placement_change"])
    unchanged_change = int(field["simulated_unchanged_menu_placement_change"])
    assert total_change <= unchanged_change < 0
    # Positive magnitudes of net placement declines, not gross exits.
    macros["fieldUnchangedLoss"] = str(-unchanged_change)
    macros["fieldTotalLoss"] = str(-total_change)
    (data / "chile-main-macros.tex").write_text("".join(
        f"\\newcommand{{\\{k}}}{{{v}}}\n" for k,v in macros.items()))


def field_tables():
    base = "publication/two-studies/field-study/"
    trial = rows(base + "trials/explorer-contrasts.csv")
    macros = {}
    for outcome, name in [("submitted_application", "submission"),
                          ("second_ranked_highlight_worthy", "secondChoice"), ("assigned", "anySeat")]:
        row = one(trial, contrast="maps_plus_info_vs_control", outcome=outcome)
        macros[name] = f"{100 * float(row['estimate']):.1f}"
        macros[name + "SE"] = f"{100 * float(row['se_upper']):.1f}"
        macros[name + "N"] = f"{int(row['reported_n']):,}"
    toy = rows(base + "interference/interference-market.csv")
    for arm, key in [("0", "oldPlaces"), ("1", "newPlaces")]:
        row = one(toy, a_arm=arm, b_arm=arm, c_arm=arm)
        macros[key] = row["total"]
    enumeration = json.loads(source_text(base + "interference/scalar.json"))
    macros["directPositive"] = f"{enumeration['positive_direct_and_rollout_loss']:,}"
    macros["rolloutLoss"] = f"{enumeration['negative_rollout']:,}"
    (HERE / "data/field-main-macros.tex").write_text("".join(
        f"\\newcommand{{\\{name}}}{{{value}}}\n" for name, value in macros.items()))


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    rentals()
    anchors()
    incidence()
    retention()
    access_tables()
    policy_comparison()
    chile_tables()
    field_tables()
    (HERE / "figure-sources.json").write_text(json.dumps(sources, indent=2, sort_keys=True)+"\n")
    print("Built five figures and presentation tables from", len(sources),
          "aggregate sources, with PDF and PNG versions.")


if __name__ == "__main__":
    main()
