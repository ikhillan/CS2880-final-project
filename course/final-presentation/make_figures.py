"""Three presentation figures from committed aggregate evidence."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

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
    plt.close(fig)


def anchors():
    fig, ax = plt.subplots(figsize=(6.0, 3.1))
    for y, suffix in enumerate(("replication/", "")):
        base = "findings/jobs-gpt-frozen/" + suffix
        pair = one(rows(base + "accuracy_pairs.csv"), rule="GPT (5 jobs)",
                   baseline="Random (5 jobs)", group="all")
        anchor = one(rows(base + "anchor.csv"), variant="primary", rule="GPT (5 jobs)")
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
    ax.axvline(0, color=GREY, ls="--", lw=1)
    ax.set_yticks([0, 1], ["Replication", "Frozen"])
    ax.set_ylim(-.55, 1.6)
    ax.set_xlim(-21, 2)
    ax.set_xticks([-20, -15, -10, -5, 0])
    ax.set_xlabel("Placement difference from random (pp)", fontsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.23), ncol=2, frameon=False)
    clean(ax)
    save(fig, "jobs-anchor.pdf")


def incidence():
    table = rows("findings/chile/group_comparison.csv")
    group = one(table, year="total", comparison="rank_vs_blind", group="no_initial_offer")
    all_people = one(table, year="total", comparison="rank_vs_blind", group="all_linked")
    shares = [100*int(group["linked_applicant_years"])/int(all_people["linked_applicant_years"]),
              100*int(group["placement_change"])/int(all_people["placement_change"])]
    fig, ax = plt.subplots(figsize=(9.3, 2.2))
    ax.barh([1, 0], shares, color=[GREY, RUST], height=.38)
    ax.set_yticks([1, 0], ["Share of linked applicants", "Share of net placement loss"])
    for y, value in zip([1, 0], shares):
        ax.text(value+2, y, f"{value:.1f}%", va="center", fontsize=16,
                color=RUST if y == 0 else GREY)
    ax.set_xlim(0, 100)
    ax.set_ylim(-.5, 1.5)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.set_major_formatter(PercentFormatter())
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


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    anchors()
    incidence()
    retention()
    (HERE / "figure-sources.json").write_text(json.dumps(sources, indent=2, sort_keys=True)+"\n")
    print("Built three figures from", len(sources), "aggregate source tables.")


if __name__ == "__main__":
    main()
