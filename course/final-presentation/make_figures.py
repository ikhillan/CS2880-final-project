"""Three main presentation figures and one appendix figure from released evidence."""
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
    windows = (("stage1_pilot", "Development 2026"),
               ("heldout_primary", "Held out 2026"),
               ("heldout_replication", "Replication 2025"))
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
    ax.set_xlabel("Placement difference (pp; 0 = random)", fontsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.23), ncol=2, frameon=False)
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


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    rentals()
    anchors()
    incidence()
    retention()
    (HERE / "figure-sources.json").write_text(json.dumps(sources, indent=2, sort_keys=True)+"\n")
    print("Built three main figures and one appendix figure from", len(sources),
          "aggregate sources, with PDF and PNG versions.")


if __name__ == "__main__":
    main()
