"""Draw the Chile hindsight bound from the verified aggregate table."""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / "findings" / "chile"
    with (folder / "retention_bounds.csv").open(newline="") as stream:
        total, = [row for row in csv.DictReader(stream) if row["year"] == "total"]
    exact = int(total["exact_contract_gain"])
    lower = int(total["attained_retention_gain"])
    upper = int(total["retention_upper"])
    assert exact <= lower <= upper

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    fig, ax = plt.subplots(figsize=(12.8, 5.1))
    fig.subplots_adjust(left=.24, right=.96, top=.70, bottom=.25)
    fig.text(.035, .93, "Keeping a seat allows more gains than keeping the same assignment",
             weight="bold", fontsize=16)
    fig.text(.035, .86, "Historical school coverage: the achievable gain and the remaining optimization gap",
             fontsize=12, color="#555555")

    ax.scatter([exact], [1], s=85, color="#526777", zorder=3)
    ax.text(exact, 1.17, f"{exact:,} exact optimum", ha="center", fontsize=12)
    ax.plot([lower, upper], [0, 0], lw=5, color="#a8c5e6", solid_capstyle="butt")
    ax.scatter([lower], [0], s=85, color="#2868ae", zorder=3)
    ax.plot([upper, upper], [-.09, .09], color="#2868ae", lw=2)
    ax.text(lower, -.22, f"{lower:,} attained", ha="center", fontsize=12, color="#20558f")
    ax.text(upper, -.22, f"{upper:,} upper bound", ha="center", fontsize=12, color="#20558f")
    ax.text((lower + upper)/2, .17, "Optimum remains within this interval",
            ha="center", color="#555555", fontsize=11)
    ax.set_yticks([1, 0], ["Same course\nand seat type", "Seat retained;\nschool may change"])
    ax.set_xlim(0, upper * 1.10)
    ax.set_ylim(-.5, 1.5)
    ax.set_xlabel("Additional April-school hits over the rank-blind baseline", labelpad=10)
    ax.xaxis.grid(True, color="#eeeeee")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=16)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.text(.035, .07, "2022–24; fixed common/union menus; 1,024 design lotteries per cohort. Gains use historical April labels.\n"
             "This is a hindsight optimization bound, not a confidence interval or an effect of deploying recommendations.",
             fontsize=10, color="#555555", linespacing=1.5)
    output = folder / "retention_bounds.png"
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
