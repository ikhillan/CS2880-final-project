"""Three-way comparison charts for findings/README.md (Chile schools, NYC rentals, NYC jobs).

Values are copied from committed tables:
  Chile: publication/research/observed/school/summary.csv (k=5, rank_minus_blind rows)
  Rentals: findings/gpt-luna-rentals/accuracy.txt (enriched, gpt_k5 vs uniform_k5) and
           findings/rental-capacity-gpt/*/capacity_levels.csv (placed_lower / placed_higher)
  Jobs: findings/jobs-gpt-frozen/{,replication/}accuracy.csv and groups (primary, llm_k5 vs uniform_k5)
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]
BLUE, ORANGE, LIGHT, GREY = "#2E6FBF", "#C8571B", "#E9A27C", "#555555"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False})

PANELS = [
    ("Chilean schools (observed)", "rank-informed vs rank-blind classifier", ["2020", "2021"]),
    ("NYC rentals (simulated)", "GPT vs random lists", ["dev 2026", "held out", "repl. 2025"]),
    ("NYC jobs (simulated)", "GPT vs random lists", ["frozen", "replication"]),
]


def label(ax, bars, vals, fmt):
    for b, v in zip(bars, vals):
        y = b.get_height()
        ax.annotate((fmt or ("{:+.2f}" if abs(v) < 1 else "{:+.1f}")).format(v), (b.get_x() + b.get_width() / 2, y), xytext=(0, 3 if y >= 0 else -3),
                    textcoords="offset points", ha="center", va="bottom" if y >= 0 else "top", fontsize=9.5)


def panels(title, data, series, fname, note):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, (name, sub, groups), vals in zip(axes, PANELS, data):
        w = 0.8 / len(series)
        for j, (lab, col, fmt) in enumerate(series):
            xs = [i + (j - (len(series) - 1) / 2) * w for i in range(len(groups))]
            label(ax, ax.bar(xs, vals[j], w * 0.92, color=col, label=lab), vals[j], fmt)
        ax.axhline(0, color=GREY, lw=0.8)
        ax.set_xticks(range(len(groups)), groups)
        ax.set_title(f"{name}\n", fontsize=12, fontweight="bold", loc="left")
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=10, color=GREY)
        lo, hi = ax.get_ylim()
        pad = (hi - lo) * 0.12
        ax.set_ylim(lo - pad, hi + pad)
    axes[0].set_ylabel("Change, percentage points")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", ncol=len(series), frameon=False, bbox_to_anchor=(0.005, 0.95))
    fig.suptitle(title, x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.01, note, fontsize=9, color=GREY)
    fig.tight_layout(rect=(0, 0.04, 1, 0.88))
    fig.savefig(f"{OUT}/{fname}", dpi=150)
    plt.close(fig)


# Chart 1: accuracy vs placement, rule B minus rule A
panels("The smarter rule places fewer people in all three settings, with equal or better accuracy",
       [[[0.32, 0.33], [-0.12, -0.11]],
        [[8.0, 8.1, 7.9], [-2.0, -3.7, -3.3]],
        [[1.6, 0.8], [-15.5, -17.5]]],
       [("Accuracy (best option on the list)", BLUE, None), ("Placed", ORANGE, None)],
       "three_way_accuracy_placement.png",
       "Each panel has its own scale. 5 options per person. Chile accuracy = official offer on the list; NYC = full-information match on the list. "
       "Rentals: enriched design; jobs: primary design.")

# Chart 2: placement change for the group with more options vs fewer options
fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
GROUPS = [
    (["More than 5\nchoices", "5 or fewer\n(list unchanged)"], [[-0.64, -0.52], [-0.07, -0.08]], ["2020", "2021"]),
    (["Higher-income\n(~46 options)", "Lower-income\n(~14 options)"], [[-5.6, -6.7, -6.0], [1.6, -0.7, -0.5]], ["dev", "held out", "repl."]),
    (["Lower-wage\n(~21 options)", "Higher-wage\n(~11 options)"], [[-22.6, -23.0], [-8.5, -12.0]], ["frozen", "repl."]),
]
for ax, (name, sub, _), (glabs, vals, wins) in zip(axes, PANELS, GROUPS):
    n = len(wins)
    w = 0.8 / n
    for g in range(2):
        for k in range(n):
            x = g + (k - (n - 1) / 2) * w
            b = ax.bar(x, vals[g][k], w * 0.92, color=ORANGE if g == 0 else LIGHT)
            label(ax, b, [vals[g][k]], None)
            ax.annotate(wins[k], (x, 0), xytext=(0, 4 if vals[g][k] < 0 else -12), textcoords="offset points",
                        ha="center", fontsize=8, color=GREY)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xticks([0, 1], glabs)
    ax.set_title(f"{name}\n", fontsize=12, fontweight="bold", loc="left")
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=10, color=GREY)
    lo, hi = ax.get_ylim()
    pad = (hi - lo) * 0.12
    ax.set_ylim(lo - pad, max(hi, 0) + pad)
axes[0].set_ylabel("Change in share placed, pp")
fig.suptitle("The placement loss lands on the group with more options", x=0.01, ha="left", fontsize=14, fontweight="bold")
fig.text(0.01, 0.01, "Each panel has its own scale. Dark = group with more feasible options; light = fewer. "
         "Chile groups are by list length, not income.", fontsize=9, color=GREY)
fig.tight_layout(rect=(0, 0.04, 1, 0.93))
fig.savefig(f"{OUT}/three_way_who_loses.png", dpi=150)
plt.close(fig)
print("ok")
