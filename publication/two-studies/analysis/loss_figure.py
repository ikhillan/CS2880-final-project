"""Generate the placement-loss breakdown from paired applicant counts in both studies."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter


def breakdown(housing, school):
    records = []
    for design, title in (("primary", "NYC: ordinary"), ("enriched", "NYC: enriched")):
        lost = housing[(housing.design == design) & (housing.group == "all") & (housing.delta_placement == -1)]
        counts = lost.groupby(["same_menu", "delta_coverage"])["count"].sum()
        vals = [counts.loc[1].sum(), *(counts.loc[0].get(i, 0) for i in (0, -1, 1))]
        records.append((title, vals))
    for year in (2022, 2023):
        lost = school[(school.year == year) & (school.reference == "full_DA") & (school.delta_placement == -1)]
        counts = lost.groupby(["group", "delta_coverage"])["count"].sum()
        vals = [counts.loc["same_menu"].sum(), *(counts.loc["changed_menu"].get(i, 0) for i in (0,-1,1))]
        assert sum(vals) == counts.loc["all"].sum()
        records.append((f"Chile: {year}", vals))
    return records


def main(root):
    parts = breakdown(pd.read_csv(root/"evidence/joint.csv"), pd.read_csv(root/"evidence/school-joint.csv"))
    colors = ["#164a60", "#adc4cf", "#eeb87c", "#9c4933"]
    cats = ["Same shortlist", "List changed, coverage unchanged", "Full-list partner newly hidden", "Full-list partner newly shown"]
    fig, ax = plt.subplots(figsize=(6.55, 2.63))
    ax.spines[["left","top","right"]].set_visible(False)
    for row, (_, counts) in enumerate(parts):
        base = 0
        for col, count in enumerate(counts):
            share = count/sum(counts)
            ax.barh(row, share, left=base, height=.60, color=colors[col], edgecolor="white", linewidth=.5)
            if share >= .063:
                ax.text(base+share/2,row,f"{share:.0%}",ha="center",va="center",fontsize=8,
                        color="white" if col in (0,3) else "#19262c")
            base += share
        ax.text(1.012, row, f"{sum(counts):,}", va="center", ha="left", fontsize=8, color="#24353b")
    ax.set_xlim(0,1);ax.set_ylim(3.43,-.51)
    ax.set_yticks(np.arange(len(parts)),[p[0] for p in parts],fontsize=8)
    ax.set_xticks([0,.25,.5,.75,1]);ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.tick_params(axis="x",labelsize=7,direction="out",length=2)
    ax.tick_params(axis="y",length=0,pad=7)
    ax.set_xlabel("Share of people who lost a placement", fontsize=8,labelpad=3)
    fig.subplots_adjust(left=.156,right=.915,top=.84,bottom=.30)
    fig.legend([Patch(facecolor=c) for c in colors],cats,loc="lower center",ncol=2,
               frameon=False,fontsize=8,handlelength=1.0,columnspacing=2.1)
    ax.text(1.012,-.55,"Losses",fontsize=8,ha="left",fontweight="bold")
    fig.savefig(root/"figures/placement-losses.pdf",metadata={"CreationDate":None,"ModDate":None})
    plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--root",type=Path,
            default=Path(__file__).resolve().parents[1]);main(parser.parse_args().root)
