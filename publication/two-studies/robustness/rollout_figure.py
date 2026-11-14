"""Plot replayed adoption and the endpoint interpolation in their native units."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent


def draw():
    nyc=pd.read_csv(HERE/"nyc/rental-rollout.csv")
    schools=pd.read_csv(HERE/"chile/school-rollout-summary.csv")
    fig, axes=plt.subplots(1,2,figsize=(6.45,2.75))
    colors=["#195c72","#aa472e"]
    line_styles=["-","-"]
    for ax,title,unit in zip(axes,("New York rentals","Chilean schools"),("Placement change (percentage points)","Placement change (pupils)")):
        ax.axhline(0,color="#bcc6ca",linewidth=.65)
        ax.spines[["top","right"]].set_visible(False)
        ax.set_title(title,fontsize=10,loc="left",fontweight="bold")
        ax.set_ylabel(unit,fontsize=8)
        ax.set_xlabel("Applicants assigned the new rule",fontsize=8)
        ax.set_xticks([0,.25,.5,.75,1],["0%","25%","50%","75%","100%"])
        ax.tick_params(axis="both",labelsize=7,length=2)
        ax.margins(x=.04,y=.12)
    for k,(design,label) in enumerate((("primary","Ordinary"),("enriched","Enriched"))):
        z=nyc[(nyc.design==design)&(nyc.metric=="placed")].sort_values("p")
        axes[0].plot(z.p,z.p*z.delta_pp.iloc[-1],color=colors[k],lw=1,linestyle="--",alpha=.65)
        axes[0].plot(z.p,z.delta_pp,color=colors[k],ls=line_styles[k],marker=["o","s"][k],ms=3.4,lw=1.7,label=label)
    for k,year in enumerate((2022,2023)):
        z=schools[(schools.year==year)&(schools.metric=="placed_delta")].sort_values("uptake")
        axes[1].plot(z.uptake,z.uptake*z["count"].iloc[-1],color=colors[k],lw=1,ls="--",alpha=.65)
        axes[1].plot(z.uptake,z["count"],color=colors[k],marker=["o","s"][k],ms=3.4,lw=1.6,label=str(year))
    for ax in axes:ax.legend(loc="lower left",frameon=False,fontsize=7)
    fig.tight_layout(pad=1.0,w_pad=1.9)
    fig.savefig(HERE.parent/"figures/adoption.pdf",metadata={"CreationDate":None,"ModDate":None})
    plt.close(fig)


if __name__=="__main__":
    draw()
