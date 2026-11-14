"""Calculate directional proxy tests and contrasts from published rounded estimates."""
import argparse
import csv

from pathlib import Path
from statistics import NormalDist

NORMAL = NormalDist()
Z = NORMAL.inv_cdf(.975)

def read_table(path):
    with path.open() as f:
        return {(r["arm"], r["outcome"]): r for r in csv.DictReader(f)}

def signed_outcome(table, arm, var):
    name = "not_assigned" if var == "assigned" else var
    row = table[arm, name]
    sign = -1 if var == "assigned" else 1
    return sign*float(row["estimate"]), float(row["se"]), int(row["n"])

def rows(table):
    outcomes = ("submitted_application","first_ranked_highlight_worthy","second_ranked_highlight_worthy",
                "assigned","enrolled_at_assigned_school","enrolled_highlight_worthy")
    for contrast in ("maps_plus_info_vs_control","maps_plus_info_vs_info_only"):
        for var in outcomes:
            b2,s2,n = signed_outcome(table,"aggregate_and_highlights",var)
            if contrast == "maps_plus_info_vs_control":
                effect, se_lower, se_upper = b2,s2,s2
            else:
                b1,s1,n1 = signed_outcome(table,"aggregate",var)
                assert n == n1
                effect = b2-b1
                se_lower, se_upper = abs(s2-s1), s2+s1
            yield [contrast,var,n,effect,se_lower,se_upper,effect-Z*se_upper,effect+Z*se_upper]

def verify(table, out):
    contrasts = list(rows(table))
    by_key = {(r[0],r[1]):r for r in contrasts}
    r2 = by_key["maps_plus_info_vs_control", "second_ranked_highlight_worthy"]
    allocation = by_key["maps_plus_info_vs_control", "assigned"]
    # H1 is an intersection. Reject only when both one-sided component tests reject.
    p_ranking = NORMAL.cdf(-r2[3]/r2[5])
    p_placement_loss = NORMAL.cdf(allocation[3]/allocation[5])
    # Nuisance correlations for incremental contrasts can vary anywhere in [-1,1].
    # Here both incremental proxies have favorable signs, so using the largest s.e.
    # produces the largest one-sided p-values (rounded point estimates).
    inc_r = by_key["maps_plus_info_vs_info_only","second_ranked_highlight_worthy"]
    inc_p = by_key["maps_plus_info_vs_info_only","assigned"]
    assert inc_r[3]>0 and inc_p[3]<0
    p_r_upper=NORMAL.cdf(-inc_r[3]/inc_r[5]); p_p_upper=NORMAL.cdf(inc_p[3]/inc_p[5])
    assert round(r2[3],3)==.054 and round(allocation[3],3)==.002
    assert round(inc_r[3],3)==.038 and round(inc_p[3],3)==-.010
    assert round(max(p_ranking,p_placement_loss),3)==.572
    with out.open("w",newline="") as f:
        writer=csv.writer(f,lineterminator="\n")
        writer.writerow(["contrast","outcome","reported_n","estimate","se_lower","se_upper",
                         "normal95_lower_using_se_upper","normal95_upper_using_se_upper"])
        for contrast,var,n,*values in contrasts:
            writer.writerow([contrast,var,n,*[f"{v:.8f}" for v in values]])
    print(f"Map and information vs control, one-sided proxy component p-values: {p_ranking:.6f}, {p_placement_loss:.6f}; intersection-union p={max(p_ranking,p_placement_loss):.6f}")
    print(f"Map and information vs information, largest one-sided component p-values over missing correlations: {p_r_upper:.6f}, {p_p_upper:.6f}; conservative intersection-union p={max(p_r_upper,p_p_upper):.6f}")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--moments",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args(); verify(read_table(a.moments),a.out)
