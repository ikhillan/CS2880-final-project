"""Compare the released paired estimator to exact own-treatment effects on a DA example."""
from fractions import Fraction as F
from itertools import product
from math import isclose
from summarize_mixed import GRID, pair_outcomes


def placed(z):
    # a and b choose x if untreated, y otherwise. c always chooses y.
    # x priority a > b, y priority b > a > c. Both schools have one seat.
    a, b = z
    return (1 - a*b, int(a or b), int(a == b == 0))


baseline = placed((0, 0))
for k in GRID:
    p = F(k, 1000)
    # Single-person masses for the two antithetic assignments from one Uniform(0,1).
    boundaries = sorted({F(0), p, 1-p, F(1)})
    pairs = []
    for low, high in zip(boundaries,boundaries[1:]):
        u = (low+high)/2
        pairs.append(((int(u<p), int(1-u<p)),high-low))
    # Check whole population and subgroups, including own-menu invariant c alone.
    for group in ((0,1,2), (0,2), (2,)):
        centered_mean, level = 0, 0
        for ((z0,w0),p0),((z1,w1),p1) in product(pairs,repeat=2):
            row = {"permille":k, "base_placed":sum(baseline[i] for i in group),
                   "base_same_placed":baseline[2] if 2 in group else 0}
            for tag,arm in (("next",(z0,z1)),("anti",(w0,w1))):
                y = placed(arm)
                row[tag+"_placed"] = sum(y[i] for i in group)
                row[tag+"_same_placed"] = y[2] if 2 in group else 0
                row["ht_"+tag+"_numerator"] = sum((1000*arm[i]-k)*(y[i]-baseline[i])
                                                       for i in group if i < 2)
            out = pair_outcomes(row)
            centered_mean += float(p0*p1) * out["own_direct"]
            level += p0*p1 * F(out["placement_level"])
        # Exact own effects under independent Bernoulli assignment.
        assert isclose(centered_mean, float(sum((-p,1-p,F(0))[i] for i in group)), abs_tol=1e-12)
        assert level == sum((1-p*p, 2*p-p*p, (1-p)**2)[i] for i in group)
print("Verified centered paired direct estimator and levels at all five saturations for three populations.")
