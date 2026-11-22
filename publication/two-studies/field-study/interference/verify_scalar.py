"""Independent scalar enumeration with exact arithmetic and no third-party dependencies."""
from collections import Counter
from hashlib import sha256
from itertools import permutations, product
from pathlib import Path
import json
import struct
import sys

people = range(3)
menus = list(product(people, repeat=3))
orders = list(product(permutations(people), repeat=3))
popcount = tuple(bin(z).count("1") for z in range(8))


def award(menu, orders_by_school):
    seats = 0
    for school, order in enumerate(orders_by_school):
        for pupil in order:
            if menu[pupil] == school:
                seats += 1 << pupil
                break
    return seats


def analyze(outcomes):
    total = [popcount[m] for m in outcomes]
    own, other = [0] * 3, [0] * 3
    for i in people:
        bit = 1 << i
        for z in range(8):
            if z & bit:
                continue
            k = popcount[z]
            before, after = outcomes[z], outcomes[z | bit]
            own[k] += bool(after & bit) - bool(before & bit)
            other[k] += popcount[after & ~bit] - popcount[before & ~bit]
    # T(p) is a cubic in p. Its derivative has three coefficients.
    strata = [sum(total[z] for z in range(8) if popcount[z] == k) for k in range(4)]
    a, b, c, d = strata
    q0, q1, q2 = (x + y for x, y in zip(own, other))
    assert (b-3*a, 6*a-4*b+2*c, 3*(d-c+b-a)) == (q0, q1-2*q0, q0-q1+q2)
    assert 6*(d-a) == 2*q0 + q1 + 2*q2
    own_num, spill_num = [9*values[0] + 3*values[1] + values[2] for values in (own, other)]
    return (d-a, own_num, spill_num)


def enumerate_markets():
    table = {menu: [award(menu, order) for order in orders] for menu in menus}
    digest = sha256()
    dist = Counter()
    by_changed = {k: Counter() for k in range(4)}
    for baseline, alternative in product(menus, repeat=2):
        changed = sum(a != b for a, b in zip(baseline, alternative))
        mixed = [tuple(alternative[i] if z & (1 << i) else baseline[i] for i in people)
                 for z in range(8)]
        for outcomes in zip(*(table[menu] for menu in mixed)):
            rollout, own, spill = analyze(outcomes)
            digest.update(struct.pack("11b", *outcomes, rollout, own, spill))
            dist[rollout, own, spill] += 1
            by_changed[changed]["total"] += 1
            by_changed[changed]["rollout_losses"] += int(rollout < 0)
            by_changed[changed]["reversals"] += int(rollout < 0 and own > 0)
    cases = sum(dist.values())
    mismatch = sum(n for (r, d, s), n in dist.items() if r < 0 and d > 0)
    negative = sum(n for (r, d, s), n in dist.items() if r < 0)
    positive = sum(n for (r, d, s), n in dist.items() if d > 0)
    extrema = [d for r, d, s in dist if r < 0 and d > 0]
    example = [award(tuple((1,1,1)[i] if z & (1 << i) else (0,0,1)[i] for i in people),
                     ((0,1,2), (1,0,2), (0,1,2))) for z in range(8)]
    example_outcomes = analyze(example)
    assert example_outcomes == (-1, 8, -16)
    assert cases == 3**6 * 6**3
    return {
        "markets": cases, "assignments": cases*8, "negative_rollout": negative,
        "positive_direct": positive, "positive_direct_and_rollout_loss": mismatch,
        "range_48_times_positive_direct_among_reversals": [min(extrema), max(extrema)],
        "by_number_whose_choices_differ": by_changed,
        "example_admitted_masks": example,
        "example_rollout_48direct_48spill": example_outcomes,
        "market_record_sha256": digest.hexdigest(),
    }


if __name__ == "__main__":
    report = json.dumps(enumerate_markets(), indent=2) + "\n"
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(report)
    print(report, end="")
