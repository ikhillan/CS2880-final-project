"""Verify the paper's example with full lists and each of the eight mixed assignments."""
import argparse
import csv
from collections import deque
from fractions import Fraction
from itertools import product
from pathlib import Path

FULL = (("x", "y", "w"), ("x", "y", "w"), ("y", "x", "w"))
PRIORITIES = {"x": (2, 0, 1), "y": (1, 0, 2), "w": (0, 1, 2)}
A = ("x", "x", "y")
B = ("y", "y", "y")


def deferred_acceptance(lists):
    free = deque(range(len(lists)))
    proposal = [0] * len(lists)
    held = {}
    while free:
        i = free.popleft()
        if proposal[i] == len(lists[i]):
            continue
        school = lists[i][proposal[i]]
        proposal[i] += 1
        incumbent = held.get(school)
        order = PRIORITIES[school]
        if incumbent is None or order.index(i) < order.index(incumbent):
            held[school] = i
            if incumbent is not None:
                free.append(incumbent)
        else:
            free.append(i)
    assigned = [None] * len(lists)
    for school, pupil in held.items():
        assigned[pupil] = school
    return tuple(assigned)


def check(output):
    baseline = deferred_acceptance(FULL)
    assert baseline == ("w", "y", "x")
    assert tuple(ranking[0] for ranking in FULL) == A
    outcomes = {}
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["a_arm", "b_arm", "c_arm", "a_placed", "b_placed", "c_placed",
                         "total", "full_list_partner_hits"])
        for z in product((0, 1), repeat=3):
            menus = [B[i] if treated else A[i] for i, treated in enumerate(z)]
            assert all(j in FULL[i][:2] for i, j in enumerate(menus))
            assignments = deferred_acceptance([(j,) for j in menus])
            placed = tuple(int(j is not None) for j in assignments)
            assert placed == (1 - z[0] * z[1], int(z[0] or z[1]), int(z[0] == z[1] == 0))
            coverage = sum(j == target for j, target in zip(menus, baseline))
            assert coverage == z[1] and sum(placed) == 2 - z[0] * z[1]
            outcomes[z] = placed
            writer.writerow([*z, *placed, sum(placed), coverage])
    for p in (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)):
        own, spill = [], 0
        for i in range(3):
            direct = 0
            for peers in product((0, 1), repeat=2):
                z = list(peers)
                z.insert(i, 0)
                before = outcomes[tuple(z)]
                z[i] = 1
                after = outcomes[tuple(z)]
                mass = p ** sum(peers) * (1-p) ** (2-sum(peers))
                gain = after[i] - before[i]
                direct += mass * gain
                spill += mass * (sum(after) - sum(before) - gain)
            own.append(direct)
        assert own == [-p, 1-p, 0] and sum(own) + spill == -2*p
    print("Verified full-list DA, all eight assignments, coverage, own effects and spillovers.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    check(parser.parse_args().out)
