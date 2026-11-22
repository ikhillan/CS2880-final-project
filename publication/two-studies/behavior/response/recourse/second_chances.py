"""Maximum additional recipients on the historical complementary-round choice graph."""
import argparse
import csv
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, maximum_flow

def recovery(applicant, course, capacity, unplaced):
    """Unit-capacity applicants, published course seats, and a verified min-cut."""
    keep = unplaced[applicant] & (capacity[course] > 0)
    if not np.any(keep):
        return 0
    persons, rows = np.unique(applicant[keep], return_inverse=True)
    programs, cols = np.unique(course[keep], return_inverse=True)
    n, m = len(persons), len(programs)
    right = n + 1 + cols
    sink = n + m + 1
    left_ids = np.arange(1, n + 1)
    prog_ids = np.arange(n + 1, sink)
    heads = np.r_[np.zeros(n, dtype=int), rows + 1, prog_ids]
    tails = np.r_[left_ids, right, np.full(m, sink)]
    values = np.r_[np.ones(n + len(rows), dtype=np.int64), capacity[programs]]
    graph = coo_matrix((values, (heads, tails)), shape=(sink+1,sink+1)).tocsr()
    flow = maximum_flow(graph, 0, sink)
    residual = graph - flow.flow
    residual.eliminate_zeros()
    assert np.all(residual.data > 0)
    reached = breadth_first_order(residual, 0, directed=True, return_predecessors=False)
    source_side = np.zeros(sink+1, dtype=bool)
    source_side[reached] = True
    assert not source_side[sink]
    cut = values[source_side[heads] & ~source_side[tails]].sum()
    assert cut == flow.flow_value <= len(persons)
    return int(cut)

def study(args):
    out = []
    for y in range(2022, 2026):
        def read(folder, suffix, dtype):
            return np.fromfile(folder / f"{y}-{suffix}", dtype)
        people, program, rank, novel = read(args.search, "second-edges.i4", "<i4").reshape(-1,4).T
        cap, regular, school = read(args.search, "second-courses.i4", "<i4").reshape(-1,3).T
        blind, informed = [read(args.status, p+".u1", "u1").astype(bool) for p in ("blind","rank")]
        left_a, left_b = [read(args.status, p+"-remaining.i4", "<i4") for p in ("blind","rank")]
        assert len(left_a) == len(left_b) and (cap >= 0).all()
        assert (people >= 0).all() and people.max() < len(blind) == len(informed)
        assert (program >= 0).all() and program.max() < len(cap)
        assert (rank >= 1).all() and np.isin(novel, [0,1]).all()
        residual = np.zeros_like(cap)
        valid = regular >= 0
        residual[valid] = np.minimum(left_a[regular[valid]], left_b[regular[valid]])
        register = args.registers / f"{y}-register"
        with register.open("rb") as f:
            np.fromfile(f, "<i4", len(left_a))
            april = np.fromfile(f, "<i4")
        assert len(april) == len(blind)
        linked = april >= 0
        binding_hits = [read(args.status,p+"-hit.u1","u1").astype(bool) for p in ("blind","rank")]
        base_linked_gap = int((informed & linked).sum()) - int((blind & linked).sum())
        base_gap = int(informed.sum()) - int(blind.sum())
        for choices, use in (("new_school", novel == 1), ("all_explicit", np.ones(len(novel),bool))):
            a, j = people[use], program[use]
            for supply, seats in (("published_A2",cap),("common_regular_residual",np.minimum(cap,residual))):
                rescued_a, rescued_b, bound, bound_reverse = [recovery(a, j, seats, u) for u in
                    (~blind, ~informed, blind & ~informed, informed & ~blind)]
                linked_a, linked_b, linked_bound = [recovery(a,j,seats,u & linked) for u in
                    (~blind, ~informed, blind & ~informed)]
                # A family pursues a recorded choice only if its first-stage menu left
                # it unplaced and that course is in the common positive-vacancy pool.
                hits = []
                reaches_april = (school[j] == april[a]) & (seats[j] > 0)
                for placed, hit in zip((blind,informed),binding_hits):
                    pursued = hit.copy()
                    pursued[a[reaches_april & ~placed[a]]] = True
                    hits.append(int(pursued.sum()))
                extra = rescued_b - rescued_a
                assert -bound_reverse <= extra <= bound
                out.append(dict(year=y,choices=choices,supply=supply,first_stage_gap=base_gap,
                    blind_maximum=rescued_a,rank_maximum=rescued_b,difference_in_maxima=extra,
                    combined_gap=base_gap+extra,rank_only_upper=bound,blind_only_upper=bound_reverse,
                    linked_first_stage_gap=base_linked_gap,blind_linked_maximum=linked_a,
                    rank_linked_maximum=linked_b,linked_combined_gap=base_linked_gap+linked_b-linked_a,
                    linked_rank_only_upper=linked_bound,blind_pursued_hits=hits[0],
                    rank_pursued_hits=hits[1],pursued_hit_gap=hits[1]-hits[0]))
    with args.out.open("w") as f:
        writer=csv.DictWriter(f,out[0].keys(),lineterminator="\n")
        writer.writeheader()
        writer.writerows(out)

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("search","status","registers","out"):
        parser.add_argument("--"+name,required=True,type=Path)
    study(parser.parse_args())
