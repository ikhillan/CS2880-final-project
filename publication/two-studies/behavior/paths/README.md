# Tracing placements on the binding menus

The theorem in the paper compares two stable assignments with fixed program priorities and capacities. Applicant lists can change membership and order. A witness strictly prefers their assignment under A in A's application order and strictly prefers their assignment under B in B's application order. Options excluded from a policy are below the outside option. Each lost placement has an injectively paired witness. The gain direction has its own injection. A person may witness both a loss and a gain.

## The frozen menus

In the binding school and housing comparisons, the positions of common options preserve their original relative order. Hence any witness must have changed their menu and at least one of their assigned options must be absent from the other menu. A loss path has length zero precisely when the new rule removes the individual's former partner. Among the 2,527 Chilean losses, 693 have length zero and 1,834 have positive length. Of those 1,834, 1,752 pupils kept their entire list and 82 kept the former course despite some other change to their list. The 1,834 terminal applicants are distinct. The split for gains is 75 zero-length paths and 665 positive-length paths. These paths identify allocation-dependent certificates. Removing just the endpoint applicant's edit in isolation need not reverse the displacement.

| Binding comparison | Gross loss | Old option still listed | Entire list unchanged among losses | Gross gain | New option already listed under A | Longest loss path |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Chilean admissions, four cohorts | 2,527 | 1,834 | 1,752 | 740 | 665 | 7 |
| NYC, ordinary inventory | 945 | 299 | 74 | 475 | 218 | 3 |
| NYC, enriched inventory | 2,110 | 439 | 83 | 1,212 | 427 | 2 |

School totals count applicant-years. NYC totals are unweighted counts of market instances. The school files contain only year and aggregate counts. The rental files identify simulation markets and aggregate counts. We pair departures to arrivals in increasing applicant-index order within each course, after canceling common contracts. The set of terminal applicants can depend on the pairing. We assert each correct priority comparison at each traversed seat, each strict comparison of the two assignments at a terminal, and disjointness across all paths in each direction. For school seats, the original tie order remains fixed.

## Reproduce

To reconcile the public counts against the paper's earlier placement and unchanged-menu tables, run `python verify_paths.py --paper "$PAPER"` with `$PAPER` set to the directory containing `main.tex`. This uses only the Python standard library and distributed aggregates.

The project directory must contain the rental inputs, cached menus and school-market reader described in the paper's replication guides. The early and late school directories contain the corresponding original cohort binaries, masks and archived partner vectors. From this directory run

```sh
python binding_paths.py school --repo "$REPO" --early "$EARLY" --late "$LATE" --out "$OUT"
python binding_paths.py housing --repo "$REPO" --out "$OUT"
c++ -O3 -std=c++17 verify_profiles.cpp -o /tmp/verify-shortlist-profiles
/tmp/verify-shortlist-profiles
```

The independent finite check enumerates all 125 profiles of ordered, possibly empty lists for three applicants and two programs, all 36 strict priority profiles, and all 16 capacity vectors in {0,1,2,3}². It finds feasible stable assignments by checking blocking pairs directly, then tests both path injections for each ordered pair of stable outcomes, including different selections under the same profile. The original fixed-graph study used separate three-by-three markets. The public result in `mixed-profile-check.json` refers only to the three-by-two mixed-graph check.
