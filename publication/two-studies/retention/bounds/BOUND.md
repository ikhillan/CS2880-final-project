# Finite-lottery hindsight bounds for keeping every baseline recipient placed

Freeze the rank-blind and rank-informed course menus. For each pupil retain every common course and the blind menu length. Every other displayed course must come from the union of the two menus. Score whether at least one displayed course belongs to the pupil's historical April school. Fix the three 1,024-draw supports, seed batch 513 and matched within-course reconstruction used for the 314 exact-contract certificate. A menu passes seat retention if every pupil who receives a seat under the blind policy on that draw receives any seat under the same draw on the edited lists. Each year's policy is fixed across all draws. This diagnostic uses the future April label. It is distinct from the prospective policies.

## Two necessary conditions

Pupil i contributes at most one additional hit and can contribute one only if the blind menu misses April and the proposed-only coordinates include April. Thus the count of such pupils is an unconditional upper bound for gross gains and for the net objective, without assumptions on other pupils' edits.

For a refinement, consider a candidate applicant–course pair on any given draw. Exclude protected continuity fallback, which always guarantees retention if displayed. Let Q be the total physical seats at the course. A competitor known to propose to the course first under *all* allowed displays and relevant family states must either take a physical seat there, receive a continuity fallback, or be rejected by every typed seat. We count only competitors who have no fallback at this course. If Q distinct such competitors each strictly outrank the applicant in every nonempty seat category, the applicant cannot end at this course. Indeed, if they held a seat of category c, at most Q−1 of the competitors could also hold seats at this course. Some competitor would have been rejected at c, which is impossible if it outranks a final holder there. With Q=0 rejection is automatic. To make the priority comparison valid even when higher-grade assignments change, use the least favorable plausible recent-sibling flag for each competitor and the most favorable plausible flag for the target. A first choice is certified only when common to the two menus and no possibly promoted course could outrank it. Static tiers and special orders retain their institutional meanings.

If the pupil was placed at baseline in a draw, any feasible hit-producing row must also contain a course not ruled out by this criterion. All common courses can appear. With two or more exchangeable slots, every member of the union can co-occur with a proposed-only April course. With a single exchangeable slot the only co-occurring options are common courses and proposed-only April courses. Excluding pupils for whom any protected draw violates this necessary condition gives an upper bound. No joint feasibility or stability of the remaining rows is assumed.

## Constructing the lower bound

For each possible net new coverage hit, propose one added April course and remove one old-only course; leave all other pupils' menus unchanged. Prefer an added edge and removable edge of the old stable-assignment certificate when available, then take the first eligible add and the last eligible drop in original order. The earlier exact-contract policy remains an independent feasible candidate. The driver checks that the final certified menu reaches at least its hit count. Across progressively larger nested sets of design draws, clear both complete national markets under the same institutional lottery. If any incumbent loses a seat, identify candidate edits by following the two allocations' application paths through changed pupils, reached courses and affected younger relatives, and withdraw edits in the connected part of the graph containing a pupil who lost a seat. The paths only choose which edits to discard. They are not a proof that a remaining edit is harmless. A final, fixed menu must pass direct institutional re-clearing on every one of the 1,024 draws. The aggregate national checks supply the lower-bound certificate. Every retained edit adds exactly one April hit and removes no old hit. The resulting optimum interval uses no claims about behavior, untested draws or historical attendance under a deployed policy.

## Reproduce from private stage inputs

Use the source tree, public raw-file instructions and frozen masks in the institutional replication. Set PUB to the checkout's publication directory, SRC to this bounds directory, EARLY to the private numeric directory holding the 2022/2023 market binaries plus blind/rank masks, LATE to its 2024 equivalent, FEATURES to the directory with YEAR-tiers and YEAR-register, and SIDES to the directory with each YEAR.side. A standard C++17 compiler and Python 3 suffice. Scratch space is under user control.

~~~sh
c++ -std=c++17 -O3 -I"$PUB/research/observed/school" -I"$PUB/two-studies/audit/chile" \
    -I"$PUB/two-studies/institutional" "$SRC/retention_oracle.cpp" -o "$TMPDIR/retention-oracle"
for year in 2022 2023 2024; do
    python3 "$SRC/run_bounds.py" --binary "$TMPDIR/retention-oracle" --numeric "$EARLY" \
        --numeric-2024 "$LATE" --features "$FEATURES" --sides "$SIDES" --year "$year" \
        --scratch "$TMPDIR/retention-private" --results "$TMPDIR/retention-results" --workers 2 --draws 1024
done
python3 "$SRC/check_bounds.py" "$TMPDIR/retention-results"
~~~

The number of workers changes how the independent draws are split. Integer AND/OR, sorted repetition identifiers and deterministic masks ensure the same result. The file with suffix '-oracle-draws.csv' reports gross losses of baseline recipients, entrants and course switches among applicants who remained placed, separately by cohort and priority draw. The seven columns also include the number of changes in detailed (course, seat category) outcomes. Run `python3 check_forced_rejection.py` for an independent exact check of the simple rejection certificate in typed-seat markets. The pruning file records simultaneous removal of edits; the nested 8, 32, 128, 512, 1,024 draws are design checks and never a holdout. In the final draw receipts, a switch can be to a different course at the same school.

A second pass adds the eligible April hits back to the exact-contract core, including rows that the first procedure discarded. For a positive-hit row outside that core, it chooses among old-only drops using 64 paired baseline design draws. A dropped edge that held its pupil's seat receives a penalty of ten if no other old course clears an observed cutoff for the pupil, otherwise one. The added April course is chosen to minimize the number of baseline draws on which it would block; deterministic ties favor later old courses and earlier additions. We prune conflicts using the two assignments' changed proposal tails and ultimate contracts. If a loss has no attributed edit, a broader connectivity pass proposes removals. These are search heuristics. The retained gain is certified only after complete recomputation on all original 1,024 draws. Select the larger outcome from the two independently verified candidate menus in each year. Both menus use April labels, so this selection is a hindsight lower bound.

After the first pass, run the second from the same scratch directory. The second program writes outside its neighbor's private work directory. Store each annual addback output with the other result files for the final comparison.

~~~sh
c++ -std=c++17 -O3 -I"$PUB/research/observed/school" -I"$PUB/two-studies/audit/chile" \
    -I"$PUB/two-studies/institutional" "$SRC/drop_risk.cpp" -o "$TMPDIR/retention-drop-risk"
for year in 2022 2023 2024; do
    python3 "$SRC/run_addback.py" --bin "$TMPDIR/retention-oracle" --risk-bin "$TMPDIR/retention-drop-risk" \
        --early "$EARLY" --late "$LATE" --features "$FEATURES" --sides "$SIDES" --year "$year" \
        --scratch "$TMPDIR/retention-private" --out "$TMPDIR/retention-results" --workers 2
done
~~~
