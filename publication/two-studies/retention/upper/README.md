# An admission-cutoff upper bound for retaining every incumbent

This outer bound uses the same union, shared courses, menu quotas, national baselines and April school labels as the contract-preservation calculation. It asks which individual *sets of displayed courses* remain possible if the pupil must keep a seat in every scenario where they were placed by the blind menus. A single row must pass all tested draws. We drop several joint constraints to obtain an upper bound. No feasible menu is constructed from the relaxed sets.

## Deferred acceptance on prefixes

For an applicant with no family link capable of changing the ordering of a candidate course or the priority at any such course, retain the longest initial proposal sequence common to all legal menus. In original order, collect shared courses and stop just before the first exclusive course from either of the original proposals. Fill all other applicants' rows with the empty list. Run the institutional engine on these prefixes. This relaxed market has fewer applications than any admissible target market.

Fix one target menu profile, a priority draw and one grade. Conditional on the target's higher-grade assignments, every nonempty relaxed list is a prefix of the target's operational contract list. Its within-course order and every relevant seat priority on those contracts are identical, by the exclusion of family-sensitive pupils. Rows removed from the relaxed market can return with *any* target-specific preferences and priorities. Fill in the missing applicants' actual priorities at the schools before comparing the two clearings. This does not affect the relaxed outcome because those applicants do not apply there. A personal continuity fallback can be represented by its own private extra seat. With strict priorities and fixed responsive capacities, a slot's worst held applicant can only improve, or its vacancies can only fall, when proposal lists are extended at the end. To see this, postpone new proposals until the DA run on the prefixes ends. Resume from exactly those held offers. Previously rejected contracts remain rejected since holds only improve as proposals continue. The completed run equals DA on the longer lists by proposal-order independence. The argument applies separately in each grade of the target run. It does not require the relaxed higher-grade assignments to agree with the target's.

A pupil cannot obtain a course if every nonempty physical category was filled in the relaxed outcome and its worst holder outranks the pupil even under the pupil's most favorable possible sibling priority. A current-school fallback is always allowed. For each pupil whose blind menu missed April but a legal row could hit it, enumerate every admissible row hitting April. On each design draw where the blind menus placed that pupil, discard a row if every displayed course is impossible by this test. Count pupils with at least one row remaining after intersecting the draws. Since old hits can supply at most their original count, this is an upper bound on the *net* gain of any menu preserving every incumbent. It remains valid for a larger panel containing the tested draws. It does not certify reachability of any surviving hit. Ignoring competing gains and allowing the most favorable sibling scenario tend to loosen the bound.

The release contains only the joint counts. Per-pupil masks are local intermediates. Old outcome files accompany the historical market format; no recorded offers are used as predictive features. This construction intentionally uses retrospective enrollment to bound a hindsight objective.

## Reproduction

Set `PUB` to the pinned repository's publication folder, `ANALYSIS` to this directory, and supply the numeric data and institutional sidecar used for the existing contract ceiling. The prefix contains the frozen market, blind menu mask, and rank-informed menu mask. Python requires only its standard library. Use C++17.

~~~sh
c++ -O3 -std=c++17 -I"$PUB/research/observed/school" -I"$PUB/two-studies/institutional" \
    -I"$PUB/two-studies/audit/chile" "$ANALYSIS/ghost_cutoff.cpp" -o "$STAGE/ghost"
python3 "$ANALYSIS/run_bound.py" --exe "$STAGE/ghost" --prefix "$PREFIX" \
    --year "$YEAR" --tiers "$TIERS" --register "$REGISTER" --side "$SIDE" \
    --scratch "$STAGE/work" --out "$STAGE/aggregate" --draws 64 --workers 2
python3 "$ANALYSIS/verify_prefix.py"
~~~

The default seed batch is 513. The simulation uses the recorded within-tier strategy and matched-course interleaving from the contract certificate. A separate exhaustive verifier checks 110,592 pairs of relaxed and full typed-seat outcomes with different category priorities, routes and candidate rows. For the first draw processed in each shard, the program additionally checks that no actual baseline recipient is rejected at their actual baseline assignment by the relaxed cutoffs.

## Which courses can be used without pushing an incumbent out?

With `--hall`, retain only the incumbent pupils on each draw. Draw a graph from these pupils to all courses they could attend at the relaxed priority cutoffs, under the union of admissible menus. Collapse seat categories into the total physical capacity of each course. Give pupils who keep a continuity fallback access to a common private-fallback pool with spare capacity; this models a separate guaranteed place for every such pupil. In this relaxation a fallback pupil might even take a physical seat whose cutoff would bar that pupil. That additional possibility can only weaken the upper bound.

The original assignment of these incumbents is a feasible matching in this graph. For each incumbent originally at course a and each course j in their relaxed set, draw a directed edge a→j on the course graph. Reassigning this person to j can preserve every incumbent if and only if j can reach a course with spare capacity, or j can reach a. For the reverse direction, move this pupil and the occupants along a simple path to the vacant course or back to a. If the path passes a before a vacancy, stop there. Distinct course nodes along the simple path mean distinct displaced pupils. For the forward implication, take any feasible reassignment which uses the edge a→j. The differences from baseline decompose into applicant-labeled cycles and paths ending at a baseline vacancy; the specified edge lies on one of these. Thus exactly all supported edges in the capacity relaxation can be recovered from the strongly connected components of the course graph and one reverse search from all vacancies. An alternative school must occur on the *same* row as the historical hit, and the same row must survive each tested draw. A fixed cutoff bound already eliminates impossible individual edges before this joint test.

A second verifier (`python3 verify_hall.py`) compared the support characterization against direct enumeration of entire assignments over 7,680 three-pupil baseline-specific capacitated graphs and 43,920 supported edges. In the Chilean application these are upper bounds because stable selection, family rules, externally affected pupils and missing seats may rule out additional target rows.

The `--veto` option adds a priority check to that capacity test. On a given draw, take a course j whose physical seats must all go to specific incumbents in every feasible assignment of incumbent seats in the relaxed graph. Exclude people with a fallback from this set of compelled physical occupants. If an outsider i outranks at least one such incumbent in *every* positive-capacity category, even under i's least favorable and that incumbent's most favorable sibling priority, i cannot both (a) propose to j and (b) leave every incumbent placed. If i eventually receives j the compelled occupants no longer fit. If i moves on to a later option, then j has rejected i from every category. The category containing the compelled, lower-priority holder could not have done so in a DA run. Consequently when testing i's menu under a fixed order, there must be a globally supported opportunity strictly before any such vetoed course. If i was originally unplaced they can still stay unplaced, but their application must avoid any vetoed proposal. Such a person can stop at an earlier course only if adding their assignment can preserve incumbent placements. In the residual graph this means the earlier course can reach a spare course. We apply the ordering test only to candidate rows where superior-grade block priorities cannot reorder courses. Switched rows can also create seats for previously unmatched applicants; leaving such applicants out of the relaxed feasibility system preserves validity of the upper bound. An independent verifier (`python3 verify_veto.py`) checks the veto against direct DA at all 16 treatments in each of 1,800 synthetic four-pupil markets, including multicategory school seats.


## Intersect the certified relaxations

The primary run uses the 1,024 matched-course design draws (batch 513). Reproduce that run with `--draws 1024` and a separate scratch folder for each cohort. A second run uses `--veto --draws 1024` with the same inputs and another scratch folder. The original compulsory-first-proposal bound supplies one byte per changed pupil for the same cohort. Set `UPPER` to that file in the original bound's scratch folder, `GHOST` and `VETO` to the two scratch folders, and use the following command. Intersecting whole menu configurations before counting a person can discard more than taking the smaller aggregate bound.

~~~sh
python3 "$ANALYSIS/merge_bounds.py" --exe "$STAGE/ghost" --prefix "$PREFIX" \
    --year "$YEAR" --tiers "$TIERS" --register "$REGISTER" --side "$SIDE" \
    --scratch "$STAGE/work" --other "$UPPER" --certificates "$GHOST" "$VETO" \
    --output "$STAGE/aggregate/joint-$YEAR.csv"
~~~

The columns `relaxation_1_upper` and `relaxation_2_upper` count pupils surviving each certificate separately. The next column counts those with a specific menu configuration surviving both simultaneously. The last column also imposes the original compulsory-first-proposal bound.

## Verified national upper bounds

| Application year | Eligible new hits without a retention constraint | Prefix cutoff, 1,024 draws | Prefix, residual capacity and forced-incumbent veto, 1,024 draws | Joint upper |
| --- | ---: | ---: | ---: | ---: |
| 2022 | 1,425 | 1,357 | 1,350 | 1,350 |
| 2023 | 1,633 | 1,570 | 1,564 | 1,564 |
| 2024 | 1,765 | 1,714 | 1,706 | 1,706 |
| Total | 4,823 | 4,641 | 4,620 | 4,620 |

The strongest bound excludes 203 potential new enrollment-school hits. Incumbent support and priority constraints remove 21 additional pupils beyond the 1,024-draw prefix-cutoff test. This result still relaxes competition between different proposed new hits. The compulsory-first-proposal check (4,815) imposes no further restriction. Only an actually cleared feasible menu can furnish a lower bound.

The full Hall/veto run used `--start 129 --resume-prefix "$PREVIOUS"` after archiving those first 128 draws for each cohort. Each previous receipt includes the year, seed batch, start, end, stride and draw count. The driver checks that previous shards jointly cover precisely draws 1 through 128 with no overlaps, then takes their intersection with the new draws. Use a new, empty scratch directory per call and carry forward only certificates produced by the same options. Running from draw 1 through 1,024 from scratch with `--veto` implements the same calculation without the optional resume step.

The distributed aggregate files are in `results/cutoff` for the full-panel prefix test, `results/capacity` for the full-panel capacity and priority test, and `results/joint-YYYY.csv` for the intersection.
