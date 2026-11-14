# Keeping a seat when pupils can switch schools

The comparison preserves receipt of a seat for each individual placed by the blind menu. The person can move to a different school, and pupils originally unplaced can enter. Changes in aggregate headcounts alone cannot verify this requirement. The 2022–24 design panels, fixed common options and course quotas are exactly those used for the paper's earlier 314-hit contract certificate. Coverage is the number of historical April enrollment schools displayed, counting each linked pupil once. The 314 result requires preserving the same course and seat category on every design draw. It supplies a feasible starting point for the weaker requirement.

| Application year | Same course and seat type | Seat retention, achieved | Seat retention, upper bound | Unconstrained optimum |
| --- | ---: | ---: | ---: | ---: |
| 2022 | 96 | 170 | 1,350 | 1,425 |
| 2023 | 111 | 199 | 1,564 | 1,633 |
| 2024 | 107 | 273 | 1,706 | 1,765 |
| Total | 314 | 642 | 4,620 | 4,823 |

| Question | Files |
| --- | --- |
| Can an optimum decompose by pupil? | [Hardness and exact feasibility checks for simple DA](theory/README.md) |
| Which real gains were impossible in the national panel? | [Upper-bound certificates](upper/README.md) |
| Which joint menus actually preserved all incumbents? | [Two hindsight constructions and complete national re-clears](bounds/README.md) |
| What rule can run before the actual lottery? | [Calibration, design, holdout tests and data access](policy/README.md) |
| Can one retain a seat under every lottery order? | [Structural guarantee, small-market verification and national replays](structural/README.md) |
| Could the recorded offers or ordinary tie priorities affect the decisions? | [Decision-time input check](isolation/README.md) |
| How did both rules perform on the identical lotteries? | [Paired policy comparison](comparison/README.md) |
| Which historical groups gained coverage? | [Group totals](coverage-groups.csv) |

The policy and structural subfolders contain separate pre-assignment rules. Both can run after families submit their lists and the platform has the special-admission priorities, without observing the ordinary tie-breaks or the offers for the current year. The learned policy uses only a previous cohort's April register for training. Its fresh tests use new priority seeds for the same national applicant pools. All results measure allocations conditional on the reports, supply, family rules and specified generators. The bounds subfolder deliberately uses evaluation-cohort April labels when constructing a hindsight feasible example. The upper-bound code may also see these labels when certifying an impossible gain. Neither use is part of prospective scoring.

## Inputs and reproduction

Run the corresponding [institutional input reconstruction](../institutional/REPRODUCTION.md) in the complete source repository. The public SAE releases, April register URLs and hashes are documented there and in its manifests. For 2021 calibration follow the policy subfolder's input instructions. Scratch files are pupil-level and stay locally. The distributed draw and group tables are aggregates. See the question-specific links above for compilation and exact commands.

After checking the frozen menus, reproduce the descriptive group analysis with the same config used by the policy driver. The structural scratch folder contains YEAR-prospective.mask. The learned policy scratch folder contains YEAR.mask.

~~~sh
python3 score_groups.py --config "$CONFIG" --structural "$STRUCT_SCRATCH" \
    --learned "$LEARNED_SCRATCH" --structural-freeze structural/frozen-masks.csv \
    --learned-freeze policy/policy-freeze.json --output "$OUT/coverage-groups.csv"
~~~

The two complete partitions of each cohort use the official disadvantaged priority flag and initial-offer status. Both are historical classifications. The script verifies the decision-time masks using their hashes before computing group differences. Its totals in each partition reconcile to the complete linked population.

Run `python3 verify.py` to reconcile the cohort bounds, the two equity partitions and all 480 paired test draws directly from the public aggregates.
