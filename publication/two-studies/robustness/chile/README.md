# Chilean school admissions, institutional and rollout checks

These are exploratory checks on the same 2022 and 2023 application cohorts as the paper. No student records, individual treatments, preferences or assignments are published in these outputs. They address three questions with the published 2019-trained rank-blind (A) and rank-informed (B) offer predictors.

1. **Adoption.** Assign each applicant independently to B with probability 0, 0.25, 0.5, 0.75 or 1. Every other applicant gets A. A returning pupil keeps their continuity course outside the five discretionary slots. Within a replication, the same uniform quantiles select applicants at each adoption probability. The second half of the replication uses the complementary quantiles. There are 64 independent antithetic pairs per school cohort. The standard error of a mean is the sample standard deviation across pair averages divided by eight. Endpoints are deterministic. Coverage is affine in adoption probability because it sums indicators evaluated against a fixed partner and own menu. At 50% each applicant receives B in exactly one antithetic half, so average coverage between those halves equals the endpoint average exactly. Counts of placements need not obey that identity.
2. **List length.** Set every discretionary quota to 1, 3, 5, 8 or 10 (honoring rows shorter than the quota). At each quota regenerate A and B with the original tie-breaking keys, leave the full-market partner fixed, and compare both policies. The historical offer is always the initially recorded offer. The two model fits never change. These ten year/quota comparisons share records, and their signs are not independent statistical tests.
3. **Priority lotteries.** Keep the lists, capacities, returning-pupil protections and every course's published priority tiers in place. In addition to the observed strict order, consider two counterfactual priority schemes. Assign either one independent lottery key per application edge or one per applicant shared across all of their courses. Lower keys win within each tier; cross-tier precedence always wins. Independently draw 32 priority realizations per scheme and cohort. Use the same realization for both menus **and for full-list DA** to keep the reference coherent. Historical offers and the displays remain fixed, so their coverage cannot change in this sensitivity. This is a scenario comparison with distinct lottery correlation structures, not a claim about Chile's actual administrative lotteries.

Every simulated allocation runs the original stable matching check, program capacity check, applicant feasibility check and entitlement check. At the original quota the program matches both original masks byte-for-byte and all three vectors (A, B, full) of archived partner IDs. Quotas are checked against the existing cohort endpoint aggregates. Lottery tiers are streamed from original ministry releases after checking each raw release hash and agreement on the exact applicant-edge order, program IDs and original strict priorities in the stored binary. The observed lottery row must reproduce both policies in each cohort.

## Columns

- In the rollout CSVs, `placed` counts people receiving any place. `full_covered` counts people whose full-list DA partner is on their current display. `recorded_covered` counts people whose historical offer is shown. `named` counts people actually receiving their full-list partner. Someone without a partner on full lists cannot be covered or named.
- `gains` and `losses` compare each realization with the all-A allocation. `fixed_*` restrict those transitions to people whose own A and B displays are identical; membership in this group is the same at all adoption levels. `pp` divides counts by the whole cohort and multiplies by 100. To use a conditional rate for the fixed group instead, divide its count by `fixed_rows`. The stored conditional Monte Carlo standard errors describe numerical precision over simulated adoption draws, conditional on these fixed cohort inputs. They are not uncertainty intervals for a rollout on a population of other markets or for actual participation.
- `midpoint_direct` is the summed own-assignment effect at 50% adoption, estimated as the aggregate difference in each person's two antithetic placement outcomes, signed according to which half gave that person B. Sum it over rows where A and B lists differ. With independent applicant assignment, the resulting expectation equals the sum of the effects of switching a person's own list while integrating out peers at 50% adoption. Its sign need not agree with total endpoint changes, and the two quantities cannot be subtracted to obtain a spillover effect without further structure.
- The quota and lottery outputs contain only paired totals. `joint_sign_draws` counts priority realizations in which using B raises full-partner visibility and lowers total placement. Means, dispersion and sign counts describe these explicit lottery distributions.

## Run

Set `PUB` to the repository's `publication/` folder, `NUM` to the private binaries and saved masks assembled for the two-study school replay, and `ARCHIVES` to the local directory containing the two original Ministry archives. Build with a standard C++17 compiler; keep assertions enabled. The Python programs require NumPy, pandas and SciPy and the original loader requires bsdtar with RAR support. The commands below are shown from this directory.

```sh
mkdir -p results
c++ -O3 -std=c++17 -I"$PUB/research/observed/school" school_rollout.cpp -o results/school_rollout
c++ -O3 -std=c++17 -I"$PUB/research/observed/school" school_lotteries.cpp -o results/school_lotteries
set -o pipefail
for year in 2022 2023; do
    results/school_rollout "$NUM" "$year" results 64
    python school_priority_tiers.py --archives "$ARCHIVES" --numeric "$NUM" --school "$PUB/research/observed/school" --cohorts "$PUB/research/policy/equity/cohorts.json" --year "$year" |
        results/school_lotteries "$NUM" "$year" 32 "results/school-$year-lottery.csv"
done
python school_rollout_summary.py --inputs results --output results --baseline-evidence "$PUB/two-studies/evidence"
```

The randomizer uses mt19937_64. The within-pair quarter-intervals are the top two bits of one draw per pupil; quarter `q` has antithetic quarter `3-q`. The stream is set by year and replicate. The separate lottery uses year, scheme and replicate in its independent seed. This makes the checks repeatable on the frozen inputs and separates numerical integration from the comparisons across real application cycles.

## Numerical checks

The five quotas give rank-informed minus rank-blind replay placement counts of −18,907, −2,367, −468, −109 and −48 in 2022 and −17,922, −2,264, −517, −104 and −51 in 2023. Recorded-offer coverage gains remain positive, as do same-priority full-list DA partner coverage gains. In all 128 new within-tier lottery draws, full-list partner coverage rose and total placement fell. The observed-priority arm separately reproduced the original matched partners, coverage and placements.

At 50% rollout, the 2022 cohort lost 230.01 placements relative to all-A (MC SE 0.45) and the 2023 cohort lost 250.81 (0.44). These departures from the endpoint interpolation are 3.99 and 7.69 places upward. The counts of permanently unchanged-menu pupils are 554,724 and 519,606. Their mean losses/gains at half adoption were 271.92/144.93 (MC SE 0.45/0.31) and 271.15/106.67 (MC SE 0.42/0.36), respectively. Direct effects at the same adoption probability were −202.91 (0.54) and −167.80 (0.64) when summed over edited rows. The rollout label is randomized in the simulation; no real pupil received an intervention.
