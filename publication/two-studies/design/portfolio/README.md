# Selecting five courses before the priority draw

A rule that predicts the course a pupil will receive can spend several slots on courses that offer little additional insurance. We compare the archived partner-frequency shortlist with a rule that keeps a course when it covers a simulated lottery outcome in which the other selected courses would all be unavailable. All policies use the same reported lists and entitlements. The two simulated policies are given exactly the same 64 full-list deferred acceptance (DA) draws in a batch. The coverage rule uses additional per-edge cutoff bookkeeping and exact discrete optimization.

## Algorithm and exactness

Fix the profile of submitted lists and five public priority categories. Each scenario clears the full eligible lists under new within-tier priority orders. For each applicant–course edge, store a 64-bit mask whose bit is one when the course has an unfilled positive-capacity seat or the applicant ranks at least as high as its worst holder. Within each course every simulated priority is strict. A holder passes at equality. An empty zero-capacity course has no attainable edge. Stability implies that the highest-ranked attainable course for each applicant is their full-list assignment. The program checks this identity for every applicant in every simulated draw and in the realized full-list replay.

Choose up to five discretionary courses and always include the protected return course when present. The primary score counts scenarios in which at least one selected course clears its fixed cutoff. Break ties by the total number of times the pupil's full-list partner appears in the menu, then by the sum over courses of their numbers of attainable draws, and finally by the lexicographically earliest ordered vector of submitted ranks. For the secondary and tertiary criteria, the protected option adds the same constants to every menu. Clearing the retained lists uses the family's original submitted order.

The ablation drops the primary criterion. Sorting by partner frequency, then the number of attainable draws, then submitted rank exactly optimizes all its criteria: an optimal sum of individual fixed-cardinality scores must include every score above the selection boundary, leaving only equal scores to the next criterion. For the full rule, first check whether this ablation already covers the union of all scenario masks. If it does, each optimality condition holds. Otherwise enumerate five-element subsets in rank order. At each partial subset the union with all still-available masks bounds final coverage; prune only if this bound is strictly below the incumbent. Evaluate remaining leaves on the entire lexicographic objective. This is exact with work up to a factor of the number of five-element subsets per applicant; there is no claim of an efficient exact algorithm for an unrestricted list budget. The independent verifier compares the implementation with direct enumeration of all feasible sets on 2,048 seeded small rows, including empty masks, a guaranteed choice, equal scores and the 64th bit.

A fixed-cutoff score does not recompute other applicants' assignments. When everyone uses a shorter menu, the cutoffs and the resulting access can change. The empirical outcome comes from clearing the whole cohort anew at its recorded priorities. The recorded priority order, offers and enrollment records never enter menu selection. The algorithm uses every submitted list, so it could run after collection of reports and before the lottery.

## What the recorded-priority replays show

All eight batches per year use the same evaluation-year priority order. Numbers in the table are separate medians of paired counts over the eight batches. Bracketed values give the minimum and maximum over seeds. They describe simulation sensitivity, not sampling uncertainty across people or years. Independent course-specific lotteries are the primary training simulation. The validation cohorts’ earlier endpoint patterns were already known when this experiment was designed.

| Application year | Applicants | Placement gain over partner frequency | Placement gain over backup tie-break | Newly placed / lost vs. frequency | First choices vs. frequency | April school shown / assigned vs. frequency |
|---|---:|---:|---:|---:|---:|---:|
| 2022 | 570,891 | +42.5 [29,57] | +24.5 [13,30] | 175 / 133.5 | −55 | +151 / −114.5 |
| 2023 | 536,353 | +39.5 [32,50] | +21.5 [11,24] | 156.5 / 113 | −46.5 | +137 / −83 |
| 2024 | 473,482 | +40.5 [27,46] | +22 [9,26] | 113 / 75 | −40.5 | +254 / −61 |

The net increment over the archived MC policy is under 0.011 percentage points in every evaluation. Compared with the frozen rank-blind predictor, this new rule still places a median 387.5, 435.5 and 426 fewer pupils respectively. Among pupils without a historical initial offer, the net changes against the MC rule range from −2 to +41, +1 to +23 and −13 to +9. Net increases in total placements therefore do not establish a benefit for this subgroup. Showing the historically attended April school more often can coexist with assigning fewer pupils to it. Neither endpoint measures how families would respond to the policy. In 2023 one or more batches also showed fewer historical offers (range −8 to +32) despite improving total placement. Unchanged menus accounted for a median 99, 82.5 and 55.5 lost placements. Across seeds the three corresponding gross-loss ranges were 93–115, 76–88 and 44–64.

The sensitivity run samples one random permutation of applicants and uses that ordering at every course within its own priority categories. Both simulated policies use this kernel, but evaluation stays at the identical recorded priority order. Coverage gained 94, 125 and 118.5 placements versus partner frequency. Compared to the backup tie-break alone, the differences were +0.5 [−4,2], 0 [−1,1] and 0 [0,1]. Only 17–38 pupils per cohort/batch received a different menu under these last two rules, compared with 686–1,207 under independent-course priorities. Among initial-no-offer pupils, coverage versus partner frequency lost a median 92, 88 and 74 placements under this alternative simulator. These results are a deliberate kernel stress, not outcomes under an implemented common lottery.

## Reproduction

Use Python with NumPy and pandas, a C++17 compiler, and bsdtar. The existing publication pipeline prepares the private regular-round numeric markets and the two frozen predictor masks. The original archive and register paths below are directories. The three register files must be named matricula-2023.rar, matricula-2024.rar and matricula-2025.rar. Set the indicated shell variables to the directory containing this README, the publication directory, each numeric and source directory, and a directory for intermediate private data.

```sh
python "$PORTFOLIO/prepare.py" --publication "$PUBLICATION" \
  --early-archives "$EARLY_ARCHIVES" --early-numeric "$EARLY_NUMERIC" \
  --late-archives "$LATE_ARCHIVES" --late-numeric "$LATE_NUMERIC" \
  --enrollments "$ENROLLMENT_ARCHIVES" --out "$PRIVATE"

python "$PORTFOLIO/run_batches.py" --publication "$PUBLICATION" \
  --early-numeric "$EARLY_NUMERIC" --late-numeric "$LATE_NUMERIC" \
  --private "$PRIVATE" --kernel independent --out portfolio-batches.csv

python "$PORTFOLIO/run_batches.py" --publication "$PUBLICATION" \
  --early-numeric "$EARLY_NUMERIC" --late-numeric "$LATE_NUMERIC" \
  --private "$PRIVATE" --kernel common --out portfolio-common-batches.csv

python "$PORTFOLIO/verify_portfolio.py" "$PRIVATE/independent-portfolio-replay"
```

The runner verifies identical original linked-register school coverage for the frozen predictors, the archived DA benchmarks and the original frequency policy under the independent kernel. It checks that net placements equal individual gains minus losses for every policy. Aggregate files contain one row for each course-priority kernel, cohort, seed and policy (the kernel is the filename). Counts of simulated events sum over 64 training draws. All comparator differences with suffix vs_mc use partner frequency trained under the kernel in that same file. Fields april_shown and april_match count respectively historical April school visibility and replay assignment to that school. Strata beginning no_official refer solely to no initial D1 offer. The 2025 tie procedure changed, putting it outside this three-year validation.

## Relation to past work

Arteaga, Kapor, Neilson and Zimmerman implemented real feedback about the risk that a Chilean school list fails to produce a placement. Their intervention prompted families to extend their lists. Ali and Shorrer analyze optimal application portfolios with correlated admissions outcomes. We treat management of portfolio risk and simulation-based advice as established ideas. The comparison here fixes five discretionary slots for every pupil, holds submitted orders fixed, and measures the change from partner prediction to scenario coverage when the entire market clears on the new lists. [Arteaga et al.](https://www.nber.org/papers/w28946) [Ali and Shorrer](https://doi.org/10.1257/aer.20211137)
