# How families pursue options outside their school shortlist

We use the same four national cohorts, pooled capacities, published priority orders, five discretionary choices, and protected continuity programs as the original rank-informed versus rank-blind comparison. These exploratory scenarios change which programs each pupil pursues and the order of pursuit. We score first-choice assignments against the original family-entered preference in every scenario, even when the order of pursuit changes. A school's recorded offer is a separate coverage target. All alternative assignments are counterfactual, and the recovery rates are assumptions. PLAN.md, TAIL-PLAN.md, and ALL-TAIL-PLAN.md document the sequence of design decisions. Earlier fitted-policy endpoints were already known.

## Changing which submitted options can be pursued

In `full` recovery, a recovering applicant can pursue the entire original eligible report. In `one` recovery, the applicant can also pursue their highest-ranked omitted eligible course. Both responses keep original relative ranks and take the original five as the baseline. Eligibility here includes courses at or above a returning pupil's guaranteed current school. A protected right stays in the menu in both arms. At each recovery rate, the same random pupils recover options under both predictors. The 32 pairs per cohort use complementary quarter-interval assignments and share the same uniforms across probabilities and response rules. Half-recovery coverage differences equal the midpoint of the two extreme coverage differences, exactly for each complementary pair. This identity holds for either fixed target. A deterministic endpoint is evaluated once. Points at 0.25, 0.50, and 0.75 use 64 realizations per scenario and cohort. The standard error is computed over 32 pair means. For the four-year sum, each replication-pair index first sums over separately seeded cohorts. Error estimates condition on each cohort's data, assumed response and original rules.

| Probability of full recovery | Expected additional historical offers shown | Change in placements | Change in family first choices | Gross losses on identical effective menus |
|---|---:|---:|---:|---:|
| 0 | 5,552 | −1,787 | +2,218 | 1,752 |
| 0.25 | 4,164 | −1,336.16 | +1,608.02 | 1,349.75 |
| 0.50 | 2,776 | −886.95 | +1,057.44 | 943.33 |
| 0.75 | 1,388 | −440.22 | +529.16 | 505.80 |
| 1 | 0 | 0 | 0 | 0 |

The historical coverage column uses its exact expectation from endpoint interpolation. The other columns use simulation means at interior probabilities. At 50% full recovery the Monte Carlo standard errors of the placement and first-choice changes are 1.69 and 2.46 pupils. With 50% single-option recovery in original order, the changes are −1,359.67 and +647.97 (MCSE 1.26 and 1.92 pupils). `school-recovery-summary.csv` gives each policy level, difference, and Monte Carlo error at all five probabilities. Year 0 denotes the sum over all four cohorts. Separate year-specific CSVs contain the aggregate counters for every paired clearing. Altogether the fractional experiment used 3,072 interior whole-cohort DA clears (two policies × two responses × three interior rates × two complementary realizations × 32 pairs × four years). People can reappear across years.

## Changing where the recovered course appears

`in_order`, `before_right`, and `last` give every family the same additional option under a given predictor, whenever one was omitted. The first restores the course at its position on the submitted list. The second puts it after all old discretionary choices and just before any guaranteed return. The third appends it after the entire previous menu. In the two tail rules, the recovered option can be the family's original first choice even though the family pursues it later. `all_last` appends every omitted eligible course in original relative order; `all_original` restores the full original list. Both full-graph policies give both predictors the exact same option sets. A guaranteed current-school option blocks access to any new option strictly below it in the effective order. We separately count family-entered first choices by excluding administratively appended fallbacks.

| Response by all applicants | Difference in recorded offers shown | Difference in placements | Difference in first choices | Gross losses on identical effective order |
|---|---:|---:|---:|---:|
| Keep original shortlist | +5,552 | −1,787 | +2,218 | 1,752 |
| Restore one at submitted rank | +676 | −940 | −846 | 747 |
| Add one before guaranteed return | +676 | −1,764 | +2,161 | 1,754 |
| Add one after all old options | +676 | −1,761 | +2,163 | 1,754 |
| Add every omitted option after all old options | 0 | −1,331 | +2,005 | 1,788 |
| Restore full submitted order | 0 | 0 | 0 | 0 |

Relative to each policy's binding allocation, appending one option literally last adds 39 net blind placements (131 entrants, 92 exits) and 65 net rank-informed placements (189 entrants, 124 exits). No program loses aggregate occupancy in either case. Appending every omitted option increases each policy's own placement count by 95 or 551, even though 197 or 362 original pupils lose places. No extended pupil with continuity receives a course below their guarantee. The before-return scenario permits such a pupil to obtain the recovered course above that guarantee. Relative to each policy's binding outcome, 37 under blind and 57 under rank-informed receive such a course. All outcomes use the same program capacities and priorities. The serial solver checks stability on the effective submitted order for every assignment.

`school-tail-summary.csv` gives policy differences by year and over all four. `school-tail-YYYY-arms.csv` supplies the levels and compares each rule with the same predictor's binding allocation. A column labeled `named` counts exact assignments to the full-list DA partner. `received_recovered` counts recipients of an off-menu addition. `school-tail-YYYY-comparison.csv` gives turnover and unchanged-set versus unchanged-order counts. Across the all-option appended comparison, 1,977,930 pupils have identical orders and all 2,044,649 have identical sets. Of those who lose or gain a place, 1,788 of 1,871 losses and 515 of 540 gains occur on identical orders. These movements are mechanically due to competition when others’ effective orders change. They do not reveal what families actually would do in response to advice.

## Assignment and displacement checks

`check_recovery.py` reconstructs the effective order independently with a group sort, regenerates midpoint assignments using the shared randomization keys and reruns batched DA. `school-recovery-verification.json` records matching complete assignment-vector fingerprints (32,714,384 applicant–policy outcomes across 64 national allocations) and independent checks of public endpoints, first-choice ranks and historical visibility. The checked draws include one paired 50%-recovery realization of each response for each original policy and year and every deterministic pursuit order in the table. Only aggregate fingerprints and counts are distributed. The same randomization helps isolate how pursuing recovered options can change allocation under the two menus, given the specified behavioral responses.

`school_order_witness.py` compares the two all-tail outcomes. Among 1,885,128 records that are placed under both rules, 9,027 strictly favor their own assignment under each respective pursuit order. These preferences concern the induced order, distinct from the original family-entered ranking used to evaluate first choices. At each school we pair departing and incoming holders in order of pupil index, after removing holders common to both allocations. Starting from each loss and then separately from each gain, the program follows holder links and checks the priority and old-order inequalities at each step, the reversal at its endpoint, and the uniqueness of the visited applicants within direction. All 1,871 loss paths and all 540 gain paths end at distinct reversal witnesses within their direction. The two endpoint sets overlap on 191 applicant records for this particular pairing. Paths range from one to seven programs. The path trace is a certificate that explains the two assignments, independent of proposal chronology. `school-order-witnesses.csv` and `school-order-path-lengths.csv` contain the year-by-year aggregates. The path pairing depends on the numerical owner order and is one possible certificate.

## Reproduction

Install the upstream research repository's NumPy, pandas and C++17 dependencies. Point `REPO` at that checkout, including `publication/two-studies/audit/chile` and the original public aggregates. Prepare `EARLY` and `LATE` with the archived national market binaries, masks and partners for 2022–23 and 2024–25 as described in the original study. Run from this folder. Its source files share the original national solver for the main sweeps. The independent checker uses the batched solver in `publication/research/policy/equity/transitions.py`.

```sh
mkdir -p replay
c++ -O3 -std=c++17 -I "$REPO/publication/research/observed/school" school_recovery.cpp -o replay/school_recovery
c++ -O3 -std=c++17 -I "$REPO/publication/research/observed/school" school_tail.cpp -o replay/school_tail
for year in 2022 2023 2024 2025; do
  if [ "$year" -le 2023 ]; then numeric="$EARLY"; else numeric="$LATE"; fi
  replay/school_recovery "$numeric" "$year" replay 32
  replay/school_tail "$numeric" "$year" replay
done
python recovery_summary.py replay
python check_recovery.py --repo "$REPO" --early "$EARLY" --late "$LATE" --directory replay
python school_order_witness.py --repo "$REPO" --early "$EARLY" --late "$LATE" --directory replay
```

`python recovery_summary.py .` regenerates summaries and checks endpoint and midpoint identities using only published aggregate counts. The summed study population is 2,044,649 application records. DO NOT sum `year=0` with the individual cohorts; these rows repeat the same records as a pooled cohort summary.
