# University of Nebraska at Omaha mentoring

Christian Haas and Margeret Hall (2019), “Two-Sided Matching for mentor-mentee allocations—Algorithms and manipulation strategies”, PLOS ONE 14(3), e0213323. The author supplement supplies three annual mentoring cohorts, including original ranked tiers on both sides. The preprocessing uses mutually reported acceptable pairs. Every original mentee remains in the placement denominator. The third tab contains 29 mentees with at least one mutual edge. Table 1 of the paper says 28, while its Table 2 reports 29. All three tabs contain unique within-list alternatives and valid opposite-side IDs. Each nonempty tier may contain several tied partners. The selectors and DA use different tie lotteries. The supplement contains no verified realized assignment and we use no published allocation outcomes as learning targets. [Article and attribution](https://doi.org/10.1371/journal.pone.0213323) · [Original S1 Data](https://doi.org/10.1371/journal.pone.0213323.s001)

## Run

Online download of the public CC BY supplement is required. This evidence directory distributes only aggregate outcome summaries, model coefficients and code. No source table of individual preferences is reproduced. The parser verifies the supplied workbook's SHA256. The first command reads only the three original rankings tabs. Expected dependencies are Python 3.13, NumPy 2.2.6 and SciPy 1.17.1. No training label, ranking or outcome is sent to a third party.

~~~sh
curl --fail --location 'https://journals.plos.org/plosone/article/file?type=supplementary&id=info:doi/10.1371/journal.pone.0213323.s001' -o mentor.xlsx
python3 -B benchmark.py --source mentor.xlsx --out replay
python3 -B tie_check.py --source mentor.xlsx --out replay
python3 -B exact_primary.py --source mentor.xlsx --out replay
~~~

PROTOCOL.md locks the main outcomes and independent random streams before evaluation. The source article, codebook and tab structure were inspected first. The primary comparison trains a regularized logit on cohort numbers 1 and 2 and tests on cohort number 3. The regularization penalty is applied to every coefficient except the intercept. Scores see weak reports, mutual eligibility and competition summaries. The fitted policy never sees a strict target-cohort tie refinement. The additional leave-one-out panels train on two other cohorts; these are exploratory, not additional primary holdouts. Sampling over reported ties imposes a lottery assumption on an observed profile. It adds no new mentoring preference observations. The conditional DA comparator knows the tie-drawing model and independently draws 4,096 refinements per cohort for prediction. The full-report comparison recovers the simulation's target; it does not re-create an authoritative program assignment.

## Evidence files

- support.csv gives the numbers of submitted and eligible rows/columns and the number of mentees over the five-entry budget.
- results.csv gives every policy–cohort–quota estimate, fraction levels and percentage-point paired differences against uniform with conditional 95% Monte Carlo intervals. blocks.csv contains integer sums in 32 independent blocks of 256 refinements. Each field named ref counts full-DA-matched repeated mentee instances; hit/hits counts those whose full DA partner remains visible. total includes all original mentees over refinements. Lost and gained refer to changes in matched status relative to the full reference; any_loss and any_gain count refinements with at least one such change.
- fits.json contains all learned coefficients for rank and graph models.
- primary-capacity.csv reports expected reachable distinct mentors, maximum feasible placements and DA placements on the exact menus/refinements used in the paper. Capacity and placement changes use all 29 mentees as denominator. The difference of a maximum-count change and a shortfall change equals the change in placements, all in percentage points.
- TIE_SENSITIVITY.md records the follow-up motivated by the primary results. sensitivity.csv contains both the fixed-menu and rerandomized-menu-tie policies under a fresh set of evaluation refinements. Its levels and interval endpoints are percentages or percentage points. The fixed-tie numbers in this file can differ slightly from the primary estimates because the evaluation seeds change.
- Both phases verify bidirectional acceptability, individual matching consistency and absence of visible blocking pairs. The primary phase additionally checks 2,400 selected rollout assignments against an independent sequential queue implementation. The capacity phase compares a small augmenting-path routine against SciPy sparse maximal-cardinality matching.

A special feature of this release is how little truncation five choices induce. In the two training scenarios every mutually acceptable row already has length at most five. The five-slot policies are identical there. Only 15 of 29 mentees in the numbered third cohort lose any alternatives. It is possible to learn partner frequencies on the training scenarios even when no shortlist truncation is needed at k=5, because competition and ties affect the full-report matching there.
