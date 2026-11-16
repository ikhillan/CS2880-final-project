# Coordinated vacancy referrals after provisional offers

This experiment takes a learned shortlist and the assignments it produced as its starting point. Applicants who received nothing can swap one old slot for a seat at a program they actually listed. The platform directs that referral using the observed provisional vacancy inventory. Theorem and proof are in theorem.tex. With fixed reports and capacities, an integral residual matching gives a stable enlarged witness under every subset of participants who accepts. A full-market DA rerun can be used instead of separately allocating seats. The maximum is sharp for expansions admitting a witness that preserves the original contracts. A positive answer does not imply that a decentralized public vacancy feed would produce it.

The 2026 rank-informed baseline left 30,866 applicants unmatched and 21,061 vacancies. Only 160 people could reach any vacancy, and a minimum cut bounds their access at 147 places. The reference 147 comprise 140 people already placed by full-list DA and seven who were unmatched by that benchmark. Independent applicants taking their favorite vacant program gained 156 placements while 11 tentative recipients lost places. Serial reservation, residual DA and a cardinality-maximizing flow improved placement by 144, 143 and 147, with no tentative recipient lost. Compare Lepe (2025), which runs a vacancy submarket for unassigned students and extends the option set to nearby schools inferred to be attractive. Here only previously submitted, pruned options can be recovered. The welfare control allows baseline recipients to move only to a preferred program and computes an optimistic maximum while ignoring priorities.

## Code and replay

Python with NumPy, pandas and SciPy, C++17. Component paths are command-line arguments. In the delivered archive, set U to research/observed/university (containing run.py), INPUT to the raw regeneration directory from its fetch.py, FEATURES to the locally prepared output of prepare_offer.py, and FIT to the saved/replayed six model prediction files or their ensembles. The numeric score used in either year's policy ignores applicant identifiers, which serve as joins only. Reproducing the exact archival neural ties uses the original H200 probabilities. When starting from weights use the supplied replay_offer.py in the matching component. Near-tied probabilities can change a display under another hardware backend. No individual-level field records or program rankings are supplied here.

~~~sh
mkdir -p regenerated
OPENBLAS_NUM_THREADS=1 python repair.py "$U" "$INPUT" "$FEATURES" "$FIT" regenerated
OPENBLAS_NUM_THREADS=1 python referral_rank.py "$U" "$INPUT" "$FEATURES" "$FIT" regenerated
OPENBLAS_NUM_THREADS=1 python upper.py "$U" "$INPUT" "$FEATURES" "$FIT" regenerated
OPENBLAS_NUM_THREADS=1 python verify.py "$U"
OPENBLAS_NUM_THREADS=1 python rental.py "$PANELS" "$LEARNED" regenerated
~~~

PANELS is research/ai/panels, LEARNED is research/ai/learned. For the original development layout use work/ec-ai-learn, outputs/ec-ai-revision/learned, work/ec-observed-university, with its features and fit subdirectories. All quantities describing admissions outcomes under changed recommendations use modeled DA, not observed changes in admissions or actual enrollments.

* PROTOCOL.md records the scope before any vacancy outcomes were computed. WELFARE-CONTROL.md and RELATED-BASELINE.md are explicit follow-ups after primary PAES results.
* panel.csv and reachable.csv contain all PAES policy/market combinations. Columns include additional assignments (gross_gain), losses among the baseline's recipients (gross_loss), ordinal declines including lost placements (old_worse), residual eligible applicants, residual vacancy counts, and Hall/minimum-cut witnesses. No benchmark samples were dropped.
* sensitivity.json reports voluntary take-up draws, 17 random tie resolutions for maximum flow and forecast-only policies. Either precision estimate is conditional on one cohort. Priorities with score noise are a mathematical stress test. Selection across maximum-size extensions is not unique.
* upper.csv records the exact cardinality maximum permitting incumbents to move to a program they rank at least as high while ignoring program-side stability. These are optimistic welfare-feasible upper bounds, not implementable DA recommendations.
* rental-summary.csv gives gains by income group and geographically standardized rate changes by design and season; rental.csv.gz archives market-level counts. In the lower-income-priority arm, integer assignment weights lexicographically maximize total cardinality and the number of lower-income new recipients. Neither income classification nor simulated renter ranks are field outcomes.
* checks.json contains the independent toy enumeration receipts. Each PAES DA result was checked for stability and feasibility. The primary baseline and every main second-stage policy were replayed with an independent simultaneous-proposal solver. Each certified witness also passed a full revised-display stability check; every realized applicant rank was compared with that witness. The flow check verifies equal primal/dual values and the rank-floor upper bound verifies an explicit matching. The rental code compares independent assignment solvers on each market.

RANK-CONTROL.md records the additional check of referral quality in the primary cohort. referral-ranks.csv gives the exact maximum extension and its minimum sum of submitted ranks at all rank ceilings from one to twenty. At five, ten, fifteen and twenty the maxima are 4, 61, 109 and 147. A fixed cutoff is a scenario, not observed take-up.
