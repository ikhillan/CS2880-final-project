# Partial adoption on the archived NYC markets

This is an exploratory stress test on 480 development simulations. A is the original uniform five-listing menu and B is the archived Haiku five-listing menu. The 480 simulated markets are evaluated in each condition and contain 64 renters and 64 listings each. No new recommendation calls or holdout data enter the calculations.

## Rollout assignment and weighting

For each market, the generator is initialized from the first 128 bits of SHA-256 of the prefix `NYC-partial-rollout-v1|` and the original market ID. It generates 64 independent vectors of uniform random numbers, one number for each applicant. In one orientation, a person switches from A to B when their number is below p. In the complementary orientation, they switch when their number is at least 1−p. The same numbers are reused at p = 0, .25, .50, .75 and 1. Both endpoints are evaluated exactly; the three interior levels produce 184,320 full market reruns. Rows identical under A and B are held fixed in all comparisons.

Public output contains counts per market and assignment only. A gain means placed here after being unmatched under A; a loss is the converse. The fixed subgroup includes applicants whose exact option sets are identical under A and B, independent of the realized adoption draw. All displayed percentage rates in this rollout, including gains and losses of fixed applicants, divide by all applicants. Raw counts and subgroup denominators are available separately. Within each supply design we average borough-month proportions using the original ACS recent-mover weights and equal month weights, normalizing over the 12 estimable cells. Staten Island contributes zero market draws. The panel of 240 markets per design comprises 20 seed blocks and covers 97.95% of the intended geography weight.

Coverage is exactly affine *in expectation*: only an individual's own switch determines whether their full-list DA assignment appears. At half adoption, the complementary switches make the mean of a pair exactly equal to the endpoint interpolation in every market. Endpoint coverage and placement reproduce every archived group estimate and both interval endpoints within 1e−10 percentage points. Partial-rollout assignments are newly simulated. The field `interpolation_residual_pp` subtracts (1−p) times the endpoint A rate plus p times the endpoint B rate. The corresponding placement residual may have either sign. At half adoption it was +0.71 points in ordinary markets and +1.60 points in enriched markets.

Each conditional Monte Carlo standard error is the sample standard deviation of the 64 aggregate antithetic pair means divided by eight. The separate pointwise intervals resample 20 entire seed blocks 10,000 times within each supply design and preserve all borough-month cells and endpoints. They include residual numerical Monte Carlo error. No interval covers uncertainty about real housing preferences or repeated independent model prompts.

At p=.5 each applicant's own adoption status differs between paired orientations. For the own-list effect, for each applicant whose menus differ, multiply the placement change from the complementary to the first orientation by +1 if that applicant got B in the first orientation and −1 otherwise. Sum and divide by all applicants. Applicants with identical rows contribute zero by construction. Because each applicant's first-orientation switch is independent of the peer switches and the complements have the same distribution, the estimator is unbiased for the own-row effect on that applicant averaged over independent half adoption by peers. The two observations do not hold the particular peer vector fixed. The estimate describes the effect of assigning one’s own shortlist under this rollout, not the effect of rolling out to the entire market. The duplicate `rate_pct` and `delta_pp` fields for the own-list row both store this signed contrast.

## Assumption checks

Each scenario reuses the same real-data-derived apartment and household instances in their original positions, feasibility and both original menus. We rerun full-list DA separately in each scenario to define its reference partner. The existing market builder holds primitive random shocks fixed as the following parameters vary one at a time.

| Scenario | Change from the calibrated market |
| --- | --- |
| Income-blind listing lottery | Set the listing payoff coefficient on the bounded income-to-rent ratio from 2.5 to 0, keeping the original pair shock. |
| Common income priority | Every listing uses the same order by renter income, with the builder's deterministic tie rule. This replaces the original listing-specific score and noise. |
| Weaker price penalty | Change the renter penalty for rent as a share of budget from 4 to 1. |
| Larger taste shock | Change the renter-specific pair shock coefficient from 0.5 to 1.25. |

The full-adoption coverage gain and placement decline both survive in every scenario at the all-renter level. In the enriched sample, the lower-income placement effect can change sign. Its estimates are −0.92 points under the calibrated model, +2.19 under common income priority, −2.73 under the income-blind lottery, −4.18 under the weaker price penalty and −1.27 with larger taste shocks. These scenarios examine a fixed set of plausible or diagnostic assumptions and do not establish robustness over the space of possible preference and priority specifications.

## Files and rerun

Use the upstream project's Python dependencies and the original three development catalogs, Census renter profiles and committed Haiku cache. Set REPO to an absolute path to the checkout containing `publication/two-studies/evidence/policy-rates.csv`.

```sh
python rental_rollout.py --repo "$REPO" --out replay --pairs 64
```

- `rental-rollout.csv` gives standardized rates, interpolation residuals, conditional Monte Carlo errors and frozen-market block intervals.
- `rental-rollout-counts.csv.gz` contains the 307,200 market-pair-orientation-level aggregate rows (including repeated deterministic endpoints).
- `rental-midpoint-own.csv.gz` contains 30,720 market-pair sums for the midpoint own-row contrast.
- `rental-sensitivity.csv` and `rental-sensitivity-counts.csv.gz` contain overall/group rates, pointwise intervals and the market-specific numerators and denominators.
