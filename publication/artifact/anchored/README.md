# How much does it matter which five listings a renter sees?

In the four-borough NYC development simulation, showing the cheapest five eligible units gave at least one recommendation to the same renters as showing five spread across the market. The first rule displayed an average of 11.26 distinct listings across 64 renters. The second displayed 63.26. Under independent listing priorities, the lower-income match rate rose from 1.32% to 6.25%. Among lower-income renters with at least one option, the mean asking rent on the shortlist rose by $23.16. Among higher-income renters the additional cost of the displayed options was much larger, $1,580.50. All displayed listings still passed the same affordability and bedroom rules. Displayed asking rents measure what the renter sees. These differences do not measure rent paid.

## Three ways to fill the same shortlist

Each renter receives exactly `min(5, number of eligible units)` listings.

- **Cheapest.** Take the lowest asking rents. Break ties using the original market's shared random listing order. Uniform samples eligible units without replacement.
- **Exposure-balanced.** At the start of the market all listings have exposure count zero. Run five rounds. In each round visit renters in an independently shuffled order. Show each renter the eligible unit with the fewest exposures so far among those absent from their list. Break ties in the cheapest order, then increment the selected listing's count immediately. Skip a renter who has exhausted all eligible units. Only eligibility, prices, and a random order enter this rule. Its independent random stream is the previously unused ninth child of the original market seed. Both priority specifications use the same shortlists.
- **Anchored, a full-information oracle.** Run renter-proposing DA with full preferences and listing priorities. If a matched renter's resulting partner is absent from the cheapest list, replace its last slot with that partner. An unmatched renter keeps the original cheapest list. Anchoring uses full preferences to construct a diagnostic reference. It requires information unavailable to a service that only knows prices and eligibility.

After shortlist construction, all rules run ordinary renter-proposing DA using the same underlying renter preferences and priorities on the visible edges. The full reference exposes every eligible listing.

## Independent listing priorities

L and H denote households below or at the income cutoff and above it. Each cell with a slash gives L/H. Match, own blocker and assignment change are percentages of **all** renter draws within group after standardization. Own blocker means the renter is an endpoint of at least one full-graph blocking pair. Assignment change is relative to full-graph renter-proposing DA and includes changes in unmatched status. The last column counts full-stable markets out of 480.

| Rule | Matched, % | Own blocker, % | Assignment change, % | Mean displayed rent among positive-degree renters, $ | Full-stable markets |
|---|---:|---:|---:|---:|---:|
| Full | 4.35 / 91.94 | 0 / 0 | 0 / 0 | 1,791 / 3,510 | 480 |
| Uniform | 4.89 / 90.24 | 0.96 / 74.54 | 2.96 / 82.02 | 1,792 / 3,511 | 0 |
| Cheapest | 1.32 / 21.37 | 1.89 / 88.73 | 3.42 / 88.11 | 1,773 / 2,336 | 0 |
| Exposure-balanced | 6.25 / 94.38 | 0.79 / 73.41 | 3.87 / 82.21 | 1,796 / 3,917 | 0 |
| Anchored | 4.35 / 91.94 | 0 / 0.14 | 0.02 / 0.07 | 1,775 / 2,566 | 473 |

Balancing vs cheapest increased matched shares by 4.93 percentage points for L (paired 95% Monte Carlo interval 4.58 to 5.29) and 73.01 for H (72.41 to 73.59). Own-blocker incidence fell by 1.10 points for L (decrease 0.96 to 1.25) and 15.31 for H (14.72 to 15.89). The additional mean displayed rent had paired intervals of $20.25 to $26.27 for L and $1,555.97 to $1,603.21 for H. The same shortlist price estimates apply under both priority rules.

Relative to uniform, balancing increased the lower-income matched share by 1.36 points (1.11 to 1.61), reduced own-blocker incidence by 0.17 points (decrease 0.08 to 0.25), and raised mean displayed rent by $4.44 ($3.18 to $5.75). Among higher-income renters the corresponding differences were +4.14 points (3.81 to 4.47), −1.12 points (decrease 0.47 to 1.75), and +$406.19 ($391.39 to $420.28).

The anchored rule replaced 369 lower-income slots and 14,353 higher-income slots across the 480 heterogeneous-priority markets. After standardization, these were 2.80% and 16.67% of slots, respectively. Exactly 14 renters (3 L, 11 H) obtained a different full-list assignment, all strictly preferring their oracle assignment to their full-DA assignment. The seven unsettled markets had 24 blocking pairs involving 24 distinct renter draws, all higher-income. Match counts alone could not detect these differences.

## Common listing priority

| Rule | Matched L/H, % | Own blocker L/H, % | Assignment change L/H, % | Full-stable markets |
|---|---:|---:|---:|---:|
| Full | 5.23 / 92.11 | 0 / 0 | 0 / 0 | 480 |
| Uniform | 4.97 / 88.58 | 0.93 / 74.35 | 2.88 / 82.06 | 0 |
| Cheapest | 1.55 / 21.08 | 1.86 / 88.36 | 4.23 / 89.44 | 0 |
| Exposure-balanced | 6.36 / 93.48 | 0.74 / 72.24 | 3.21 / 81.88 | 0 |
| Anchored | 5.23 / 92.11 | 0 / 0 | 0 / 0 | 480 |

Uniform exposed an average of 54.57 distinct listings per market. Full visibility exposed 63.94 (some sampled units were infeasible for everyone). Anchoring exposed 37.29 under independent priorities and 37.54 under common priority. Balancing used neither preference model to set exposures.

## Why the oracle matches exactly the same agents as full DA

Write D for renter-proposing DA on the full graph E. On any visible subgraph H containing D, D remains stable. Deleting other edges cannot introduce a blocking pair for it. Under strict one-to-one preferences, all H-stable assignments have the same sets of matched renters and matched listings. Hence DA(H) matches precisely those matched by D. Renter optimality on H gives every renter an assignment weakly preferred to D. If DA(H) is also E-stable, renter optimality on E gives the reverse comparison for every renter, so DA(H) equals D. The converse is immediate. With a common listing priority, the serial priority order selects the same partners on H by induction whenever D remains visible.

Full stability concerns every eligible renter–listing pair. Truncation can match more agents than full DA while leaving mutually preferred deviations. The experiment records both outcomes.

## Estimand and reproduction

For comparison with the winter replication, `display_rent_cell_conditional` averages the displayed rent of positive-degree renters **within each borough-month and income group**, then weights the 12 cell means by the study's month and borough weights. The original summer estimates above condition after standardization. Conditioning first gives each supported cell its fixed weight even when eligibility is scarce there. Every reported summer estimate, including all 5,000 paired bootstrap draws, had at least one positive-degree renter in every cell and group.

| Summer displayed rent, July 2026 dollars | Lower income | Higher income |
|---|---:|---:|
| Cheapest | $1,805.19 | $2,336.94 |
| Uniform | $1,816.57 | $3,511.64 |
| Exposure-balanced | $1,819.27 | $3,918.13 |
| Balanced minus cheapest (95% Monte Carlo interval) | +$14.08 ($12.65 to $15.39) | +$1,581.19 ($1,556.99 to $1,603.59) |
| Balanced minus uniform (95% Monte Carlo interval) | +$2.69 ($1.98 to $3.36) | +$406.50 ($391.79 to $420.47) |

The original 40 independent seed blocks each contribute 12 markets (three months by four boroughs). Within each borough-month cell and income group, sum all 40 blocks before taking renter means. Weight the 12 cell means by equal month weights and the original four-borough recent-mover weights. For displayed prices, first compute the arithmetic mean of each renter's displayed asking rents, setting it to zero when they have no eligible listings. Divide the standardized mean of this quantity by the standardized positive-degree share, producing the conditional mean in the same standardized population. The slot replacement fraction divides standardized replacement counts per renter by standardized displayed slots per renter. A replacement counts once per newly added listing relative to the cheapest mask. For the untruncated full benchmark the corresponding field counts all additions and has a different quota. Raw stable-market frequencies weight the 480 markets equally. Paired 95% percentile intervals resample the same 40 seed blocks 5,000 times, recomputing the standardized rates each time. These intervals quantify simulation variation conditional on the fixed development catalogue and preference specifications.

Run from the artifact directory using the same data preparation as `nyc-matching-dev`. The script uses that directory's DA routines and the `nyc-costs` hash-validated input loader (NumPy and SciPy). It checks exact raw counts and standardized rates for the original uniform and cheapest runs and checks the full and anchored invariants in every market. It opens only the three development catalogue months and selects only records labeled calibration from the donor file.

```sh
OPENBLAS_NUM_THREADS=1 python3 anchored/run.py --data /path/to/nyc-scarcity/data --catalogue-module /path/to/nyc-scarcity/scarcity.py --out anchored/results.json
```

The runs are exploratory, use simulated preferences, and analyze repeated draws from the development inputs. The additional two rules were specified after the initial development results. They provide no observed lease outcomes or held-out performance evidence.
