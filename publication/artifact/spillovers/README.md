# Can a renter lose a match after seeing every feasible listing?

Yes, in these simulated markets. Hold the recommendations shown to every lower-income renter fixed. When higher-income renters receive the five cheapest eligible listings instead of five drawn uniformly, the lower-income match rate falls from 4.89% to 2.33% under independent listing priorities. Among 1,783 lower-income renter instances with one to five eligible listings, 377 lose a provisional match and 5 gain one. Each sees every eligible listing in both runs. Checking only whether their own shortlist hides a blocking pair would miss all 377 losses.

## What changes between runs?

Each of the 480 development markets has the same 64 renter profiles, 64 listings, one vacancy per listing, eligibility graph, renter preferences and listing priorities across four shortlist policies. Both sides accept every feasible edge over remaining unmatched. Feasibility requires a monthly asking rent of at most 40% of the household's adjusted monthly income and enough bedrooms for household size. All priorities, tastes and shortlist assignments are modeled. Renter-proposing deferred acceptance operates on the visible edges. Lower-income renters have adjusted 2024 household income at or below the project cash-renter median ($64,435.014).

The first policy letter refers to lower-income renters and the second to higher-income renters. U samples at most five feasible listings uniformly without replacement per renter. C shows the five cheapest feasible listings, breaking equal-price ties with one random ordering shared by all renters in that market. Every arm shows all feasible listings whenever a renter has five or fewer. C represents a coordinated ranking convention at the specified list length. It gives no evidence about how any platform or broker currently ranks homes.

| Lower/higher policy | Lower matched | Higher matched | Lower with a full-graph blocking pair | Higher with a full-graph blocking pair | Maximum possible matched, all renters |
| --- | ---: | ---: | ---: | ---: | ---: |
| U/U | 4.89% | 90.24% | 0.96% | 74.54% | 64.26% |
| C/U | 4.38% | 90.27% | 1.39% | 74.56% | 63.77% |
| U/C | 2.33% | 21.54% | 0.82% | 88.65% | 14.30% |
| C/C | 1.32% | 21.37% | 1.89% | 88.73% | 13.52% |

The four outcome columns use independent listing priorities. A renter has a blocking pair if both the renter and the listing prefer each other to their provisional assignments, with being unmatched ranked last. The last column allows any assignment along visible edges, including unstable assignments. It has its own whole-market denominator of 64 and does not depend on priorities. Distinct listings reached by at least one shortlist fall from 87.18% (U/U) to 16.84% (C/C). Even if higher-income renters had the market to themselves, their maximum possible group match rate on their existing lists would fall from 96.79% to 23.00%. Overlap on the cheapest lists thus puts a ceiling on how many renters any allocation rule can place.

## Who accounts for the change?

Take the lower-income group match rate and change one group's shortlists at a time. There are two possible orders. Averaging each group's marginal change over those orders assigns −0.76 percentage points to switching lower-income shortlists (95% Monte Carlo interval −0.85 to −0.68) and −2.81 points to switching higher-income shortlists (−3.05 to −2.57). They sum before rounding to the joint change of −3.57 points (−3.82 to −3.33). These are contributions of policy switches that apply to whole groups. A group switch can change competition even among members of that same group. The analogous contributions with a single shared listing lottery are −0.76 and −2.67 points.

In the U/U to U/C comparison, the lower-income renters with one to five eligible listings account for −2.23 points (−2.47 to −2.00) of the −2.56 point change in the whole lower-income group. Renters with more than five account for the remaining −0.33 points. Renters with zero feasible listings are unmatched under both policies. The paired counts above are unweighted and may contain repeated donor profiles across markets. The percentage-point contribution divides by all lower-income renters within each cell before standardizing, including those with zero options. A common listing lottery also produces losses among the fully exposed subgroup (340 lose, 8 gain).

## Reproduce

Run from the manuscript artifact directory, with local inputs prepared as described in the companion scarcity analysis. Use Python 3.11 or newer, NumPy, and SciPy, plus the scarcity loader and its sibling degree.py.

```sh
OPENBLAS_NUM_THREADS=1 python3 -B spillovers/replay.py \
  --data /path/to/nyc-scarcity/data \
  --catalogue-module /path/to/nyc-scarcity/scarcity.py \
  --reference nyc-matching-dev/results.json \
  --out spillovers/results.json
```

The script reconstructs the 40 original independent seed blocks in each of 12 borough-month cells from June through August 2026. Months receive equal weight. The Bronx, Brooklyn, Manhattan and Queens are weighted by ACS recent-mover shares renormalized to those boroughs. For group rates, divide the sum of the outcome counts within each cell by the sum of instances in that group in that cell, then average using these common weights. Across all 480 markets there are 12,916 lower-income and 17,804 higher-income renter instances. Only calibration donor rows enter the experiment.

A second implementation of deferred acceptance processes proposals in simultaneous rounds, independently of the original queue-based program. Every assignment is checked for feasibility, consistency and visibility. All 3,840 resulting allocations have zero blocking pairs on visible edges. All raw match, blocker and degree-above-five counts in the two original policies reproduce exactly under both priorities, as do their standardized rates within numerical tolerance. The script also asserts that a renter with at most five eligible listings never has an omitted blocking pair. Maximum cardinalities use SciPy's bipartite matching routine, applied separately to each market's shortlist graph and the two group-only induced graphs.

The four arms and both priority rules share each block's market draws. Market seeds use NumPy SeedSequence([20261112, block, month, borough]) with zero-based indices in calendar order and the borough order above. Percentile intervals resample the 40 blocks jointly 5,000 times, seed 19112026, recomputing cell ratios each time. They measure simulation uncertainty conditional on the input distributions, tastes and allocation rules. The analysis was developed after the scarcity results were available. Observed leases, broker relationships and actual recommendation logs were unavailable.

The JSON includes rates and all seven policy contrasts, lower-minus-higher differences, matched/unmatched transitions, graph capacities, source hashes and validation status. Values labeled “pp” are differences in percentage points. “distinct_listings_alone” divides distinct listings by the size of a renter group, so its value can exceed 100. The whole-market distinct listing statistic divides by the number of listings.
