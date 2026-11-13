# The cost of stability for the lower-income rent burden

These are counterfactual assignments on the original 240 calibration draws, each comprising an independent 64-renter/64-listing market for one borough and month, June–August 2026. The selected parameters are a renter price coefficient of 4 and private taste noise of 0.5. The resulting full-market deferred-acceptance assignments contain 217 lower-income and 7,755 higher-income placements. The first quartile of pooled lower-income matched burdens is the 55th smallest burden because 1+(217−1)/4=55. Burden means twelve times the listing's monthly asking rent divided by annual income. Renter draws and listing records can repeat across markets.

## Sharp minimum at fixed volume

| Allocation requirements | Best attainable lower-income Q25 |
|---|---:|
| Both groups keep their pooled placement counts | 27.77728% |
| Both groups keep their placement counts in each market | 28.13243% |
| Every originally matched renter keeps a placement | 28.37952% |
| The same renters and the same listings remain matched | 28.37952% |
| Full-market stability | 31.21173% |
| Original renter-proposing DA | 31.21262% |

The stable maximum is **31.21262%**, agreeing with the original DA outcome. The entire stable interval thus spans **0.00089 percentage points**. An optimal stable matching improves the first quartile by less than one thousandth of a percentage point. Allowing an arbitrary allocation of the same renters to the same occupied units could reduce that burden by **2.83221 percentage points** compared with the best stable assignment. This is a descriptive difference between nested feasible sets at one selected market and outcome specification. It does not establish a cross-city or behavioral effect. A lower rent burden is a planner's criterion; lower burdens need not improve the households' ordinal preferences.

## Characterization and algorithms

Fix any list of strict one-to-one markets on disjoint participant instances and let G identify the designated renters. The matched participants in every full-stable assignment agree market by market. Let N be the total number of matched G renters and k an integer order-statistic index in [1,N]. For threshold t and market m, let h_m(t) be the maximum number of G renters assigned an edge of burden at most t by a stable assignment of m. A global stable assignment achieving its kth smallest burden at most t exists exactly when the sum of h_m(t) over markets is at least k. Necessity is immediate upon restricting any assignment to each market. Sufficiency follows by selecting one maximizing assignment in each independent market. Taking minima within each market instead yields the first threshold at which every global stable assignment has its kth smallest burden at most t, hence the worst attainable kth burden. Searching in the finite set of edge burdens gives both exact endpoints. The argument applies to integral order statistics; an interpolated quantile at a noninteger position requires treating both order statistics. The stable edge indicators maximizing h_m(t) are a linear objective on the ordinary integral strict stable-matching polytope. This gives a polynomial algorithm using the known stable polytope and order-statistic threshold reductions, not a new algorithm for finding one stable assignment.

The primary implementation enumerates all full-stable assignments separately by market. It computes both DA extremes. Any renter with the same partner at both extremes is fixed. The remaining renters permute the listings that also changed between the extremes, subject to both sides' inclusive stable-partner rank intervals. Depth-first enumeration with these restrictions covers every possible stable assignment; an independent full-graph blocking check filters completions. The enumeration returns 369 local stable assignments in total, with 154 of 240 markets uniquely stable and at most eight alternatives in any one market. Eighty-six markets have more than one stable assignment, yet just three of the 217 lower-income recipients have multiple stable partners, all in two markets. This provides a preference-only check ahead of cardinal optimization: any two full-stable assignments can differ in their lower-income empirical burden CDF at a common threshold by at most 3/217, or 1.38 percentage points. That count by itself would not imply the observed narrow quantile interval.

For all the minima, the strict-below count is 54 and the at-threshold count is 55. The stable maximum uses the minimum attainable cheap-edge count for these adjacent thresholds instead. Each witness in witnesses.json.gz stores a complete assignment for every market. For the same-renters and same-listings comparisons, threshold feasibility is ordinary fixed-row assignment with unit penalty on lower-income edges above threshold. The group-market-count comparison uses separate outside slots for each group's exact number of unmatched renters, cheapest ahead of any eligible real edge. Thus all private group slots fill; remaining eligible real edges maximize the count of cheap lower-income assignments. For pooled group counts, the lower-group-only matching upper bound rules out 55 strictly cheaper placements; its 55 threshold edges extend by an integral two-group maximum flow to exactly 217 and 7,755 placements. This satisfies the quotas without relaxing higher-income group volume.

A second implementation tests the stable endpoints directly using the stable-matching LP. The variables x are feasible edge indicators. For each edge, p and q are the inclusive prefix sums of x up to that edge on the renter's and listing's preference lists. Their unit interval bounds impose degree constraints. Each stability row requires p+q−x≥1. We verify the integral primal solutions, numerical dual optimality and each market's agreement with enumeration on both sides of the two exact breakpoints. The replay also verifies every archived assignment's feasibility, all appropriate occupancy constraints, full stability where required and its quartile value. No external models or raw household identifiers enter these artifacts.

## Reproduce

With the calibrated inputs prepared, run from this directory:

~~~sh
python -B check.py --repo /path/to/implementation --out replay
python -B verify.py --repo /path/to/implementation --evidence replay --out replay/verification.json
~~~

The scripts rely on the implementation used by the revised paper, NumPy and SciPy. The ordered list of 240 simulation keys in results.json and witnesses.json.gz maps every index to its original market. The study plan was recorded before learning results for alternative stable assignments.

## Exploratory sensitivity to renter preferences

After seeing the very narrow stable interval at (4, 0.5), we computed exact intervals at the eight other original calibration candidates. At each setting, we first enumerated all local stable assignments wherever a lower-income renter could have more than one stable partner. We deduplicated the sorted vectors of local lower-income matched burdens and enumerated their Cartesian product with the other markets' fixed burdens. We evaluated the interpolated sample quartile on each feasible global allocation. This handles noninteger quantile indices without combining order-statistic endpoints from different matchings. There were at most 12,288 different combinations.

| Price coefficient | Private taste noise | Lower-income matches | Min–max Q25 (%) | Width (percentage points) |
|---:|---:|---:|---:|---:|
| 1 | .50 | 351 | 32.2030–32.3553 | .15234 |
| 1 | .75 | 350 | 32.1392–32.2552 | .11598 |
| 1 | 1.25 | 364 | 32.2374–32.3190 | .08165 |
| 2 | .50 | 298 | 31.6566–31.6740 | .01743 |
| 2 | .75 | 314 | 31.7251–31.7492 | .02406 |
| 2 | 1.25 | 321 | 31.7216–31.7251 | .00351 |
| 4 | .50 | 217 | 31.2117–31.2126 | .00089 |
| 4 | .75 | 241 | 31.4880–31.4880 | 0 |
| 4 | 1.25 | 280 | 31.6996–31.7243 | .02468 |

Across settings, 61–98 of the 240 markets admitted multiple stable assignments, and 3–27 lower-income recipients had several stable partners. Varying the stable assignment could change the lower-income Q25 by at most 0.153 percentage points in any of the nine settings. Each setting uses its own matched population, so these widths measure within-setting stable flexibility. Every original DA match count and quartile matched the archived calibration outputs.

~~~sh
python -B grid.py --repo /path/to/implementation --calibration /path/to/calibration.json --out replay/grid_results.json
~~~
