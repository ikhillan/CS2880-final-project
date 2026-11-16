# How much does catalogue size explain the first-quartile mismatch?

Within each of the original 240 June–August calibration markets, we hold 64 renter profiles and their group membership fixed. There are 6,454 lower-income renter instances across these markets and 217 lower-income placements under the selected full-disclosure DA specification. The sensitivity retains that pooled placement count as the catalogue grows. Reusing a household or listing in different market simulations is allowed. Only disjoint pairs within each market count as an assignment. The full-borough case ranges from 572 to 10,306 listings per market and changes the listing-to-renter ratio substantially. It represents a supply-side upper-bound experiment. It does not predict the matching behavior of a larger population of renters competing for the expanded collection of units.

The survey target is the conditional first quartile of ACS contract-rent burden among recent movers in the lower-income calibration group whose burden is within 40%. The experiment uses asks at first advertisement. Costs are in July 2026 dollars, with the original household income and space eligibility rules. Supply, selection into occupied units, renters' outside options and the price measure can all affect a direct interpretation of this comparison.

## Exact quantities

The 25th percentile at 217 observations is exactly the 55th smallest burden. For each threshold, maximize the cardinality of the eligible lower-income matching restricted to edges with burden at or below that threshold. Sum this over the independent simulated markets and compare it with 55. Invert the step function to get a candidate floor. A cover of fewer than 55 vertices in total on the strictly lower graphs proves that no assignment of 217 could have a smaller quartile. Separate witnesses consisting of 217 feasible matched pairs attain every reported floor.

| Listings offered to each market | Capacity at target 21.998% | Exact best Q25 (217 pooled placements) | Best Q25 after removing asks below $1,000 |
|---|---:|---:|---:|
| 32 | 2 | 29.14% | 29.14% |
| Original 64 | 2 | 27.78% | 27.78% |
| 128 | 9 | 26.85% | 26.85% |
| 256 | 16 | 25.29% | 25.45% |
| Whole borough-month | 178 | 18.46% | 19.87% |

The full-catalogue target capacity of 178 is derived from only 13 distinct borough–month–listing identifiers that have an eligible target-burden edge in at least one simulation. No qualifying edge occurs in Manhattan. This count and the original two out of 55 both concern simulated placements. Thirteen refers to different listing records, not distinct tenancies. With sub-$1,000 asks removed, the full-catalogue target capacity decreases to 138 across 11 distinct listing records. The price exclusion is applied after each nested drawing step without replacement.

**Where people may be placed matters.** In the whole-catalogue sensitivity a pooled best Q25 of 18.46% allows changing which of the 240 simulated markets supply the 217 placements. If every market has to preserve its original lower-income placement count, the exact floor is 22.35%. If each original DA recipient has to receive a unit, the floor is 22.46%. At the ACS target, the corresponding valid upper bounds are 43 and 38 cheap placements, respectively, both fewer than 55. Removing asks below $1,000 gives 22.40% and 22.61% floors under these same constraints. We found these constrained floors by threshold matching per market with its own quota and eligible recipient set. The saved assignments satisfy those constraints exactly. Lower-income priority is unrestricted in these comparisons, and higher-income renters consume no units.

## Reproduction

Use the public team repository at commit 981ed97f6b1c517b8aa9a785531c2592a700133c, the prepared June–August 2026 development catalogues and calibration.json from the calibration replay. The helper uses only the three development listing dates and original seeds 0–19. The first 64 listing identities in each market exactly reproduce the original draw, permuted to yield a random nested half-market. The remaining records are permuted with an independently seeded deterministic stream. No preferences are generated for the newly added edges. The original DA outcomes fix the group volumes and recipients. Run on CPU using the same NumPy/pandas/SciPy versions as the team replay.

~~~sh
python nested_support.py --repo /path/to/team-repository --calibration /path/to/calibration.json --out replay
python verify_witnesses.py --repo /path/to/team-repository --calibration /path/to/calibration.json --evidence replay
python verify_flow.py --repo /path/to/team-repository --calibration /path/to/calibration.json --evidence replay
~~~

results.json records the sources and uncompressed data hashes, thresholds, capacities, counts and attainments. witnesses.json stores one primal assignment and two sets of market-level vertex covers per scenario with market-local indices. One set covers all sub-floor edges and the other covers all edges at or below the ACS target. The verification step reconstructs the original draws and checks pair membership, one-to-one feasibility, any fixed market or recipient counts, exact order statistics and cover inequalities without solving an optimization problem. Development estimates are descriptive and one nested extension was drawn for each original market.

For the original and whole-catalogue cases, verify_flow.py independently checks the threshold results at an exact volume of 217. It solves a classic minimum-cost cardinality-constrained bipartite assignment (the LP projection of a unit-capacity flow problem), setting expensive edges to cost one and cheap edges to zero. Market-specific equalities or restricted recipients impose the two stricter conditions. All 24 solves returned integral matchings of cardinality 217, and the optimal count of cheap edges crossed 55 at every stated minimum. flow_check.json records the results.
