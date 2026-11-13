# What did lower-income rank protection cost in the development markets?

All 480 simulated markets had zero additional audit cost for the specified lower-income group and uniform shortlist. The DA endpoint screen certified 470. We solved the remaining ten exactly. In June Bronx block 26, the cheapest way to reveal the edges of a full-stable assignment cost 15 audits. All 4,096 ways of attaining that bound produced the same unstable DA response. The cheapest executable repair cost 16, with or without group protection.

## Markets and audit actions

We replay the three June–August 2026 calibration catalogues, four boroughs and 40 paired seed blocks used in [the development diagnostic](../nyc-matching-dev/README.md). Each market has 64 renters and 64 listings. The shortlist is uniform over eligible listings, capped at five. Each listing has an independently simulated strict priority. The threshold for a lower-income renter with a shortlist match is their original full-list rank if full-catalogue renter-proposing DA can attain that rank or better. Rank one is best. The test compares each protected renter with their own original match. It does not cover initially unmatched renters or unavoidable rank losses.

Every renter has a separate representative. On the listing side we join the original retained listing ID to its `source_group` and verify unique normalized group names for all retained IDs. We apply Unicode NFKC normalization, case folding, whitespace trimming and internal whitespace collapse. The generic names `Owner` and `By Owner`, blanks, and the literals `NA`, `N/A`, `unknown`, `none` and `null` receive a separate group for each listing ID. Other names retain punctuation and legal suffixes. A shared source can be a feed or management company. These are modeled disclosure groups; we did not validate aliases, common ownership, cooperation or brokerage relationships.

An audit costs one unit per renter representative or listing group. Auditing either endpoint adds all its incident eligible edges to the shortlist. Eligible pairs, full preferences and listing priorities stay fixed. We recompute renter-proposing DA and check for blocking pairs on the original eligible graph. The experiment measures additional units of audit cost for a rank constraint on a repaired outcome. It gives no estimate of real audit costs, observed broker conduct, housing placement or the benefit of auditing for households who cannot afford a listing.

## Exact certificate

For any executable repair, its induced assignment must be full-stable and every assigned edge must be visible. We generate all full-stable assignments in each of the ten cases as follows.

1. Run full-catalogue DA in both directions. The matched participants are the same at both extrema and all intermediate stable assignments. Fix every renter with identical partners at the extrema. Enumerate bijections between the remaining matched renters and their possible remaining listings, respecting the two sides' endpoint rank intervals. At most seven renters vary in these cases. Directly reject any completed bijection with a full-graph blocking pair. Every possible full-stable assignment survives these restrictions.
2. For each surviving assignment, group its hidden assigned edges by listing source. Each edge can be exposed by the renter's singleton representative or its listing source. A renter appears on at most one assigned edge. The groups therefore form disjoint stars. A star requires at least one audit, and auditing its source attains this bound. The minimum cost to reveal the assignment equals its number of stars. If a star has more than one leaf, only its center gives a one-audit cover. For a singleton star either endpoint suffices. For the protected support bound, minimize the star count only over assignments satisfying all thresholds.
3. For each witness of minimum support cost, generate its minimum covers and run DA. A single protected DA response at the ordinary lower bound proves both optima. When the ordinary bound is one below a protected incumbent and every plan at that bound fails full stability, integer costs close the gap. Every possible ordinary repair at the bound must cover some full-stable witness; its plan must therefore be among the enumerated covers. For the exceptional June market there is precisely one witness attaining 15. It has 12 singleton stars and three stars with two leaves, hence 4,096 plans.
4. An independent synchronous, round-based DA implementation and direct pairwise blocking tests check all 4,096 rejected plans and each returned incumbent. For each incumbent we check the complete vector of individual group thresholds. The code stores the assignment, local action IDs, thresholds, all stable witnesses, naive support lower bounds, count of plans examined and the final optimum intervals. There were no solver timeouts. The remaining 470 cases required only the analytical endpoint test, which establishes equality of costs without finding either absolute optimum. Of these 470 passes, 211 had no eligible threshold and 259 had at least one.

## Residual instances

| Month | Borough | Block | Full-stable assignments | Ordinary support minimum | Protected support minimum | Ordinary optimum | Protected optimum |
|---|---|---:|---:|---:|---:|---:|---:|
| June | Bronx | 3 | 2 | 14 | 14 | 14 | 14 |
| June | Bronx | 6 | 2 | 12 | 12 | 12 | 12 |
| June | Bronx | 16 | 4 | 16 | 16 | 16 | 16 |
| June | Bronx | 24 | 2 | 13 | 13 | 13 | 13 |
| June | Bronx | 26 | 3 | 15 | 16 | 16 | 16 |
| July | Bronx | 27 | 2 | 11 | 11 | 11 | 11 |
| July | Queens | 20 | 2 | 24 | 24 | 24 | 24 |
| August | Bronx | 4 | 2 | 14 | 14 | 14 | 14 |
| August | Bronx | 20 | 2 | 19 | 19 | 19 | 19 |
| August | Bronx | 30 | 2 | 15 | 15 | 15 | 15 |

In June Bronx block 26, the failed plans share two full-graph blockers: renters 8 and 9 each prefer listing 22, which ranks each above its DA incumbent. All indices are zero-based within this simulated market. The support calculation alone would have suggested a one-unit premium for protection. Accounting for the platform's actual assignment rule yields a zero premium.

## Reproduction

From the manuscript artifact directory, after preparing the same inputs as the development diagnostic:

```sh
python -B nyc-costs/run.py --data /path/to/nyc-scarcity/data --catalogue-module /path/to/nyc-scarcity/scarcity.py --out replay-costs.json
```

The script requires NumPy and the Python standard library. It checks the six archived SHA-256 digests and reconciles the reconstructed aggregate renter, eligibility, blocker and screen counts against the preceding development run. Only calibration donor rows enter market construction; the other split's profiles are never sampled. Listing source IDs are local integers obtained by sorting normalized source keys in each market, offset by 64. Renter representatives are numbered 0–63. The results are deterministic given the recorded input files and versions. The 480 markets reuse three catalogues; 480/480 is an exact finite-simulation count and does not determine the prevalence of positive premiums in future markets.
