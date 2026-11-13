# Who occupied the listings after the higher-income switch?

Hold every lower-income renter's shortlist fixed at uniform. Switch higher-income shortlists from uniform to the transparent score, preserving each person's display quota. In each of the paired DA outputs, each listing's occupant belongs to one of three classes: vacant, lower-income, or higher-income.

Let n(a,b) count listings whose occupant class changes from a to b. The change in the number of lower-income renters matched equals n(higher,lower) − n(lower,higher) + n(vacant,lower) − n(lower,vacant). The calculation follows directly by subtracting the two totals of lower-occupied listings. It also holds within each simulated market. Counts concern final occupants of matched listings. They do not identify a causal path through the DA proposal sequence.

| Design and budget | Higher→lower minus lower→higher | Vacant→lower minus lower→vacant | Total lower-income change |
|---|---:|---:|---:|
| Ordinary, five | −0.43 | +0.04 | −0.38 |
| Ordinary, 40% | −0.28 | −0.00 | −0.28 |
| Enriched, five | +10.95 | −2.20 | +8.75 |
| Enriched, 40% | −1.08 | −0.10 | −1.18 |

All entries are changes per 100 lower-income renter instances. Totals are computed before rounding. Within each of twelve borough-month cells, sum changes over 20 market draws and divide by the pooled number of lower-income renters. Standardize using the common recent-mover borough weights, normalized to the four supported boroughs and weighted equally across three development months. The JSON reports paired 95% Monte Carlo intervals from 10,000 complete seed-block resamples. The overall point estimates and intervals reproduce the independent round-based DA simulation in the original transparent spillover artifact. The JSON also reports number of distinct listings reached by both groups, with units of listings per market.

The calculation reuses the 480 original development markets. It was specified after seeing aggregate changes. It provides assignment accounting, with no untouched catalogue months, household-choice data or provider calls. Ordinary draws sample renters and listings without an additional affordability screen. Enriched draws oversample affordable listings and restrict the lower-income sample to renters with at least one feasible option in the draw.

Run from a prepared project checkout or pass its explicit path. The reference is the archived JSON for the original transparent spillover analysis.

~~~sh
python -B /path/to/occupancy/run.py --repo /path/to/team-checkout --reference /path/to/transparent_spillover.json --out /tmp/occupancy-results.json
~~~
