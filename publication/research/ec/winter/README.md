# Do the price-weight spillovers transfer to a different rental season?

We held all renter rankings at price coefficient 4 and private noise 0.5, as selected from the summer calibration. We varied the price coefficient of the observable recommendation score for higher-income renters only. Lower-income renters' individual lists are identical across all three arms. On four-borough January–March 2026 ordinary top-five markets, the lower-group match rate was 3.07% under all-uniform lists, 4.57% under a scoring coefficient of 1 for higher-income lists, and 2.56% under coefficient 4. The primary paired comparison between scoring coefficients equals **+2.01 percentage points** (pointwise 95% Monte Carlo interval 1.85–2.18). Its simultaneous interval is 1.77–2.25. Both individual spillovers relative to all-uniform have simultaneous bands excluding zero in opposite directions.

| Catalogue design and display quota | All uniform | Higher-group score 1 | Higher-group score 4 | Score 1 minus score 4 | Simultaneous 95% interval |
|---|---:|---:|---:|---:|---:|
| Ordinary, top five | 3.07% | 4.57% | 2.56% | +2.01 pp | [1.77, 2.25] |
| Ordinary, 40% | 2.02% | 3.19% | 1.60% | +1.58 pp | [1.32, 1.84] |
| Enriched, top five | 39.90% | 53.08% | 48.76% | +4.32 pp | [3.92, 4.72] |
| Enriched, 40% | 38.22% | 43.70% | 36.75% | +6.95 pp | [6.52, 7.38] |

The last two columns use the unrounded levels. The code verifies the January, February, March and July values used for inflation adjustment against the team's observed CPI reference.

## Data and target population

We froze DESIGN.md (SHA-256 23a0328ddbc05e1d91353c38a51e536c50876a254b8442903e1fb8730133c089) before replaying this counterfactual on the earlier listings. The same winter source files were previously used to study a cheapest-first rule with another model. The winter score-only test uses independent seeds 2026111700–2026111779 and the same summer calibration households as the primary EC experiment. Feature scaling, rent references, PUMA centroids and neighborhood labels are computed exclusively from the three summer development catalogues before the filtered winter files are loaded. Only calibration-group survey donors enter simulated outcomes.

Across January–March the source contains 54,110 arrivals. The ordinary calibrated pipeline leaves 47,873 records in the four boroughs. We exclude another 2,100 source records sharing a summer ID or normalized unit, 1,059 cross-winter or within-source-identity repeats, and 24 records with ambiguous units. Remaining totals are 14,784 in January, 13,310 in February and 16,596 in March. These 44,690 distinct winter records can reappear across simulated draws. The cross-period normalized physical key is borough, uppercase alphanumeric street address, and uppercase alphanumeric apartment number, excluding missing tokens. Both raw-source and geographically assigned winter borough labels are tested against summer source keys. Listing identifiers are also compared against raw summer IDs. Spelling variants can conceal physical-unit overlap. Monthly asks describe arrival records rather than contemporaneously open dwellings. We preserve the team model's original default affordability band and distinct-listing criteria in enriched draws. All twelve borough-month cells support both sampling schemes and all 80 blocks. Each design has 960 paired market draws, each with both display quotas and all three policies. The ordinary counts are 25,975 lower and 35,465 higher renter draws. The enriched sample contains 30,720 in each group. Simulated participants, priorities, and outcomes differ from observed tenancies.

Within each group we pool numerators and denominators in each borough-month before applying common recent-mover ACS borough shares and equal month weights, normalized over the four supported boroughs. The percentile intervals resample whole random seed blocks with replacement 10,000 times, keeping both groups, designs, markets, quotas, and policies together. The secondary simultaneous family uses the maximum absolute bootstrap deviation divided by each contrast's own bootstrap standard deviation across all twelve lower-income contrasts. The intervals quantify Monte Carlo uncertainty conditional on the data and model. They do not quantify field treatment effects or season-to-season variation.

## Validation and replay

Every one of the 11,520 DA outcomes passed an explicit visible-edge blocking check. For the first and last block of each cell, quota, and policy, a separate simultaneous-round implementation agreed with both complete partner vectors (288 outcome checks). For the first blocks, calculating the coefficient-1 transparent order by adjusting the coefficient-4 score replicated the native coefficient-1 score and selected lists. The latent preference and priority matrices used in all assignments belong to the unchanged coefficient-4 market.

Use Python 3.13 with the calibrated research implementation from the paper's source package and its prepared June–August 2026 inputs. Source URLs and checksums for winter files and the BLS snapshot are in sources.json. The official CPI-U values are 325.252, 326.785, 330.213 and 333.918 (January, February, March and July 2026). Supply summer and winter raw CSV files as separate directories, with names YYYY-MM.csv. Geographic data, the summer cleaned catalogue, CPI table and renter cache must be prepared under the implementation directory before running. The command writes only aggregate counts, exclusions, fingerprints and summary estimates.

~~~sh
python run.py --repo /path/to/research/implementation --summer-raw /path/to/raw/june-to-august --winter-raw /path/to/raw/january-to-march --out replay
~~~

For a quick numerical replay, copy counts.csv.gz and inputs.json into replay and rerun the same command with --summarize-only. No raw inputs are accessed in that mode.
