# Temporal replication on January–March listings

Switching higher-income renters from random to cheapest-first recommendations reduced the lower-income provisional match rate from 6.65% to 3.19% while every lower-income renter's own list stayed fixed. Among 1,919 lower-income renter instances with one to five feasible listings, 417 lost a match and four gained one. These renters saw every feasible listing in both runs. In the earlier June–August simulation, the corresponding counts were 377 losses and five gains among 1,783 instances.

The primary standardized change was −3.46 percentage points (95% conditional Monte Carlo interval −3.81 to −3.12). The fully exposed renters accounted for −2.95 points (−3.27 to −2.64) of that change after standardization across all lower-income renters. Counts refer to simulated renter instances, which can reuse donor profiles.

| Lower / higher shortlist policy | Lower matched | Higher matched | Lower mean shown rent | Higher mean shown rent |
| --- | ---: | ---: | ---: | ---: |
| Uniform / uniform | 6.65% | 91.76% | $1,741 | $3,361 |
| Cheapest / uniform | 5.79% | 91.81% | $1,726 | $3,361 |
| Uniform / cheapest | 3.19% | 21.09% | $1,741 | $2,193 |
| Cheapest / cheapest | 1.51% | 20.66% | $1,726 | $2,193 |
| Least-exposure / least-exposure | 8.46% | 96.14% | $1,744 | $3,808 |

Least-exposure shows as-yet lightly recommended listings first, breaking ties by price. Relative to all-cheapest, its matched fraction increased by 6.96 points for lower-income renters (6.48 to 7.45) and 75.48 points for higher-income renters (75.02 to 75.92). Mean displayed asking rent increased by $18.10 ($16.67 to $19.47) for the lower group and $1,614.26 ($1,595.05 to $1,634.39) for the higher group. These rents average each positive-degree renter's displayed options, including renters who remain unmatched. Values are in July 2026 dollars. No price paid or lease was observed.

## Listing cohort

The January–March listings were new to the analysis. PROTOCOL.md was frozen and SHA-256 recorded before retrieving the three archives. The protocol is a local record, without an externally registered timestamp. Each raw file contained the listing as first posted. We used the original within-month data cleaning rules, then removed 35 ambiguous identities, 2,108 normalized street/unit keys also found in uncleaned June–August files, and 996 duplicates among the remaining winter records. There were zero listing-ID matches to summer. The resulting 44,668 listings have distinct qualifying keys and IDs across the winter months. Source spellings can still differ for a single physical unit. Address strings and listing identifiers are absent from the aggregate artifact. [FirstMover data documentation](https://www.firstmovernyc.com/open-data)

| Month | Bronx | Brooklyn | Manhattan | Queens |
| --- | ---: | ---: | ---: | ---: |
| January | 372 | 5,817 | 6,228 | 2,360 |
| February | 320 | 5,328 | 5,585 | 2,001 |
| March | 416 | 6,530 | 7,275 | 2,436 |

All 12 cells exceeded the 64-listing threshold. Forty new blocks produced 480 markets, 2,400 verified deferred-acceptance outcomes and 30,720 renter instances (12,866 lower and 17,854 higher). The winter replication shares the summer income calibration distribution, feasibility and preference assumptions, and shortlist rules. Its checks cover temporal transfer to different catalogues under those assumptions. Its conditional intervals quantify seed variation. Remaining uncertainty includes the choice of tastes, priorities and broker behavior.

Winter/July U.S. all-items CPI-U values were 325.252 (January), 326.785 (February), 330.213 (March) and 333.918 (July). The published BLS API values match the official historical table. BLS-snapshot.json preserves the 1,017-byte November 13 API response. The runner reads only these four values. Downloading the API again may change response-time metadata or add later months. [BLS historical table](https://www.bls.gov/regions/mid-atlantic/data/consumerpriceindexhistorical_us_table.htm) · [BLS API series](https://api.bls.gov/publicAPI/v2/timeseries/data/CUUR0000SA0?startyear=2026&endyear=2026)

## Reproduction

Use Python 3.11 or newer with NumPy and SciPy. Run from the manuscript artifact directory with the same prepared calibration donor histogram, matching reference, catalogue loader and development listing snapshots used by the June–August replay. The sibling spillovers, anchored, nyc-costs and nyc-matching-dev directories supply the archived market algorithms. sources.json pins all winter listing and BLS bytes by SHA-256. results.json also pins the shared local inputs. Earlier listing files are needed only to exclude address/unit and public-ID overlaps.

```sh
mkdir -p winter-inputs
for month in 01 02 03; do
  curl -fLsS "https://raw.githubusercontent.com/benfwalla/firstmover-open-data-project/main/public/data/2026-${month}.csv" \
    -o "winter-inputs/2026-${month}.csv"
done
OPENBLAS_NUM_THREADS=1 python3 -B winter-replication/run.py \
  --data /path/to/nyc-scarcity/data \
  --winter winter-inputs \
  --catalogue-module /path/to/nyc-scarcity/scarcity.py \
  --out winter-replication/results.json
```

Each arm uses the same eligible pairs, profiles, lotteries and preferences in each market. Visible stability, feasibility, fully exposed low-degree lists and equal per-person list lengths were checked for every allocation. For each group, results pool counts across blocks within each borough-month. Common recent-mover borough weights and equal month weights then standardize the cell estimates. The 5,000 percentile bootstrap draws resample entire paired blocks. The signed primary and every secondary contrast, with units and Monte Carlo intervals, are in results.json.
