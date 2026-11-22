# GPT shortlists for NYC job seekers: frozen comparison (gpt-6-luna, Aug–Oct 2025)

This is the jobs-side counterpart of the NYC rental runs. It uses Revelio Labs job postings for the New York metro, pulled via WRDS, and ACS 2020–2024 seekers. The question is the same as for rentals: when a shortlist rule shows each seeker a few jobs, how many people end up placed under seeker-proposing deferred acceptance, and what does a stable repair cost? The comparison is between an LLM recommender and random, honest-ranker and steering rules.

The design was frozen on 2026-11-15 in `jobs_manifest.json`, before any frozen-month outcome was seen. The configuration is `configs/jobs_frozen.json`: 4 occupations × 3 months × 20 seeds × 4 variants, giving 960 markets of 24 seekers × 24 postings. Every plan the solver returned passed the full-market stability check. Exact methods certified every optimum.

**Only aggregates are published here, as the Revelio and WRDS terms require.** No posting text, `job_id` or `rcid` appears in this folder, and neither does the saved GPT output, because each answer is tied to Revelio postings. To reproduce, you need your own WRDS access; see *Reproduce*.

## Setup
- **Model:** `gpt-6-luna` through the HUIT API Platform (OpenAI Direct), with `reasoning_effort` set to none and at most 300 completion tokens. Revelio approved this route on 2026-11-13. HUIT approves it for Level 3 data under no-training terms.
- **Prompt:** `hiddenoptions/jobs/llm.py`, structured fields only.
  - Seeker: occupation, years of experience (an age proxy), bachelor's or not, and a pay floor of 0.8 × prior wage.
  - Postings: anonymized ID, posted salary range and seniority label.
  - Not included: employer, posting text or the seeker's wage group.
- **Requests:** 43,596, all valid, sent on 2026-11-15. The API returned the model as `gpt-6-luna` every time. A request is skipped when the seeker's options fit inside the list.
- **Arms:** `llm_k5` shows min(5, feasible) postings and `llm_q50` shows half of them. The comparators are `uniform` (random), `transparent` (the honest ranker by the seeker's own score), `salary_max`, `salary_min` and `under_leveled`.
- **Groups:** lower-wage means a prior wage at or below the NYC weighted median, which is $58,657 in 2024 dollars. In the CSVs these groups appear as `lower_income` / `higher_income`, an alias kept from the housing code.

## Main result: GPT's lists place fewer people than random lists of the same length

Primary variant, 240 markets, in percentage points of seekers. Each difference is the rule minus random on the same markets, with a 95% seed-block interval. From `accuracy*.csv` and `capacity_*.csv`:

| Rule (vs random, same length) | Placed | Placed, lower-wage | Placed, higher-wage | Accuracy |
|---|---|---|---|---|
| GPT, 5 jobs | **−15.5** [−16.6, −14.4] | −22.6 [−24.1, −20.9] | −8.5 [−10.3, −6.6] | +1.6 [0.2, 3.0] |
| GPT, half list | −2.8 [−3.4, −2.2] | −3.5 | −2.1 | +3.4 [1.2, 5.9] |
| Honest ranker, 5 jobs | −42.9 [−44.1, −41.6] | −56.2 | — | −0.5 [−1.9, 0.9] |

- **Levels:** under full information, deferred acceptance places 79.9% of seekers. Random 5-job lists place 77.9%, GPT 62.4%, and the honest ranker 35.1%.
- **Accuracy** is the share of seekers matched under full information whose full-information job was on their list. GPT's edge over random is small, and it is **not robust**: under the calibrated preferences (below) it falls to −0.2 [−1.5, 1.0], and in the held-out replication it is +0.8 [−1.5, 3.3]. Placement is the robust finding.
- **GPT sits between random and the honest ranker:**
  - **Salary:** it does not steer by salary. The mean salary percentile shown is 0.47 for both groups, against 0.86 / 0.72 for the honest ranker (`rec_profile.csv`).
  - **Concentration:** its lists still overlap across seekers. The three most-recommended postings in a market take 37% of all slots (random 27%, honest ranker 56%), and 18% of postings are never shown (random 4%, honest ranker 63%) (`rec_concentration.csv`).

## Why placement falls: the lists cannot hold everyone (`capacity_decomposition.csv`)

Change in placed = change in capacity − change in shortfall. *Capacity* is the largest matching the shown pairs allow. *Shortfall* is the part of that capacity which deferred acceptance does not reach.

| Rule vs random | Placed | Capacity | Shortfall | Capacity share of the drop |
|---|---|---|---|---|
| GPT, 5 jobs | −15.5 | −19.7 | −4.1 | 1.27 |
| GPT, half list | −2.8 | −4.8 | −2.0 | 1.72 |
| Honest ranker, 5 jobs | −42.9 | −51.0 | −8.1 | 1.19 |

The whole drop comes from capacity: concentrated lists send many seekers to the same few jobs. The shortfall actually shrinks, meaning there are fewer collisions than under random lists. A capacity share above 1 means capacity explains more than the full drop.

## One reserved slot recovers most of it (`anchor.csv`)

Each seeker's list keeps its length, but one slot is reserved for a job from a maximum assignment over all feasible pairs. That assignment reuses already-shown pairs where it can, and GPT's last pick is dropped to make room.

| Rule | Lists changed | Placed | Placed with anchor | Gain | Gain, lower | Gain, higher | Anchored vs random |
|---|---|---|---|---|---|---|---|
| GPT, 5 jobs | 20.9% | 62.4 | 75.4 | **+13.0** [11.9, 13.9] | +17.3 | +8.6 | −2.6 [−3.2, −1.9] |
| GPT, half list | 6.2% | 75.6 | 77.7 | +2.2 | +2.4 | +1.9 | −0.7 |
| Honest ranker, 5 jobs | 52.1% | 35.1 | 73.6 | +38.5 | +47.9 | +29.2 | −4.3 |

## Repair cost and protection
- **Cheapest stable repair:** about 3.7 of 12.1 firms for GPT k5, against 3.9 for random (`methods.csv`). Reverse-delete and prefix cover are near-optimal heuristics.
- **Protection** (no lower-wage seeker, or no seeker, ends worse than their first-round match): it costs essentially nothing extra. The premium is 0 in nearly all markets, and at most 1 audit in about 1% of them (`premium.csv`).
- **Repairs leave losers:** after the repair, 17% of seekers under GPT k5 hold a worse-ranked job than their first-round match. This is the same pattern as in housing (`groups.csv`, `worse_off`).

## Sensitivity: calibrated preferences (`calibrated/`, `calibration/`)

The frozen weights (w_wage = 2, σ = 0.75) were pilot values, frozen before any calibration. Calibration was run afterwards, on development months and calibration donors only:
- **Targets:** taken from the Atlanta Fed Wage Growth Tracker. The tracker gives only the median for job switchers, so p25 and p75 come from the all-worker spread.
- **Result:** the calibration selects w_wage = 1, σ = 1.25, a grid corner. The loss is nearly flat across the grid (0.339 to 0.370; 0.358 at the frozen point).
- **The misfit is structural:** every setting gives lower-wage seekers a median raise of about +0.56 log points, against a target of +0.04. That gap comes from feasibility and from comparing ACS annual earnings with posted salaries, not from the weights.

Rerunning the primary variant at the calibrated weights reuses the GPT answers (`experiments_jobs/remap_llm.py`; seekers, postings and feasibility were checked to be identical):

| GPT vs random | Frozen weights | Calibrated weights |
|---|---|---|
| Placed, 5 jobs | −15.5 | −15.3 [−16.3, −14.3] |
| Placed, 5 jobs, lower-wage | −22.6 | −22.0 |
| Capacity share of the drop | 1.27 | 1.24 |
| Anchor gain, 5 jobs | +13.0 | +12.9 |
| Accuracy, 5 jobs | +1.6 [0.2, 3.0] | **−0.2 [−1.5, 1.0]** |

## Held-out replication: Nov–Dec 2025 (`replication/`)

`configs/jobs_replication.json` is the frozen design applied to the two reserved months: 4 occupations × 2 months × 20 seeds × 4 variants, 640 markets. No design choice was changed after seeing the frozen results. Seeds 0–19 are reused, but the markets are new because the month is part of every RNG key.
- **Requests:** 29,390, all valid, all returned as `gpt-6-luna`. They were sent 2026-11-21 23:39 to 2026-11-22 03:40 EST, in two sessions with resume-from-cache, at about $2.40.
- **Checks:** every plan passed the full-market and protection checks, and exact methods certified every optimum. No market skipped the GPT arm.

Primary variant, 160 markets, rule minus random in pp with 95% seed-block intervals:

| | Frozen (Aug–Oct) | Replication (Nov–Dec) |
|---|---|---|
| Placed, GPT 5 jobs | −15.5 [−16.6, −14.4] | **−17.5** [−19.0, −16.1] |
| — lower-wage / higher-wage | −22.6 / −8.5 | −23.0 / −12.0 |
| Placed, GPT half list | −2.8 | −3.3 [−4.3, −2.4] |
| Placed, honest ranker 5 jobs | −42.9 | −43.9 |
| Accuracy, GPT 5 jobs | +1.6 [0.2, 3.0] | **+0.8 [−1.5, 3.3]** |
| Accuracy, GPT half list | +3.4 [1.2, 5.9] | +3.2 [0.8, 5.7] |
| Capacity share of the drop, GPT 5 jobs | 1.27 | 1.28 |
| Anchor gain, GPT 5 jobs | +13.0 | +15.9 [14.8, 17.1] |
| Anchored GPT vs random | −2.6 | −1.5 [−2.4, −0.7] |
| Top-3 share / never shown, GPT 5 jobs | 37% / 18% | 36% / 16% |
| Exact repair cost, GPT vs random (firms) | 3.7 vs 3.9 | 3.8 vs 3.9 |
| Worse off after repair, GPT 5 jobs | 17% | 16% |
| Protection premium | ≈ 0 | ≈ 0 (positive in < 1% of markets) |

Levels: full information places 83.1%, random 5-job lists 80.9%, GPT 63.5% and the honest ranker 37.1%.

**What holds and what doesn't:**
- **Holds:** the placement loss, its size for lower-wage seekers, the capacity explanation, the anchor recovery and the near-zero protection premium. The replication interval for GPT k5 placement doesn't overlap zero and is close to the frozen estimate.
- **Doesn't hold:** GPT k5's accuracy edge over random is not significant in the replication. Together with the calibrated sensitivity, this means the "more accurate" half of the claim holds only for the half-list arm. For 5-job lists the result is "places fewer, about as accurate".

## Caveats
- **Calibration:** the preferences are uncalibrated in the frozen primary, and calibration cannot fit the wage-change target at any grid point.
- **Seeker-side firms:** these are synthetic six-person agents, a shortcut the housing side has since dropped.
- **Experience proxy:** experience is estimated from age.
- **Geography:** the postings span the NY metro, while the seekers live in the five boroughs.
- **Missing pull information:** the share of postings with a stated salary cannot be checked from this pull, which keeps stated salaries only. 10.5% of rows have no employer key.
- **Analyses added after the freeze:** accuracy, capacity/anchor, protection, recommendation profile and calibration were all added after the frozen run, so they are exploratory. They are logged under `later_additions` in `jobs_manifest.json`.
- **Model version:** `gpt-6-luna` has no dated snapshot, so `model_returned` and send times are logged instead.
- **Code version:** the commit closest to the run is `a5b4bb2`; the working tree at run time was not recorded.
- **Replication summary layout:** the replication's `summary.md` comes from the current `experiments/analyze.py`, which has no theory-check section, so `replication/` has no `theory.csv`.

## Files
- `accuracy.csv`, `accuracy_pairs.csv`: accuracy and placement, by group and as paired differences against random.
- `capacity_levels.csv`, `capacity_decomposition.csv`, `anchor.csv`: capacity decomposition and the reserved-slot anchor.
- `rec_profile.csv`, `rec_profile_by_cell.csv`, `rec_concentration.csv`, `rec_overlap.csv`: what each rule shows.
- `groups.csv`, `methods.csv`, `theory.csv`: per-group outcomes, audit-method costs and solver checks for all variants. The empty `children`/`no_children` rows are removed from `groups.csv`.
- `premium.csv`: protection premiums.
- `viability.md`: cell sizes and the four-way partition by group for the frozen months.
- `calibrated/`: the same tables at the calibrated weights, primary variant.
- `calibration/`: the calibration grid and the selected point.
- `replication/`: the same tables for the held-out months (Nov–Dec 2025), all variants.

## Reproduce
You need the Revelio extract at `data/job_postings/raw/revelio_job_postings.csv` (sha256 in `revelio_2025_manifest.json`) and the ACS person file. With HUIT API access:
```sh
python -m hiddenoptions.jobs.llm run configs/jobs_frozen.json --workers 4
python -m hiddenoptions.jobs.llm collect configs/jobs_frozen.json
python -m experiments_jobs.run configs/jobs_frozen.json
python -m experiments_jobs.accuracy configs/jobs_frozen.json
python -m experiments_jobs.capacity_anchor configs/jobs_frozen.json
python -m experiments_jobs.premium configs/jobs_frozen.json
python -m experiments_jobs.recommendation_profile configs/jobs_frozen.json
python -m experiments_jobs.describe_markets configs/jobs_frozen.json
python -m experiments_jobs.calibrate configs/jobs_calibrate.json
python -m experiments_jobs.remap_llm configs/jobs_frozen.json configs/jobs_frozen_calibrated.json
python -m experiments_jobs.run configs/jobs_frozen_calibrated.json   # then accuracy / capacity_anchor / recommendation_profile
# held-out replication: same steps with configs/jobs_replication.json; then
python -m experiments_jobs.premium jobs_replication     # premium.py takes the run name, not the config path
python -m experiments.analyze jobs_replication          # summary.md, methods.csv, groups.csv
```
