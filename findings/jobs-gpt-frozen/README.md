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
| GPT, 5 jobs | **−15.5** [−16.6, −14.4] | −22.6 [−24.0, −21.0] | −8.5 [−10.2, −6.6] | +1.6 [0.2, 3.1] |
| GPT, half list | −2.8 [−3.4, −2.2] | −3.5 | −2.1 | +3.4 [1.2, 5.7] |
| Honest ranker, 5 jobs | −42.9 [−44.1, −41.6] | −56.2 | — | −0.5 [−1.9, 0.9] |

- **Levels:** under full information, deferred acceptance places 79.9% of seekers. Random 5-job lists place 77.9%, GPT 62.4%, and the honest ranker 35.1%.
- **Accuracy** is the share of seekers matched under full information whose full-information job was on their list. GPT's edge over random is small, and it is **not robust**: under the calibrated preferences (below) it falls to −0.2 [−1.5, 1.0], and in the held-out replication it is +0.8 [−1.4, 3.3]. The half-list edge is steadier, at +3.4 and +3.2 in the two windows, but under the calibrated preferences it drops to +1.6 [−0.0, 3.4]. Placement is the robust finding.
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

The reduction in maximum matching capacity exceeds the placement decline; a smaller deferred-acceptance shortfall offsets part of it. The lists are more concentrated, consistent with a capacity loss. The decomposition is an accounting identity and does not by itself identify concentration as the sole cause or measure the number of collisions.

## One reserved slot recovers most of it (`anchor.csv`)

Each seeker's list keeps its length, but one slot is reserved for a job from a maximum assignment over all feasible pairs. That assignment reuses already-shown pairs where it can, and GPT's last pick is dropped to make room.

| Rule | Lists changed | Placed | Placed with anchor | Gain | Gain, lower | Gain, higher | Anchored vs random |
|---|---|---|---|---|---|---|---|
| GPT, 5 jobs | 20.9% | 62.4 | 75.4 | **+13.0** [11.9, 13.9] | +17.3 | +8.6 | −2.6 [−3.2, −1.9] |
| GPT, half list | 6.2% | 75.6 | 77.7 | +2.2 | +2.4 | +1.9 | −0.7 |
| Honest ranker, 5 jobs | 52.1% | 35.1 | 73.6 | +38.5 | +47.9 | +29.2 | −4.3 |

These are aggregate placement gains. The released jobs tables do not measure the anchor's reference-job coverage, gross exits or individual rank changes. The original jobs inputs and saved menus were unavailable for the November 22 follow-up, so these gaps remain open for jobs. The [completed rental audit](../rental-anchor-access/README.md) applies the same anchor to the fully reconstructed public rental panels: it finds gross exits despite higher placement, and a broader balancing control outperforms the anchor on placement while losing coverage. Rental results do not fill in missing jobs outcomes or establish GPT's superiority to coordinated alternatives.

## Repair cost and protection
- **Cheapest stable repair:** about 3.7 of 12.1 firms for GPT k5, against 3.9 for random (`methods.csv`). Reverse-delete and prefix cover are near-optimal heuristics.
- **Protection:** preventing additional avoidable losses costs essentially nothing extra. The protected lower-wage or all-seeker set excludes people worse off even under full-information, seeker-optimal stable matching; see [`protection_sets`](../../hiddenoptions/audit.py). The premium is 0 in nearly all markets, and at most 1 audit in about 1% of them (`premium.csv`).
- **Repairs leave losers:** after the repair, 17% of seekers under GPT k5 receive a worse outcome than their first-round match, which can include becoming unmatched. These unavoidable losers are excluded from the protection guarantee (`groups.csv`, `worse_off`).

## Sensitivity: calibrated preferences (`calibrated/`, `calibration/`)

The frozen weights (w_wage = 2, σ = 0.75) were pilot values, frozen before any calibration. Calibration was run afterwards, on development months and calibration donors only:
- **Targets:** taken from the Atlanta Fed Wage Growth Tracker. The tracker gives only the median for job switchers, so p25 and p75 come from the all-worker spread.
- **Result:** the calibration selects w_wage = 1, σ = 1.25, a grid corner. The loss is nearly flat across the grid (0.339 to 0.370; 0.358 at the frozen point).
- **The misfit is structural:** every setting gives lower-wage seekers a median raise of about +0.56 log points, against a target of +0.04. That gap comes from feasibility and from comparing ACS annual earnings with posted salaries, not from the weights.

Rerunning the primary variant at the calibrated weights reuses the GPT answers (`experiments_jobs/remap_llm.py`; seekers, postings and feasibility were checked to be identical):

| GPT vs random | Frozen weights | Calibrated weights |
|---|---|---|
| Placed, 5 jobs | −15.5 | −15.3 [−16.3, −14.4] |
| Placed, 5 jobs, lower-wage | −22.6 | −22.0 |
| Capacity share of the drop | 1.27 | 1.24 |
| Anchor gain, 5 jobs | +13.0 | +12.9 |
| Accuracy, 5 jobs | +1.6 [0.2, 3.1] | **−0.2 [−1.5, 1.0]** |
| Accuracy, half list | +3.4 [1.2, 5.7] | +1.6 [−0.0, 3.4] |

## The rental headline, T (`contrast_T.csv`, `replication/contrast_T.csv`)

The rental write-up (`../gpt-luna-rentals/`) leads with **T**. T is how much a rule widens the lower- minus higher-wage gap in the share of seekers with a hidden blocking pair: a job they weren't shown, that would take them, and that they prefer to their match. It's measured against random lists, with no audit. `experiments_jobs/contrast_T.py` computes it with the same definition. One difference: each occupation × month cell counts equally, because rentals weight boroughs by ACS shares and the jobs cells have no comparable weight. The frozen run predates the per-seeker blocker count, so its markets were rerun with no audits (`configs/jobs_frozen_noaudit.json`). The rerun used the same markets and saved GPT answers, made no API calls, and gave outcomes identical to `jobs_frozen`.

GPT 5 jobs vs random 5 jobs, pp, 10,000 seed-block draws. Each design is paired with its rental counterpart, per the manifest:

| Jobs design (rental counterpart) | Frozen | Replication | Rentals, GPT |
|---|---|---|---|
| primary, stratified (enriched) | +0.8 [−1.5, 3.1] | +2.6 [−0.5, 5.7] | +6.7 to +8.5 |
| natural (primary) | **+4.3** [1.9, 6.8] | **+4.2** [0.5, 7.8] | +3.6 (held out) |

- **Natural markets:** in the unstratified markets, GPT widens the gap by about 4 points in both windows, close to the rental primary design.
- **Stratified markets:** in the jobs primary design, the widening is small and not significant in either window.
- **Why the stratified case differs from rentals:** for rentals, T comes from GPT cutting higher-income renters' misses (85% → 81%). For jobs, GPT raises misses for both groups by about the same amount. Lower-wage seekers go from 75% to 84% and higher-wage from 39% to 47% (frozen, primary), so the gap barely moves. On the jobs side, GPT's lists hurt everyone through capacity rather than shifting the advantage between groups.
- **Group levels run the opposite way:** lower-wage seekers miss a better job far more often than higher-wage ones. For rentals it's the reverse. The pay floor binds the other group, which is why the posting oversample favors the other side.

**At the calibrated weights** (`calibrated/T/`), the rentals' preferences are calibrated, so this checks T at the jobs calibration point (w_wage 1, σ 1.25). Configs: `configs/jobs_{frozen,replication}_calibrated_noaudit.json`. GPT answers were reused via `remap_llm.py`, with every market checked to be identical, and no API calls were made.

| Jobs design | Frozen | Replication |
|---|---|---|
| primary, stratified | +1.4 [−0.6, 3.5] | +2.6 [−0.2, 5.4] |
| natural | +3.6 [0.7, 6.5] | +3.0 [−0.7, 6.5] |

The pattern doesn't change: natural markets show a widening of about 3 to 4 points, and the stratified primary shows nothing significant. In the calibrated replication, the natural interval now just includes zero, so call the natural-design T "about +3 to +4, significant in three of four runs".

### Where T comes from: group switch (`group_switch.csv`, `replication/group_switch.csv`)

`experiments_jobs/group_switch.py` runs deferred acceptance four times per market. Each wage group sees random or GPT lists, using the same function as the rental runner, and the script splits T into S_L (switching lower-wage seekers' lists) and S_H (switching higher-wage seekers' lists). Its T equals `contrast_T.py`'s on both runs.

| GPT 5 jobs vs random, pp | S_L | S_H | T |
|---|---|---|---|
| primary, frozen | +6.6 [4.1, 9.1] | −5.8 [−7.3, −4.1] | +0.8 |
| primary, replication | +9.1 [6.6, 11.5] | −6.5 [−8.9, −4.0] | +2.6 |
| natural, frozen | +9.5 [7.7, 11.5] | −5.2 [−6.9, −3.6] | +4.3 |
| natural, replication | +10.8 [7.9, 13.6] | −6.6 [−8.6, −4.6] | +4.2 |

Each group is hurt by its own GPT lists, and the other group is barely affected.
- **Lower-wage seekers:** their misses go from 75% to 81% when their lists switch (frozen, primary).
- **Higher-wage seekers:** theirs go from 39% to 45%.
- **Cancellation:** the two harms cancel in the gap, so the near-zero primary T is not "no effect". In the natural design the lower-wage harm is larger, so T is positive.
- **Contrast with rentals:** there, T comes from GPT lowering higher-income renters' misses (S_H about +7) while lower-income lists don't matter.

The honest ranker shows no significant S_L or S_H in the primary design.

## Held-out replication: Nov–Dec 2025 (`replication/`)

`configs/jobs_replication.json` is the frozen design applied to the two reserved months: 4 occupations × 2 months × 20 seeds × 4 variants, 640 markets. No design choice was changed after seeing the frozen results. Seeds 0–19 are reused, but the markets are new because the month is part of every RNG key.
- **Requests:** 29,390, all valid, all returned as `gpt-6-luna`. They were sent 2026-11-21 23:39 to 2026-11-22 03:40 EST, in two sessions with resume-from-cache, at about $2.40.
- **Checks:** every plan passed the full-market and protection checks, and exact methods certified every optimum. No market skipped the GPT arm.

Primary variant, 160 markets, rule minus random in pp with 95% seed-block intervals:

| | Frozen (Aug–Oct) | Replication (Nov–Dec) |
|---|---|---|
| Placed, GPT 5 jobs | −15.5 [−16.6, −14.4] | **−17.5** [−18.9, −16.0] |
| — lower-wage / higher-wage | −22.6 / −8.5 | −23.0 / −12.0 |
| Placed, GPT half list | −2.8 | −3.3 [−4.2, −2.3] |
| Placed, honest ranker 5 jobs | −42.9 | −43.9 |
| Accuracy, GPT 5 jobs | +1.6 [0.2, 3.1] | **+0.8 [−1.4, 3.3]** |
| Accuracy, GPT half list | +3.4 [1.2, 5.7] | +3.2 [0.7, 5.7] |
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
- **Doesn't hold:** GPT k5's accuracy edge over random is not significant in the replication. Together with the calibrated sensitivity, this means the "more accurate" half of the claim holds only for the half-list arm, and even there it is borderline at the calibrated weights (+1.6 [−0.0, 3.4]). For 5-job lists the result is "places fewer, about as accurate".

## One firm per seeker (`firm1/`)

The frozen design groups seekers into synthetic six-person agents for auditing; the housing side gives each renter a private firm. `configs/jobs_{frozen,replication}_firm1.json` rerun the primary variant with `seeker_firm_size: 1`. GPT answers were reused via `remap_llm.py`, and no API calls were made. Plan enumeration was dropped: it only feeds the optimal-plan spread, which no finding uses. Firm assignment has its own random stream, so seekers, postings, preferences and GPT's lists are unchanged. Every optimum was certified, and every plan passed the stability check.

| Exact repair cost (firms) | Six-person agents | One firm per seeker |
|---|---|---|
| Firms per market | 12.1 | 32.1 |
| GPT 5 jobs vs random, frozen | 3.67 vs 3.86 (−0.19 [−0.24, −0.13]) | 5.66 vs 6.12 (−0.46 [−0.55, −0.35]) |
| GPT 5 jobs vs random, replication | 3.80 vs 3.87 (−0.07 [−0.12, −0.02]) | 6.04 vs 6.34 (−0.31 [−0.49, −0.11]) |
| GPT half list vs random, frozen | 3.21 vs 3.57 | 4.68 vs 5.19 (−0.51 [−0.71, −0.32]) |
| Audited, GPT 5 jobs, frozen | 3.1 seeker agents + 0.6 employers | 0.8 seekers + 4.9 employers |
| Worse off after repair, GPT / random (frozen) | 17% / 24% | 17% / 25% |
| Protection premium | ≈ 0 | ≈ 0 (at most 2 firms, in < 3% of markets) |

- **Cost and who gets audited:** repairs cost more, and they shift from overriding seekers to auditing employers.
- **GPT vs random:** GPT stays cheaper to repair than random, by more than before.
- **Unchanged:** who ends up worse off and the near-zero protection premium.
- **Not affected:** placement, accuracy, T and the capacity results don't involve audits.

Paired differences resample seed blocks with 10,000 draws.

## Caveats
- **Calibration:** the preferences are uncalibrated in the frozen primary, and calibration cannot fit the wage-change target at any grid point.
- **Seeker-side firms:** the frozen design uses synthetic six-person agents; the housing side uses one firm per renter. The one-firm-per-seeker sensitivity (`firm1/`) raises repair costs from about 4 to about 6 firms but keeps every conclusion. See *One firm per seeker* above.
- **Remapped runs redraw random lists:** in the calibrated and one-firm-per-seeker runs, the random and rule-based lists are new draws, because their random seed comes from the market ID. GPT's lists, seekers, postings and feasibility are identical. In the one-firm-per-seeker run, where preferences are also identical, the redrawn random and rule-based lists moved mean placement by at most 0.3 pp.
- **Experience proxy:** experience is estimated from age.
- **Geography:** the postings span the NY metro, while the seekers live in the five boroughs.
- **Missing pull information:** the share of postings with a stated salary cannot be checked from this pull, which keeps stated salaries only. 10.5% of rows have no employer key.
- **What was specified when:** the design was frozen on 2026-11-15 without named outcomes, and every analysis here was coded after that. Their standing differs (`later_additions` in `jobs_manifest.json`):
  - **T, group switch and protection premium** are protocol estimands, named in `EXPERIMENT_SETUP.md` before the jobs freeze.
  - **Accuracy, capacity/anchor, protection, recommendation profile and calibration** were committed before the held-out replication ran, so the replication tests them as specified in advance. Capacity loss and anchor recovery repeat. The five-job coverage advantage remains uncertain in replication, whose interval spans zero.
  - **T and group-switch code, calibrated T and one firm per seeker** came after both windows.
  - **Frozen-month results** of everything except the protocol estimands are exploratory.
- **Model version:** `gpt-6-luna` has no dated snapshot, so `model_returned` and send times are logged instead.
- **Code version:** the commit closest to the run is `a5b4bb2`; the working tree at run time was not recorded.
- **Intervals:** all intervals are 95% seed-block bootstraps with 10,000 draws, as the protocol asks and as the rentals use. `groups.csv` and `methods.csv` hold point estimates only.
- **Table layout:** `groups.csv` and `methods.csv` in both windows come from the current `experiments/analyze.py` (rerun 2026-11-22). Point estimates are identical to the 2026-11-15 tables. The current layout has no theory-check section, so `theory.csv` (frozen only) is from the original run.

## Files
- `accuracy.csv`, `accuracy_pairs.csv`: accuracy and placement, by group and as paired differences against random.
- `capacity_levels.csv`, `capacity_decomposition.csv`, `anchor.csv`: capacity decomposition and the reserved-slot anchor.
- `rec_profile.csv`, `rec_profile_by_cell.csv`, `rec_concentration.csv`, `rec_overlap.csv`: what each rule shows.
- `groups.csv`, `methods.csv`, `theory.csv`: per-group outcomes, audit-method costs and solver checks for all variants. The empty `children`/`no_children` rows are removed from `groups.csv`.
- `premium.csv`: protection premiums.
- `viability.md`: cell sizes and the four-way partition by group for the frozen months.
- `calibrated/`: the same tables at the calibrated weights, primary variant. `calibrated/T/` has T at the calibrated weights for both windows, primary and natural.
- `calibration/`: the calibration grid and the selected point.
- `group_switch.csv`: S_L, S_H and T with all four B levels, for GPT k5, GPT half list and the honest ranker, primary and natural (frozen; the replication copy is in `replication/`).
- `contrast_T.csv`: T for every variant, for GPT k5, GPT half list and the honest ranker, with B levels. Rows cover all seekers, seekers who can hide, and each month. From the no-audit rerun of the frozen markets.
- `replication/`: the same tables for the held-out months (Nov–Dec 2025), all variants.
- `firm1/frozen/`, `firm1/replication/`: `methods.csv`, `groups.csv` and `premium.csv` with one seeker-side firm per seeker, primary variant.

## Reproduce
You need the Revelio extract at `data/job_postings/raw/revelio_job_postings.csv` (sha256 in `revelio_2025_manifest.json`) and the ACS person file. With HUIT API access:
```sh
python -m hiddenoptions.jobs.llm run configs/jobs_frozen.json --workers 4
python -m hiddenoptions.jobs.llm collect configs/jobs_frozen.json
python -m experiments_jobs.run configs/jobs_frozen.json
python -m experiments_jobs.accuracy configs/jobs_frozen.json --reps 10000
python -m experiments_jobs.capacity_anchor configs/jobs_frozen.json --reps 10000
python -m experiments_jobs.premium jobs_frozen --reps 10000
python -m experiments_jobs.recommendation_profile configs/jobs_frozen.json --reps 10000
python -m experiments_jobs.describe_markets configs/jobs_frozen.json
python -m experiments_jobs.calibrate configs/jobs_calibrate.json
python -m experiments_jobs.remap_llm configs/jobs_frozen.json configs/jobs_frozen_calibrated.json
python -m experiments_jobs.run configs/jobs_frozen_calibrated.json   # then accuracy / capacity_anchor / recommendation_profile
# held-out replication: same steps with configs/jobs_replication.json; then
python -m experiments_jobs.premium jobs_replication --reps 10000   # premium.py takes the run name, not the config path
python -m experiments.analyze jobs_replication --reps 10000   # summary.md, methods.csv, groups.csv
# rental-style T (frozen run predates blocker_count, so rerun its markets without audits first; no API calls)
python -m experiments_jobs.run configs/jobs_frozen_noaudit.json
python -m experiments_jobs.contrast_T jobs_frozen_noaudit
python -m experiments_jobs.contrast_T jobs_replication
python -m experiments_jobs.group_switch configs/jobs_frozen_noaudit.json
python -m experiments_jobs.group_switch configs/jobs_replication.json
# T at the calibrated weights (GPT answers remapped, no API calls)
python -m experiments_jobs.remap_llm configs/jobs_frozen.json configs/jobs_frozen_calibrated_noaudit.json
python -m experiments_jobs.remap_llm configs/jobs_replication.json configs/jobs_replication_calibrated_noaudit.json
python -m experiments_jobs.run configs/jobs_frozen_calibrated_noaudit.json
python -m experiments_jobs.run configs/jobs_replication_calibrated_noaudit.json
python -m experiments_jobs.contrast_T jobs_frozen_calibrated_noaudit
python -m experiments_jobs.contrast_T jobs_replication_calibrated_noaudit
# one firm per seeker (GPT answers remapped, no API calls)
python -m experiments_jobs.remap_llm configs/jobs_frozen.json configs/jobs_frozen_firm1.json
python -m experiments_jobs.remap_llm configs/jobs_replication.json configs/jobs_replication_firm1.json
python -m experiments_jobs.run configs/jobs_frozen_firm1.json
python -m experiments_jobs.run configs/jobs_replication_firm1.json
python -m experiments.analyze jobs_frozen_firm1 --reps 10000; python -m experiments_jobs.premium jobs_frozen_firm1 --reps 10000
python -m experiments.analyze jobs_replication_firm1 --reps 10000; python -m experiments_jobs.premium jobs_replication_firm1 --reps 10000
```
