# WIP: NYC hidden-options experiments (status as of 2026-11-12)

For teammates and their coding agents. Read this first, then `EXPERIMENT_SETUP.md` (design) and the experiment
protocol `FINAL_EXPERIMENT_PROTOCOL_2.md` (not in this repo; it ships in the protocol author's audit package,
`IJCAI_2027_AUDIT_PACKAGE.zip`).

## How we got here (timeline)
1. **First pipeline (PR #1).**
   - **Built:** renter profiles from ACS PUMS (recent movers), listings from FirstMover (Jun–Aug 2026), feasibility (rent ≤ 40% of income, at most two people per bedroom, same borough and month), modeled preferences, DA, a full-market blocking-pair check, and the exact audit solver (rebuilt from the paper slides because the original code was not available then), plus heuristics and a Batches-API recommender pipeline.
   - **Two flaws found before any real runs:**
     - Renters were grouped under a few synthetic agents, so "audit every agent" opened the whole market cheaply and pilot costs simply tracked the agent count.
     - Rent served as both the quality proxy and the price, so renters effectively preferred higher rent.
2. **Protocol adopted.** A teammate wrote a full protocol (`FINAL_EXPERIMENT_PROTOCOL_2.md`) aimed at an IJCAI 2027 submission (abstracts Jan 4, papers Jan 11, 2027). It adds:
   - renter protection (C_L, C_all, premium Δ_G)
   - a group-switch competition experiment (S_L + S_H = T)
   - rank severity relative to full-information DA
   - uniform plus enriched sampling, held-out months, a transparent policy and open-model arms
   - a 100-renter validation study (needs IRB)
   - cluster scale (H200s for open models)

   The design doc was rewritten to match (PR #2), and all 21 published FirstMover months (2025-02..2026-10) were downloaded, with a coverage check.
3. **Protocol implemented** on `feature/protocol` (details below), then validated against the protocol author's original solver and archived markets.
4. **Scarcity observation** (from small runs, later confirmed by the pilot). This summary was sent to the team:
   - Lower-income renters are basically left with no options. At 40% of income, the highest affordable rent at the lower-income cutoff is $2,286/month, and most lower-income households can afford less. Median asking rents are about $4,800 in Manhattan, $3,900 in Brooklyn, $3,300 in Queens and $3,100 in the Bronx.
   - The original enriched design (half the listings from the cell's cheapest rent quartile) can't always be built. Manhattan's cheapest quarter runs up to about $3,700, which is still unaffordable.
   - This doesn't affect the theory. It does affect the social-impact part: lower-income renters mostly have no affordable options at all, so there is technically nothing for a recommender to hide. It is more a scarcity problem than a hidden-options problem.
5. **Framing discussion.**
   - A teammate asked whether the finding is "AI breeds inequality." Our position: **we can't say that.** The observation is a supply problem (lower-income renters mostly can't afford anything), and it exists before any AI. No AI had run at that point.
   - **The testable question:** among renters who *do* have options, does the AI widen the gap? That requires a design where lower-income renters have hideable options, which led to the affordability band and the 40% budget.
6. **Design fixes chosen:** an affordability-based enriched panel plus the 40%-of-options budget. Not doing a 50% affordability arm or bigger markets for now.
7. **Stage 1 run and a first paid Claude test** (results below).

## Where things are
- **Repo:** the GitHub repo `CS2880-final-project`.
  - `main` has PR #1 (pipeline) and PR #2 (design doc and FirstMover coverage) merged.
  - It also has a teammate's job-market mirror (`hiddenoptions/jobs/`, `experiments_jobs/`, `jobs_experiment_setup.md`), pushed directly to `main`.
- **Active branch:** `feature/protocol`, branched from `main` at `a3da0fc`. **All protocol work below lives here.** Check whether it has been pushed before relying on it remotely.
- **Data (git-ignored, rebuilt with the scripts):**
  - **Census PUMS:** `census_pums/*.csv`, from `scripts/fetch_census_pums.py`.
  - **FirstMover listings:** `data/firstmover/{raw,clean}`, built with `scripts/firstmover_download.py` and `scripts/firstmover_clean.py --month YYYY-MM`.
  - **Coverage:** `scripts/firstmover_coverage.py` writes `data/firstmover/coverage_{daily,months}.csv`, which are committed.
- **Environment:** `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`. Tests: `.venv/bin/python -m pytest tests -q`. 120 pass in about 80 s. The archive check skips unless AUDIT_PACKAGE_ZIP points at the audit-package zip (or the zip sits next to the repo).

## Code map
| Path | Role |
|---|---|
| `hiddenoptions/data.py` | ACS renters (70/30 calibration/evaluation donor split by hash), July 2026 dollars, all 21 FirstMover months, `cell_frames` |
| `hiddenoptions/markets.py` | `MarketSpec`, `Catalog`, `build_market`: designs `uniform` (primary) / `enriched` (affordability band) / `enriched_quartile` (old rule); one firm per renter; protocol preferences; `InsufficientSupply` for unestimable cells; audit-cost variants |
| `hiddenoptions/exposure.py` | E0 policies: uniform, transparent, expensive_first, full, llm (with the protocol fallback fill); budgets `k` (min(5,d)) or `q` (ceil(q·d)) |
| `hiddenoptions/llm.py` | Fixed recommender prompt (budget, bedrooms, area plus anonymized listings; no income or household size); pinned `EFFORT="high"`; shared presentation order across model arms |
| `hiddenoptions/matching.py` | `Instance`, renter-proposing DA, full-market blocking pairs, prefixes |
| `hiddenoptions/audit.py` | Exact LBBD solver (assignment / component / prefix cuts, support master), heuristics, plan enumeration, **renter protection** (`protection_sets`, `ProtectedOracle`: C_L, C_all) |
| `hiddenoptions/metrics.py` | Per-group outcomes (Y_b, q, severity vs M_E, unavoidable losses, price percentile, scarcity categories), `group_switch` |
| `experiments/run.py` | Runner; methods incl. `exact_prefix_protect_{lower,all}`, `*_nosupport`; `--shard i --num-shards n` for Slurm; resumable |
| `experiments/analyze.py` | T (cell-weighted, seed-block bootstrap; coverage counts planned-but-unestimable cells), S_L/S_H, protection premium bounds |
| `experiments/calibrate_preferences.py` | Protocol §4 grid w_p ∈ {1,2,4} × σ_r ∈ {0.5,0.75,1.25} |
| `experiments/llm_recommend.py` | `estimate` / `show` / `submit --yes` / `collect` via the Batches API; `--arm`, `--effort`; merges arms into one recs file |
| `experiments/check_archive.py` | Reproduces the 160 archived synthetic markets from the audit package (reads the zip in place) |
| `configs/` | `pilot`, `stage1_pilot`, `stage1_solver_timing`, `stage2_main`, `stage3_protection`, `sensitivity`, `solvers`, `ladder`, `scale`, `claude_test`. Seed ranges are disjoint by stage. |

## Decisions made
- **Firms:** one private firm per renter, plus real brokerages on the listing side, with unit costs.
- **Months:** development is Jun–Aug 2026. Held-out primary is **Sep–Oct 2026** (both have collection gaps, reported openly, with a gap-day sensitivity check). Held-out replication is **Jun–Aug 2025**.
- **CPI:** Oct 2025 has no value (2025 funding lapse) and is interpolated. Oct 2026 carries Sep 2026 forward until it is published.
- **Enriched panel:** an **affordability band** (listings at or below the survey-weighted 75th percentile of G+ budgets in the cell). The old quartile rule is kept as `enriched_quartile`. **Budgets:** top-5 and 40% of options.
- **Calibration:** kept to the protocol grid. Selected **w_p = 4, σ_r = 0.5**, now the defaults.
- **AI arm:** `claude-opus-5-5` at **high** effort, via batches. Server-side refusal fallback is off, which keeps the arm fixed.

## Results so far (development data only; nothing held-out has been looked at)
1. **Solver validation:** on the 160 archived markets, our exact cost matches the recorded enumeration optimum 160/160 for all three cut families. The cheapest-plan sets match (614 masks), and protected costs and both support bounds match. The paper-audit counts 146/158/12/6 reproduce. The tie-dependent "protected master candidate blocks" is 8 vs the archive's 9 (HiGHS vs scipy tie choice).
2. **Calibration:** the selected point is at the grid edge (largest w_p, smallest σ_r). The lower-income fit is poor: simulated 25th-percentile burden is 0.31 vs a 0.22 target, and only about 220–360 lower-income renters are matched. The 40% cap excludes 33% of ACS recent movers.
3. **Stage-1 pilot** (600 markets: 300 uniform plus 300 enriched; 480 ok, 120 Staten Island unestimable). The AI arms didn't run, so **transparent stands in for the recommender**.
   - **Uniform design:** 82% of lower-income renters have no feasible listing (vs 1% higher-income), and 3% are matched under full disclosure. Their q (can hide) is about 0.05 at k5, so T is essentially a higher-income effect.
   - **Enriched design:** every lower-income renter has at least one option, and q is 0.64 at k5 and 0.84 at q40.
   - **T, transparent vs uniform** (percentage points, coverage 0.979 because Staten Island is missing):

     | Design | k5 | q40 |
     |---|---|---|
     | Uniform | +0.9 [−0.3, 2.0] | +12.8 [11.4, 14.3] |
     | Enriched | **+18.6 [17.0, 19.6]** | **+26.4 [23.8, 29.0]** |

   - **Group switch, enriched k5:** S_L = +9.9 and S_H = +8.7 (S_L + S_H = T exactly).
   - **Transparent collides:** everyone gets the same top listings, so higher-income initial match rates fall from 0.87–0.96 to about 0.51–0.54.
4. **Solver timing** (96 markets, 1,728 runs):
   - **Speed:** median about 1 s per exact solve at 64 per side, under 0.2 s at 24. 99% certified.
   - **Timeouts:** 3 timeouts, all in one Brooklyn market under transparent_k5 (gap 1).
   - **Protection premiums:** essentially zero (C_L > C_DA in 0/192 cases, C_all > C_DA in 1/192).
   - **Stage-3 estimate:** about 100 core-hours; plan for about 250.
5. **Claude test** (`configs/claude_test.json`: 2 enriched markets, 218 requests, each renter asked twice):
   - **Reliability:** 218/218 valid replies, with no refusals or invalid IDs.
   - **Cost:** **about $1.30**, roughly $0.006 per request at batch prices (about 1,700 input and 254 output tokens per request).
   - **Repeat agreement:** 87% of picks on average, identical sets for 55/109 renters.
   - **Early hint, not a result:** price percentile of shown listings is about 0.65 for higher-income renters vs about 0.50 for lower-income (uniform picks give 0.5). Claude does not collide like transparent. Hidden-blocker effects are mixed across the two markets.

## Next steps
- **Run Claude on the 480 pilot markets.** This lets T compare Claude with uniform picks instead of the transparent stand-in.
  - **Cost** (batch prices, from the test's measured tokens), for top-5 only (about 26,000 requests) and for top-5 plus q40 (about 55,000):

    | Model | Top-5 only | Top-5 plus q40 |
    |---|---|---|
    | Opus 5.5 high | about $155 | about $330 |
    | Fable 5.1 | about $390 | about $820 |
    | Sonnet 5.5 | about $80 | about $165 |
    | Haiku 5.5 | about $4 | about $8 |

  - **Assumption:** every model uses about as many output tokens as Opus did at high effort. A short test per model would confirm.
  - **Time:** Claude batches usually finish within hours (24 h at most).
  - **Status:** awaiting a go-ahead.
- **Decide on the calibration grid:** keep the protocol grid, or extend it (w_p 6–8, σ_r 0.25–0.35) because the optimum sits at the edge. Extending is a protocol change and needs team agreement; it takes about 15 minutes plus a pilot rerun.
- **Write FASRC Slurm scripts** for CPU array jobs (the runner already supports `--shard`) and the cluster setup. A few hours of work.
- **Write the open-model script** for Qwen and Mistral on H200s, saving replies in the same format as Claude's. A few hours plus a GPU test. Measure throughput on 500 requests before sizing runs.
- **Pick a fallback for the Manhattan affordability band.** It has only about 32–45 listings per month; in Sep 2026 with evaluation donors it's exactly 32, so every such market reuses the same band listings.
- **Generate Claude replies for the stage-3 markets.** `configs/stage3_protection.json` points at stage-2 replies, which don't cover size 24, enriched markets or q40.
- **Share the pilot and Claude results with the team, then freeze the design** before anyone opens held-out months.

## Future steps
- **Held-out runs on FASRC** (after the design freeze):
  - main comparison on Sep–Oct 2026, with the Jun–Aug 2025 replication
  - protection runs
  - sensitivity panel
  - large markets

  Expect about 250 core-hours for the protection runs plus Claude batches, and a few days of cluster time.
- **Still to build:**
  - reviewed brokerage alias map
  - development/held-out physical-unit dedup
  - model order and repeatability panel
- **Job-market mirror:** it still uses six-person seeker agents, the same shortcut fixed on the housing side.
- **Framing caution:** lower-income scarcity dominates under uniform sampling. The pilot shows a "best match" recommender widens the income gap in hidden options in the enriched design. Whether Claude does is untested at scale, and claims about AI need Claude runs, not the transparent stand-in.

## How to reproduce (laptop)
```bash
.venv/bin/python -m experiments.calibrate_preferences --workers 7
.venv/bin/python -m experiments.run configs/stage1_pilot.json --workers 7
.venv/bin/python -m experiments.analyze stage1_pilot --treated transparent_k5 --control uniform_k5
.venv/bin/python -m experiments.run configs/stage1_solver_timing.json --workers 7
.venv/bin/python -m experiments.check_archive            # set AUDIT_PACKAGE_ZIP=path/to/IJCAI_2027_AUDIT_PACKAGE.zip
.venv/bin/python -m experiments.llm_recommend estimate configs/claude_test.json   # free
# paid: ... submit configs/claude_test.json --model claude-opus-5-5 --effort high --yes ; then ... collect
```
Never commit data, `results/` or API keys. Hold-out months stay unopened until the design is frozen.
