# Jobs mirror: experimental setup as implemented

Companion to `FINAL_EXPERIMENT_PROTOCOL.md` and `JOBS_MIRROR_SECTIONS_3-5.md`. This records what the code in `hiddenoptions/jobs/` and `experiments_jobs/` actually does as of the pilot, where it departs from the draft sections, and what is still open. The audit layer (`audit.py`, `matching.py`), the run loop, the bootstrap and the stability gate are the housing code, unchanged.

## 1. Data

**Postings.** Revelio Labs `postings_cosmos` via WRDS, pulled 2026-11-09 with the query in `sql/job_sql_query.sql`: United States, `metro_area = "new york city metropolitan area"`, `post_date` in 2025, `salary_predicted = false`, `onet_code` not null. 3,271,955 rows, 18 columns. Provenance in `revelio_2025_manifest.json`. In code we additionally keep only `remote_type` in {Fully In Office, Partially In Office} (91% of rows) and recompute the salary as the midpoint of `salary_min` and `salary_max`, falling back to Revelio's `salary` where the range is missing. Seniority is Revelio's integer 1 (entry) to 7 (executive), 43% at level 1.

**Seekers.** ACS 2020–2024 five-year person PUMS for the 55 NYC PUMAs, from `scripts/fetch_census_pums.py`. Frame: age 22–64, `ESR` in {1, 2, 3}, `WAGP > 0`, `SOCP` present. Weight `PWGTP`. Prior wage is `WAGP × ADJINC`, in 2024 dollars, deflated to the posting month with the same CPI-U file and `cpi_factor` as the housing side. Seekers live in the five boroughs; postings span the metro. We treat this as commuting and report it as a known mismatch.

**Wage groups.** Lower wage = prior wage at or below the person-weighted NYC median of the frame. This is the mirror of lower income. Education (`SCHL`, bachelor's or not) is retained only as a sensitivity grouping.

**Cells.** A market is drawn from one occupation in one month. The pilot cells are 6-digit SOC codes chosen for wage spread: 291141 registered nurses (173k local 2025 postings), 533032 heavy-truck drivers (72k), 412031 retail salespersons (54k), 311131 nursing assistants (35k). Two-digit groups were tried first and gave about two feasible postings per seeker, because feasibility needs a 6-digit match. High-wage occupations (113031 financial managers, 151252 software developers) are a sensitivity, not primary cells.

**PUMS occupation codes.** PUMS aggregates some occupations: `5330XX`/`533030` is driver/sales workers and truck drivers, `412010` is cashiers, `31113X` is nursing assistants and orderlies. We treat `X` and a sixth-digit `0` as wildcards on the seeker side, so a `533030` seeker belongs to the `533032` cell and is feasible for any `53303x` posting. Seeker counts in the frame: nurses 3,429; retail salespersons 3,933; nursing assistants 1,756 + 117; truck drivers 2,710 (combined category).

## 2. Feasibility

A seeker–posting pair is feasible when all three hold:

1. Occupation: the posting's O*NET code, truncated to six digits, matches the seeker's `SOCP` under the wildcard rule above.
2. Pay: posted salary midpoint ≥ 0.8 × prior wage (both in the posting month's dollars). This is the reservation-wage convention and the mirror of the 40% rent cap.
3. Level: posting seniority ≤ seeker experience + 2 on the 1–7 scale. The condition is one-sided: an experienced nurse can take a staff-nurse job; a new graduate cannot take a level-6 one. Experience is a proxy, `min((age − 22) // 6, 6)`; this is the crudest modeled choice in the setup and is recorded as such. A symmetric version (|gap| ≤ 2) was run first and left 47% of higher-wage seekers with no feasible posting, because higher earners are older, the proxy puts them at level 4–6, and most postings are level 1–2.

There is no education condition: the extract has no education field. There is no distance term: the extract's finest location is the metro.

The pay floor binds higher earners (a $35k seeker qualifies for every posting in the cell; a $120k seeker for the few paying near that), so lower-wage seekers have larger feasible sets than higher-wage ones (current primary: 21.6 vs 11.6 feasible postings per market). The protocol's degree-stratified primary contrast therefore belongs in the main exhibit, and the `nolevel` variant (`level_gap = 99`) reports the pay-floor-only rule. In the stratified design both wage groups are now drawn from seekers with at least one feasible posting in the cell; in housing only the lower-income group carried that condition, because higher-income renters nearly always had options.

## 3. Preferences and firms

Seeker score: $w_w \log(s_j / W_b) + w_q\,Q_j - w_f\,\mathrm{fit}_{bj} + \sigma_r \epsilon_{bj}$, with $Q_j$ the within-market standardized log salary plus a common shock, fit the capped level gap, and standard normal shocks. Pilot weights $w_w = 2$, $w_q = 1$, $w_f = 0.5$, $\sigma_r = 0.75$; $w_w$ and $\sigma_r$ are to be calibrated (§5). The transparent policy ranks on this score with both shocks removed.

Employer priority: $1.0 \cdot \mathbb{1}[W_b \in [0.8\,s^{\min}_j, 1.2\,s^{\max}_j]] + 0.5 \cdot \mathbb{1}[\text{experience} \ge \text{seniority}] + \eta_{jb}$. The pay-band coefficient is swept over {0, 1, 2.5} as the `payband0` and `payband25` variants. No education term.

Firms: the employer (`rcid`) on the posting side, with clustered sampling of three postings per sampled employer as in housing; synthetic seeker-side agents of six seekers each. Unit audit costs. Auditing an employer exposes all its feasible postings to every qualified seeker; auditing a seeker's agent shows that seeker everything, which in the recommender reading is "override the recommender for this person."

Capacity: one opening per posting in the primary run. `expected_hires` (a Revelio prediction) drives a sensitivity via slot expansion, capped at five; slots of one posting share rankings and the employer's firm, and every exposure policy shows all slots of a chosen posting together.

## 4. Exposure policies and audit methods

Each policy shows a seeker min(5, |E_b|) postings (or ⌈0.5 |E_b|⌉ in the `q50` arm): `uniform` (random), `transparent` (honest recommender), `salary_max`, `salary_min`, `under_leveled` (lowest seniority first), and `llm` (saved recommender output, not yet run). Audit methods are the housing set: no audit, full disclosure, reverse-delete, blocker greedy, prefix cover, exhaustive, and the exact Benders solver with prefix and assignment cuts, 15-second cap.

## 5. Months and sampling

From the 2025 pull: development 2025-01 to 2025-07, frozen comparison 2025-08 to 2025-10, held out 2025-11 and 2025-12. A `JobCatalog` loads only the months a config names, so a development run cannot read frozen months. The pilot uses April and May, four cells, n = 24, 20 seeds, stratified design (half lower-wage seekers, half higher-wage, both with at least one feasible posting in the cell; half of postings from the top salary quartile of the cell, `posting_band = "top"`). Variants: `natural` (no stratification), `bottomband` (the housing-style bottom-quartile oversample), `nolevel` (no level condition), `payband0` and `payband25` (employer pay-band weight 0 and 2.5). The posting oversample favors the group the feasibility cap binds: the rent cap binds lower-income renters in housing, and the pay floor binds higher-wage seekers here.

Calibration target for $w_w$, $\sigma_r$: the distribution of $\log(s_j / W_b)$ among matched seekers under full-disclosure DA against job-switcher wage-growth quantiles (Atlanta Fed Wage Growth Tracker; national, a known mismatch). Not yet run.

## 6. Pilot results (960 markets, 2026-11-10)

6 variants × 4 cells × 2 months × 20 seeds, n = 24 per side. Every returned plan passed the full-market stability check. Figures are for the primary variant unless noted; intervals are 95% seed-block bootstrap intervals.

**Solver.** Exact methods (prefix and assignment cuts) certified every optimum. The cheapest stable repair costs 3.9 of 12.9 firms on average, against 12.9 for full disclosure (housing: 4.7 of 10). Reverse-delete reaches the optimum in 92–97% of markets, prefix cover in 76–98%, blocker greedy in 56–82%. `C_support < C_DA` in 0–1% of markets, by one audit (housing: 5%). Adding one audit to a passing plan broke stability in about 1 of 1,450 tries. Cost-optimal plans number 1.4–2.7 per market and give 1.01–1.08 distinct allocations.

**Degree.** Lower-wage seekers have 21.6 feasible postings per market, higher-wage 11.6; 0% and 7% have none in the draw. With k = 5, 75% of lower-wage options are hidden against 44% of higher-wage; under q50 the shares are 49% and 44%.

**Hidden blocking pairs per seeker before audits.**

| Policy | Lower-wage | Higher-wage |
|---|---|---|
| uniform | 2.6 | 1.2 |
| under-leveled | 5.5 | 3.5 |
| salary-min | 10.0 | 4.8 |
| transparent | 11.4 | 4.3 |
| salary-max | 12.3 | 4.7 |

Every steering policy, including the transparent (honest relevance) ranker, leaves several times more blocking pairs hidden than random exposure, for both groups. A relevance ranker shows each seeker their own top choices, usually the highest-paying postings, whose employers rank them low; the pairs that would actually match, ranked moderately by the seeker and highly by the employer, stay hidden. Ranking by what a seeker likes and showing where a stable match lies are different objectives, which is why transparent, not uniform, is the comparator for the LLM arm. The ordering is the same in `natural`, `bottomband` and both payband variants.

**After the repair.** The stable outcome barely depends on the policy: matched shares are 0.96 (lower-wage) and 0.67 (higher-wage) under every policy, because equally cheap plans almost always give the same allocation. The rank-score change therefore measures how far each policy's provisional outcome was from that stable matching.

| Policy (k = 5) | Lower-wage change | Higher-wage change | Higher-wage worse off |
|---|---|---|---|
| uniform | +0.116 [0.105, 0.128] | +0.014 [−0.000, 0.029] | 23–27% |
| under-leveled | +0.261 [0.247, 0.275] | +0.102 [0.085, 0.120] | 17–19% |
| salary-min | +0.482 [0.465, 0.497] | +0.072 [0.053, 0.092] | 21–26% |
| transparent | +0.400 [0.384, 0.416] | +0.179 [0.164, 0.193] | 3–5% |
| salary-max | +0.425 [0.410, 0.440] | +0.193 [0.179, 0.209] | 3–5% |
| uniform, q50 | +0.024 [0.015, 0.034] | +0.028 [0.018, 0.038] | 17–20% |

Two cautions. At k = 5 most of the group gap reflects list length, since lower-wage seekers have more options and so more hidden; under q50 the groups look alike. Report q50 or the degree-stratified contrast next to k = 5. And mean gains hide individual losses: under uniform exposure 24–28% of seekers in both groups end with a worse-ranked job than in the provisional outcome, which replicates the housing result that a cheapest stable repair usually cannot avoid losers.

**Earlier runs, superseded.** (1) Symmetric level rule (|gap| ≤ 2), bottom-quartile oversample, 800 markets: 47% of higher-wage seekers had no feasible posting in the draw, because the age-based experience proxy put older, higher earners at levels 4–6 while most postings are level 1–2. (2) One-sided level rule, higher-wage draw conditioned on options, bottom-quartile oversample, 800 markets: 30% still had none, because the oversample favored the group the cap does not bind. That run is reproduced exactly by the `bottomband` variant above.

## 7. Open items

- LLM arm: `hiddenoptions/jobs/llm.py` (structured fields only; no `company`, no `rcid`, no text) and an OpenAI-format client for the HUIT API Platform. Revelio approved the HUIT AI Sandbox; the API Platform route awaits HUIT's confirmation of the same no-training agreement, then Revelio's. Credits: $10/month Community tier for the pilot, SEAS credit redemption for the full run.
- A jobs `describe_markets.py`: stated-salary share by month and cell, feasible degree by group, employer-portfolio skew.
- Calibration, then freezing `jobs_manifest.json`.
- Experience proxy: age-based; the symmetric level rule showed it can dominate feasibility for older seekers. Revisit if calibration or the LLM arm pattern by age.
- `plan_spread` should map slots back to postings before comparing allocations when capacity > 1.
- `analyze.py` reports all-false `children`/`no_children` rows for jobs; ignore or filter.
