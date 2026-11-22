# Headline findings: recommended shortlists in NYC rentals, NYC job postings and Chilean school admissions

Two simulation studies ask the same question. When an LLM chooses the 5 options each person sees, how does the market turn out compared with 5 random options? Both use `gpt-6-luna` through the HUIT API Platform with the same settings (reasoning off, at most 300 tokens, every reply valid). Both use seeker-proposing deferred acceptance and test on held-out months. A third study asks the same question with observed data: Chilean school admissions, where the lists come from logistic classifiers instead of an LLM (see *Observed check* below). Details and full tables:

- **Rentals:** [`gpt-luna-rentals/`](gpt-luna-rentals/README.md). StreetEasy-style listings × ACS renters, 64 × 64 markets. Windows: development (Jun–Aug 2026), held out (Sep–Oct 2026), replication (Jun–Aug 2025).
- **Jobs:** [`jobs-gpt-frozen/`](jobs-gpt-frozen/README.md). Revelio postings × ACS seekers, 24 × 24 markets in 4 occupations. Windows: frozen (Aug–Oct 2025) and held-out replication (Nov–Dec 2025).
- **Earlier model:** [`claude-haiku-pilot/`](claude-haiku-pilot/README.md), the rental development run with Claude Haiku 5.5.
- **Schools (observed):** [`publication/research/observed/school/`](../publication/research/observed/school/README.md). This is the Chilean school-admission system's regular round. Two ridge logistic models are trained on 2019 and tested on 2020 and 2021 (915,638 applicant-years). It has no LLM.

## The comparison in one table

GPT 5 options minus random 5 options, in percentage points, on the same markets. Ranges span the windows; brackets are 95% seed-block intervals (10,000 draws).

| | Rentals | Jobs |
|---|---|---|
| **Accuracy** (full-information match is on the list) | **+7.9 to +8.1**, all significant | **+1.6 / +0.8**, frozen only significant; −0.2 at calibrated weights |
| **Placed** (share matched from lists alone) | **−2.0 to −3.7** | **−15.5 / −17.5** |
| Placed, group with more options | higher-income −5.6 to −6.7 | lower-wage −22.6 / −23.0 |
| Placed, group with fewer options | lower-income +1.6 to −0.7 | higher-wage −8.5 / −12.0 |
| **T**, matched design (rental primary ↔ jobs natural) | +4.5, +3.6, +4.4 | +4.3 [1.9, 6.8] / +4.2 [0.5, 7.8] |
| **T**, oversampled design (rental enriched ↔ jobs stratified) | +8.5, +6.7, +8.1 | +0.8 / +2.6 (neither significant) |
| What the lists favour | pricier listings for higher-income renters (0.57 vs 0.50 percentile) | no salary steering (0.47 for both groups) |

The accuracy and placement rows use each study's headline design: rental enriched and jobs primary (stratified), the oversampled designs that the manifests pair. Rental values are development, held out and replication; jobs values are frozen / replication.

T is the change in the lower- minus higher-group gap in the share who miss a better option. A "better option" is a hidden blocking pair: an option they weren't shown, that would take them, and that they prefer.

## Three findings

**1. GPT's lists are concentrated, and concentration costs placements in both markets.** On five-option lists, GPT places fewer people than random in every window of both studies. The jobs loss is about five times larger. Two things drive the difference:
- **Accuracy edge:** GPT's 5-job lists are about as accurate as random lists, so nothing offsets the concentration. Renter lists are about 8 points more accurate.
- **Coverage:** five slots cover a larger share of a job seeker's options (about 17 feasible) than of a renter's (about 29).

**The capacity decomposition shows the mechanism in both markets** (jobs: `jobs-gpt-frozen/`; rentals: [`rental-capacity-gpt/`](rental-capacity-gpt/README.md)).

| GPT vs random, pp | Rentals, enriched (3 windows) | Jobs, primary (frozen / replication) |
|---|---|---|
| Capacity of the shown lists | −8.8 to −10.6 | −19.7 / −22.4 |
| Shortfall (what deferred acceptance misses) | −6.3 to −7.3 | −4.1 / −4.9 |
| Placed | −2.0 to −3.7 | −15.5 / −17.5 |
| One reserved slot: gain | +3.5 to +4.9 | +13.0 / +15.9 |
| Anchored GPT vs plain random | +1.5, +0.3, +1.6 | −2.6 / −1.5 |

- **Capacity:** in both markets GPT's lists cut capacity sharply. Concentration means the shown pairs can't hold everyone: in jobs, GPT's top three postings take 37% of slots against 27% under random.
- **Offset:** in rentals, fewer collisions win back about two-thirds of the lost capacity. In jobs only about a fifth comes back.
- **Reserved slot:** it removes the whole rental loss and most of the jobs loss.

The paper's own learned rental model shows the same rental pattern (`publication/research/observed/capacity/`).

**2. The placement loss falls on whichever group has more options.**
- **Rentals:** higher-income renters average 46 feasible listings against 14. They lose 6 to 7 points of placement, while lower-income renters barely move.
- **Jobs:** the pay floor reverses this. Lower-wage seekers have about 21 feasible jobs against 11, and they lose about 23 points.

The rule is the same in both markets: GPT concentrates the long lists, and the people with long lists bear the cost.

**3. GPT widens the group gap in missed better options when markets are drawn as they come; with oversampling it's mixed.**
- **Natural draws:** in markets drawn without oversampling (rental primary, jobs natural), T is +3.6 to +4.5 in every window of both studies. At the jobs calibrated weights it is +3.6 / +3.0 [−0.7, 6.5].
- **Oversampled designs:** the two studies diverge.
  - **Rentals:** in the enriched design, T grows to about +8. Nearly all of it comes from GPT changing higher-income renters' lists.
  - **Jobs:** in the stratified design, GPT raises misses for both wage groups by about the same amount, so the gap barely moves.

**The group switch shows why the markets differ.** Each group's lists are switched between random and GPT separately (S_L + S_H = T; rental `*/group_switch.csv`, jobs `jobs-gpt-frozen/group_switch.csv` and `replication/group_switch.csv`).

| Share who miss a better option | Rentals (held out, enriched) | Jobs (frozen, primary) |
|---|---|---|
| Higher group, own lists switched to GPT | 86% → **80%** (GPT helps) | 39% → **45%** (GPT hurts) |
| Lower group, own lists switched to GPT | 38% → 37% (no change) | 75% → **81%** (GPT hurts) |
| Spillover onto the other group | none | none |
| S_L / S_H (pp) | −0.6 / **+7.3** | **+6.6** / **−5.8** |
| T | +6.7 | +0.8 |

- **Rentals:** the gap widens because GPT's lists *help* higher-income renters find better homes, while doing nothing for lower-income renters.
- **Jobs:** GPT's lists *hurt* each wage group through its own lists, by similar amounts. The replication gives S_L +9.1 [6.6, 11.5] and S_H −6.5 [−8.9, −4.0], with T +2.6.

So a near-zero jobs T is not "no effect". It's two significant harms that cancel in the gap. In both markets, essentially all the movement comes from a group's own lists, not from competition with the other group.

## Accuracy is not the same thing as placement

The team's claim was that a recommender can be more accurate and still place fewer people.
- **Rentals:** it holds cleanly against random lists.
- **Jobs:** it holds only for half-length lists (+3.4 / +3.2 accuracy, −2.8 / −3.3 placed). For 5-job lists the result is "about as accurate, far fewer placed". That's stronger evidence that list concentration, not accuracy, drives placement.
- **Schools:** it holds between the two classifiers, but not against random lists (next section).

## Observed check: Chilean school admissions

Todd's study is the only one with real preferences, priorities and lotteries. It has a different setup from the GPT studies:
- **Lists:** each family's 5-school list is a subset of the schools it actually applied to.
- **Models:** two ridge logistic models predict the official offer. One is *rank-blind*: it uses the choice set and program demand. The other is *rank-informed*: it also uses where the family ranked each school.
- **Placement:** deferred acceptance with the published lotteries and priorities, which approximates the real mechanism.
- **Accuracy:** whether the *official* offer is on the list, rather than the full-information match.

At 5 schools, against random 5-school lists (percentage points, 2020 / 2021):

| | Rank-blind | Rank-informed |
|---|---|---|
| Accuracy, all applicants | +1.8 / +1.5 | +2.1 / +1.9 |
| Placed, all applicants | **+0.28 / +0.24** | **+0.17 / +0.13** |
| Placed, families with more than 5 choices | +4.8 / +4.3 | +4.2 / +3.8 |
| Placed, families with 5 or fewer choices (own list unchanged) | −0.13 / −0.07 | −0.19 / −0.15 |

Rank-informed minus rank-blind: accuracy +0.32 / +0.33, placed −0.12 / −0.11, about 520 fewer families placed each year.

How it fits with the GPT studies:
- **Against random lists, the classifiers place *more* families, and GPT places fewer.** "More accurate, fewer placed" appears in Chile only when rank information is added to an already-good model. GPT's lists lose placements to random lists in both NYC markets, so they do worse than a simple learned ranker on what matters for placement.
- **The congestion spillover appears in observed data too.** Families whose own list never changes still lose a little placement when everyone else's list is recommended. It's the same mechanism as finding 1: one person's list affects everyone else's chances.
- **Gains and losses concentrate where lists are cut.** Almost all the movement is among families with more than 5 choices, as finding 2 would predict. Most Chilean families list 5 or fewer schools, which is why the all-applicant effects are tiny.
- **Not directly comparable:** no LLM, no T and no audits. The groups are administrative (priority, grade) rather than income, and the accuracy measure differs.

## Repair, protection and audit costs (jobs only)

The rental runs report no audits. In the jobs markets:
- **Repair cost:** the cheapest stable repair costs about the same under GPT as under random (3.7–3.8 vs 3.9 firms).
- **Protection:** protecting lower-wage seekers, or all seekers, from ending worse than their first-round match costs essentially nothing extra.
- **Losers:** about 16–17% of seekers still hold a worse job after the repair.

These results use synthetic six-person seeker-side agents. The one-firm-per-seeker check below tests them under the housing convention: costs rise, but every conclusion holds.

### One firm per seeker (sensitivity)

The housing side gives every renter a private firm. This check gives every job seeker one too (`seeker_firm_size: 1`), replacing the synthetic six-person agents: so auditing a seeker means overriding the recommender for that one person. Setup:
- **Scope:** primary variant, both windows, no API calls (GPT answers reused).
- **Unchanged inputs:** seekers, postings, preferences and GPT's lists.
- **Redrawn:** the random and rule-based lists, because their random seed comes from the market ID.

Tables are in `jobs-gpt-frozen/firm1/`.

| Exact repair cost (firms) | Six-person agents | One firm per seeker |
|---|---|---|
| Firms per market | 12.1 | 32.1 |
| GPT 5 jobs vs random, frozen | 3.67 vs 3.86 (−0.19 [−0.24, −0.13]) | 5.66 vs 6.12 (**−0.46** [−0.55, −0.35]) |
| GPT 5 jobs vs random, replication | 3.80 vs 3.87 (−0.07 [−0.12, −0.02]) | 6.04 vs 6.34 (**−0.31** [−0.49, −0.11]) |
| Who gets audited, GPT 5 jobs (frozen) | 3.1 seeker agents + 0.6 employers | 0.8 seekers + 4.9 employers |
| Worse off after repair, GPT / random | 17% / 24% | 17% / 25% |
| Protection premium | ≈ 0 | ≈ 0 (at most 2 firms, in < 3% of markets) |

- **Repairs cost more:** they go from about 4 firms to about 6. They also switch from overriding seekers to auditing employers, because overriding one person now buys much less.
- **GPT stays cheaper to repair than random,** and its advantage grows (−0.46 and −0.31 firms). Every optimum was certified, and every plan passed the stability check.
- **No change:** who ends up worse off, and the near-zero cost of protection.
- **Not affected at all:** placement, accuracy, T and the capacity results don't involve audits, so firm grouping doesn't change them.

## What differs between the two GPT studies

| | Rentals | Jobs |
|---|---|---|
| Market size | 64 × 64 | 24 × 24 |
| Groups | income below/above the median | prior wage below/above the NYC median ($58,657) |
| The cap that binds | rent cap binds lower-income renters | pay floor binds higher-wage seekers |
| Preferences | calibrated | uncalibrated frozen primary; calibrated sensitivity gives the same placement and T pattern |
| T cell weights | borough ACS shares | occupation × month cells weighted equally |
| Audits | none (no-audit runs) | full set of exact and heuristic methods |
| Seeker-side firms | one per renter | six-person agents in the frozen design; one per seeker as a sensitivity check |
| Other arms | Claude Haiku 5.5, honest ranker | honest ranker, half-length lists, steering rules (not allowed: Claude, under the Revelio terms) |
| Published replies | yes (gzipped) | no (Revelio terms: aggregates only) |

## Open
- **Missed better options are not the same as placement:** "miss a better option" and "placed" can move in opposite directions. In rentals, GPT cuts higher-income renters' misses while also cutting their placement (finding 2). Report both, and don't read T as a welfare measure on its own.
- **What was pre-specified:**
  - **Protocol estimands:** T, the group switch and the protection premium were named in the shared protocol (`EXPERIMENT_SETUP.md`) before either GPT study ran.
  - **Accuracy and capacity/anchor:** added after results in both studies (`rental_accuracy.py`; the paper's own capacity follow-up). In jobs they were committed before the held-out replication, which they passed: placement −17.5, capacity share 1.28, anchor +15.9.
  - **Rest:** calibrated T, one firm per seeker and the rental capacity check (`rental-capacity-gpt/`) are post hoc.
  - **Record:** details are in `jobs_manifest.json` under `later_additions`.
