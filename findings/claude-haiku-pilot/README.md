# Claude shortlist pilot: Haiku 5.5, top-5, stage-1 development markets

This is the first run of the Claude recommender arm on the stage-1 pilot markets (`configs/stage1_pilot.json`, June–August 2026,
20 seed blocks per design, 64 renters × 64 listings, 480 estimable markets + 120 unestimable Staten Island markets).
It answers the protocol's main question: compared with uniform five-listing shortlists, does an AI shortlist change the
lower- minus higher-income gap in the share of renters with a hidden blocking pair (T)?

The tables in this folder were produced with the `experiments/analyze.py` on `main` (commit e430948, paired markets, joint
seed-block bootstrap, 10,000 draws). They match the earlier analysis code to the reported precision.

## Setup
- **Model:** `claude-haiku-5-5`, effort `high`, structured JSON output, Message Batches API. Not the pinned Opus model:
  this is a cheap check of the pipeline and the direction of the effect.
- **Prompt:** unchanged from `hiddenoptions/llm.py` (prompt hash `a2961eba3557f75e`; budget, bedrooms needed, target area and
  the listing table; no explicit income or household-size field).
- **Requests:** 20,740 (renters with 5 or fewer options see all of them and need no call). All replies were valid:
  no refusals, invalid IDs or fallback fills. Cost about $4.20.
- **Not run yet:** the 40% (`claude_q40`) arm, repeat prompts, held-out months, Opus.

## Main result: T (Claude − uniform), percentage points

| Design | Renters | T | 95% interval |
|---|---|---|---|
| Ordinary (primary) | all | **+5.4** | 4.1 to 6.5 |
| Enriched | all | **+11.5** | 10.0 to 12.9 |
| Enriched | renters with options to hide | +14.2 | 12.0 to 16.3 |

The month-specific estimates have the same sign in June, July and August for both designs (`contrast_T.csv`).
For comparison, the transparent formula gives T = +0.9 (−0.3 to 2.1) ordinary and +18.6 enriched.

## Where the gap comes from: group switches (`group_switch.csv`)

| Design | S_L (switch lower-income lists only) | S_H (switch higher-income lists only) | T |
|---|---|---|---|
| Ordinary | +0.0 (−0.1 to 0.1) | **+5.3** (4.1 to 6.5) | +5.4 |
| Enriched | +1.3 (0.2 to 2.3) | **+10.2** (9.2 to 11.1) | +11.5 |

The lower-minus-higher blocker gap went from −73.25 to −67.89 percentage points in ordinary markets and from −48.16 to −36.70 in enriched markets. Both absolute gaps shrank. In enriched markets, the lower-income blocker rate rose from 37.03% to 41.31% and the higher-income rate fell from 85.19% to 78.01%. The larger higher-income-list switch component tells us whose lists changed the outcome. People in either group can be affected by that switch.

## Placement and price

The follow-up replay of all 480 development markets pairs the same renters across policies. Among higher-income renters in ordinary markets, Haiku shows the full-list partner 9.17 percentage points more often and places 4.58 points fewer people than uniform. In enriched markets the differences are +10.92 and −9.88 points. Lower-income renters in the enriched design experienced substantial turnover even though the weighted net change in their placement rate has an interval spanning zero. The three metrics and their denominators are defined in the [paired analysis](paired-outcomes.md).

Mean shortlist price percentiles in the descriptive table range from 0.445 to 0.499 for Haiku across the four design/group combinations, versus approximately 0.5 for uniform. These averages alone do not establish the presence or absence of price steering. In ordinary markets, 84% of lower-income renter instances in the raw market averages had no feasible listing. They remain in the primary denominator.

## Coordinated referrals after offers

We also asked how many initially unmatched renters could take a currently vacant, mutually acceptable listing if the platform could change one recommendation per selected renter. In the ordinary Haiku markets, the unrestricted maximum was 683 additional placements. At that maximum, between 22 and 25 could go to lower-income renters, depending on how equally large solutions are chosen. In enriched markets the maximum was 1,926, with between 1,002 and 1,085 going to lower-income renters. Restricting referrals to the renter's five favorite feasible listings lowered the two maxima to 204 and 370. These are raw counts across 240 simulated markets per design, under full take-up. The same analysis for uniform menus and the verification checks are in the [referral results](referrals.md).

## Caveats
- Development months only; held-out September–October 2026 and the 2025 replication have not been touched.
- Haiku rather than the protocol's pinned model. A 2-market Opus test earlier showed some price steering for higher-income
  renters that did not appear here, so the model choice may matter.
- Simulated preferences: the lower-income rent-burden target is not reachable with asking rents (see calibration notes).
- One prompt per renter (no repeat prompts yet), so the estimates include model sampling noise.

## Files
- `summary.md`, `contrast_T.csv`, `group_switch.csv`, `groups.csv`, `methods.csv`: analysis outputs.
- `claude_recs.jsonl.gz`: the parsed selections from every saved reply (market, renter, chosen listing indices, validity counts). With the
  repo's market builder this reproduces the run without new API calls. Decompress it to `results/llm/stage1_pilot/recs.jsonl`.
- `batches.json`: model, effort, prompt text, prompt hash and batch IDs.

## Reproduce
```sh
mkdir -p results/llm/stage1_pilot && gunzip -c findings/claude-haiku-pilot/claude_recs.jsonl.gz > results/llm/stage1_pilot/recs.jsonl
python -m experiments.run configs/stage1_pilot.json
python -m experiments.analyze stage1_pilot --treated claude_k5 --control uniform_k5 --reps 10000
```
