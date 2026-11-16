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
  the listing table; no income or household size).
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

Almost all of T comes from changing **higher-income** renters' lists. Claude's picks lower higher-income renters' hidden-
option rate (enriched: 85% → 77%), while lower-income renters are about as exposed as under uniform lists. So the gap widens
mostly because Claude helps higher-income renters more, not because it hurts lower-income renters.

## Other outcomes (pooled, unweighted, from `groups.csv`)
- **No price steering:** shortlist price percentile among feasible options is 0.45–0.50 for Claude vs 0.50 for uniform.
  The transparent formula, by contrast, picks cheaper listings (0.34–0.39).
- **Matching:** Claude keeps placement close to uniform (higher-income 0.83–0.87 vs 0.88–0.96), unlike the transparent
  formula, whose identical lists collide (0.54).
- **Scarcity still dominates the ordinary design:** about 82–87% of lower-income renters have no feasible listing, so the
  ordinary-design T is driven by higher-income renters and the small group of lower-income renters with options.

## Caveats
- Development months only; held-out September–October 2026 and the 2025 replication have not been touched.
- Haiku rather than the protocol's pinned model. A 2-market Opus test earlier showed some price steering for higher-income
  renters that did not appear here, so the model choice may matter.
- Simulated preferences: the lower-income rent-burden target is not reachable with asking rents (see calibration notes).
- One prompt per renter (no repeat prompts yet), so the estimates include model sampling noise.

## Files
- `summary.md`, `contrast_T.csv`, `group_switch.csv`, `groups.csv`, `methods.csv`: analysis outputs.
- `claude_recs.jsonl.gz`: every saved Claude reply (market, renter, chosen listing indices, validity counts). With the
  repo's market builder this reproduces the run without new API calls; copy it to `results/llm/stage1_pilot/recs.jsonl`.
- `batches.json`: model, effort, prompt text, prompt hash and batch IDs.

## Reproduce
```sh
mkdir -p results/llm/stage1_pilot && gunzip -c findings/claude-haiku-pilot/claude_recs.jsonl.gz > results/llm/stage1_pilot/recs.jsonl
python -m experiments.run configs/stage1_pilot.json
python -m experiments.analyze stage1_pilot --treated claude_k5 --control uniform_k5 --reps 10000
```
