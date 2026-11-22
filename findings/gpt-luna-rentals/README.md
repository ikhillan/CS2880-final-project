# GPT shortlist arm: gpt-6-luna, top-5, all three NYC rental windows

This is the NYC rental recommender rerun with **gpt-6-luna** in place of Claude Haiku 5.5 (see `../claude-haiku-pilot/`).
Everything else is identical: the same markets, seeds, months, feasibility rules, random baseline (`uniform_k5`), prompt
and analysis. These are the numbers on the presentation's NYC slides.

## Setup
- **Model:** `gpt-6-luna` through Harvard's HUIT API Platform ("OpenAI for Community Developers"). Reasoning off
  (`reasoning_effort: "none"`), strict JSON schema output, `max_completion_tokens` 300. Sent one request at a time
  (Harvard's batch queue did not start).
- **Prompt:** unchanged from `hiddenoptions/llm.py` (prompt hash `a2961eba3557f75e`, the same as the Haiku runs).
- **Requests:** 55,115 (development 20,740; held-out 13,580; replication 20,795). Renters with 5 or fewer options see all
  of them and need no call. **Every reply was valid:** 5 valid listing IDs, with no refusals or fallback fills.
- **Cost:** $7.22 at list price ($0.10 / $0.50 per million tokens), within the $10 monthly HUIT credit (`spend.txt`).
- **Code:** `experiments/llm_recommend_openai.py` (requests), `experiments/rental_accuracy.py` (accuracy check).

## Windows
| Folder | Config | Months | Markets with results |
|---|---|---|---|
| `stage1_pilot/` | `configs/stage1_pilot.json` | Jun–Aug 2026 (development) | 480 |
| `heldout_primary/` | `configs/heldout_primary.json` | Sep–Oct 2026 (held out) | 320 |
| `heldout_replication/` | `configs/heldout_replication.json` | 2025 (held out) | 480 |

## Main result: T (GPT − random), percentage points, enriched markets

| Window | GPT T (95% interval) | Haiku T (95% interval) |
|---|---|---|
| Development | **+8.5** (6.6 to 10.4) | +11.5 (10.0 to 12.9) |
| Held out | **+6.7** (4.9 to 8.5) | +8.0 (6.3 to 9.9) |
| Replication | **+8.1** (6.0 to 10.2) | +10.6 (8.6 to 12.9) |

T is the change in the lower- minus higher-income gap in the share of renters with a hidden blocking pair ("miss a better
home"). The gap widens in every window, and every interval excludes zero. The same direction appears in the ordinary
(primary) design, which is smaller because most lower-income renters there have no feasible listing (`*/contrast_T.csv`).

**Where it comes from (`*/group_switch.csv`):** almost all of T comes from switching higher-income renters' lists
(S_H = +8.4, +7.3, +6.7). Switching lower-income renters' lists barely matters (S_L = +0.1, −0.6, +1.4).

## Slide numbers (enriched, average of the three windows)

| | Random | GPT | Haiku |
|---|---|---|---|
| Higher-income renters who miss a better home | 85% | **81%** | 79% |
| Lower-income renters who miss a better home | 38% | **41%** | 42% |
| Renters placed (average of the two groups) | 67% | **64%** | 60% |
| Accuracy: full-information home is on the list | 16% | **24%** | 25% |

The first two rows are the `B_*` columns of `contrast_T.csv` (`variant=enriched`, `renters=all`), averaged over the three
windows. Placement is `match_before` from `groups.csv`, averaged over the two groups and then over the windows.

**Accuracy vs. placement (`accuracy.txt`):** GPT's lists are 8.0 / 8.1 / 7.9 points more accurate than random, yet place
2.0 / 3.7 / 3.3 points fewer renters. All intervals exclude zero. It's the same "more accurate, fewer placed" pattern as
Chile and Haiku. Accuracy uses the jobs definition: among renters matched under full information, the share whose
full-information listing is on their shortlist.

**Price:** GPT shows higher-income renters somewhat pricier listings (price percentile among feasible options about 0.57,
vs 0.50 for random). Haiku did not (about 0.48). Lower-income renters: about 0.50 for both.

## Caveats
- One prompt per renter, so the estimates include model sampling noise.
- Simulated preferences and landlord screening, as in the Haiku runs.
- The Haiku held-out comparison numbers come from the earlier Haiku runs on the same configs. Only the Haiku development
  run's replies are committed (`../claude-haiku-pilot/`).

## Files
In each window folder:
- `summary.md`, `contrast_T.csv`, `group_switch.csv`, `groups.csv`, `methods.csv`: the `experiments/analyze.py` outputs
  (`--treated gpt_k5 --control uniform_k5 --reps 10000`).
- `gpt_recs.jsonl.gz`: every saved GPT reply (market, renter, chosen listings, validity, model, prompt hash). With the
  repo's market builder, this reproduces the run without new API calls.

At the top level:
- `accuracy.txt`: `experiments/rental_accuracy.py` output for all three windows (it includes Haiku where its replies
  were available).
- `spend.txt`: tokens and cost per window.

## Reproduce (no API calls)
```sh
for w in stage1_pilot heldout_primary heldout_replication; do
  mkdir -p results/llm/$w && gunzip -c findings/gpt-luna-rentals/$w/gpt_recs.jsonl.gz >> results/llm/$w/recs.jsonl
  python -m experiments.run configs/$w.json --workers 7
  python -m experiments.analyze $w --treated gpt_k5 --control uniform_k5 --reps 10000
done
python -m experiments.rental_accuracy configs/stage1_pilot.json configs/heldout_primary.json configs/heldout_replication.json
```
Use `>>` so you don't overwrite Claude replies already in `recs.jsonl`. Arms with no saved replies are skipped, so a
missing `claude_k5` arm only drops that arm.
