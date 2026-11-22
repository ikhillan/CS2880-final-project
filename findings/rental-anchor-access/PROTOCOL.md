# Public rental replay: coverage and individual access after coordination

22 November 2026. This follow-up uses the public rental GPT caches because the original jobs menus and private jobs inputs were unavailable. It tests the **same anchor algorithm**, but it is not a new jobs experiment or a reconstruction of missing jobs transitions.

The earlier placement results were known when this analysis was designed. All three windows are therefore reused evaluation panels, not newly untouched test sets. The policy definitions and endpoints follow the [jobs follow-up protocol](../jobs-anchor-access/PROTOCOL.md), with the differences below.

## Exact reconstruction

Run all three original configurations, including primary and enriched sampling: `stage1_pilot`, `heldout_primary`, `heldout_replication`. Of 1,600 planned markets, 1,280 can be built (480, 320, 480 respectively). Record all 320 original Staten Island supply failures rather than replacing or resampling them. Every market has 64 renters and 64 unit-capacity listings. There are 20 original seed blocks per window/design, with different seed IDs across designs.

Use the committed `gpt_recs.jsonl.gz` files. Verify all 55,115 cached candidate hashes against the reconstructed listing order and feasibility; require valid complete responses for renters with more than five choices. Renters with at most five feasible listings see their full set and require no call. Check the original random, GPT, and transparent-score placement, anchored placement, and menu-change means against the released tables: 54 checks over three windows, two designs and three policies. No new model calls.

The exporter records all input hashes, planned/omitted markets, software versions, and reproduction checks. It produces modeled preferences, feasibility, anonymous menu indices and score ranks; it does not export household identifiers, listings' real IDs, income or rents.

## Paired policies and information

Nine fixed policies: Random, GPT, Score, Anchor + Random, Anchor + GPT, Anchor + Score, Balanced + GPT, Balanced + Score, and Score anchor + GPT. The original anchor and its original deletion RNG are reproduced exactly. Repeat the anchor for GPT and random under 20 index permutations, fixing each person's random deletion order within this sensitivity. Every policy uses the same market, priorities and personal menu budgets.

Balanced menus minimize total squared listing exposure load and then maximize recommendation rank score, using exact min-cost flow. Cached GPT ranks supply scores only on recommended options; other options are tied. Model-informed score controls know the simulator's deterministic utility coefficients. Those controls are explicitly distinguished from equal-information GPT-menu controls. Policies do not see realized preference shocks, landlord rankings, full-information assignments or baseline recipients.

## Outcomes and inference

The four primary comparisons are Anchor + GPT minus GPT, Random and Anchor + Random, and Balanced + GPT minus Anchor + GPT. Primary endpoints are placement and **equal-market mean conditional reference coverage**. This coverage estimand matches `experiments/rental_accuracy.py`; the released jobs analysis instead pools recipients within seed blocks. The new pipeline also reports pooled-recipient coverage and unconditional coverage under separate metric names.

Compute reference coverage in each market among renters with a full-information match; average markets within each seed and then the 20 seeds. Placement includes every renter. Use shared, paired 10,000-draw seed-block bootstrap resamples, pointwise intervals, and a simultaneous centered maximum-deviation interval for the four comparisons × two endpoints within each window/design. The original rental accuracy figure used a market bootstrap; its interval is not substituted for these new paired intervals.

Report gross exits and entrants, both per all renters and conditional on baseline placement status; report worse modeled ranks, switches and normalized ordinal rank cost. Groups are lower/higher income and feasible degree 0, 1–5, 6–10, 11+. Undefined denominators remain visible. For randomized anchors, integrate counts over the fixed 20 tie draws within market before computing rates; these draws are not additional independent observations. Conditional transition rates are ratios of expected counts over tie draws.

Require exact headcount accounting and unchanged personal menu lengths. A positive placement difference does not establish individual retention. A confidence interval containing zero does not establish coverage equivalence. Report the strongest coordinated baseline even if it defeats the proposed anchor.

## Scope

These are counterfactual allocations with fixed simulated preferences and landlord priorities, grounded in public listing and survey inputs. They do not measure how households respond to advice, actual rental placements, utility, or effects on the NYC population. The enriched design deliberately increases feasible options; report the primary design alongside it. Additional computation cannot remove these behavioral and construct limitations.

Use Slurm CPUs for matching, exact menu-flow optimization and paired inference. Preserve aggregate outputs, source hashes and logs outside temporary scratch storage. Keep synthetic software validation distinct from the empirical replay.
