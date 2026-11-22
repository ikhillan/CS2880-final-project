# Rental coordination: more placements, some displaced renters

The one-option anchor raises mean placement, but it does not retain every person who was placed before. A broader coordination rule places substantially more renters than the anchor while covering fewer full-information assignments. This is a comparison between two coordinated policies with the same menu sizes; the broader rule can replace more options.

These are **post hoc counterfactual replays**, with modeled renter preferences and landlord priorities. The original jobs inputs and saved recommendations were unavailable, so we tested the same anchor on the recoverable rental experiments. The results below do not reconstruct missing jobs outcomes.

## What was reproduced

We rebuilt all 1,280 available markets from the three original configurations, with 64 renters and 64 listings per market. All 55,115 saved GPT candidate hashes matched the reconstructed candidates. Before export, 54 checks reproduced the original placement, anchored placement and menu-change means. The cluster replay separately reproduced all 36 placement and anchored-placement means in [reproduction.csv](reproduction.csv). There were no new model calls.

Of 1,600 planned markets, 320 original Staten Island supply failures remain omitted and documented; we did not replace them. Both sampling designs are reported. The **primary** design samples renters by survey weight and listings uniformly. The **enriched** design deliberately samples affordable listings and lower-income renters with feasible options. Neither panel should be interpreted as the NYC population.

| Original window | Months | Primary markets | Enriched markets | Seed blocks per design |
|---|---|---:|---:|---:|
| Development | Jun–Aug 2026 | 240 | 240 | 20 |
| Held out | Sep–Oct 2026 | 160 | 160 | 20 |
| Replication | Jun–Aug 2025 | 240 | 240 | 20 |

These window names describe the original studies. Their results were already known when this follow-up was designed; they are not fresh confirmatory samples.

## Policies and information

**Anchor + GPT** inserts at most one option per person from a maximum feasible matching, reusing existing menu edges where possible and dropping GPT's last pick when necessary. **Balanced + GPT** minimizes squared listing exposure loads, then maximizes saved GPT recommendation ranks, using exact min-cost flow. Unlisted options receive tied scores because their relative ranks were not elicited.

Both use the full feasibility graph, listing capacities and cached menus, without realized preference shocks, landlord rankings, full-information assignments or baseline recipients. Each person's list retains `min(5, feasible degree)` options. Balanced + GPT can replace several options, so its advantage is not an equal-edit-budget comparison. It is an optimization control, not a claim to a new congestion-balancing algorithm.

The score-based controls know the simulator's deterministic preference formula. They are model-informed benchmarks, not evidence that GPT outperforms alternatives with the same information.

## Results in all six cells

Differences are percentage points. Coverage is the share of full-information recipients whose reference listing is displayed, averaged equally across markets within each seed and then across seeds. Gross exits count modeled renter appearances placed under GPT but unplaced under Anchor + GPT; repeated simulated appearances are not distinct observed households.

| Window / design | Anchor − GPT: placed | Anchor − GPT: coverage | Anchor exits / renters | Balanced − Anchor: placed | Balanced − Anchor: coverage |
|---|---:|---:|---:|---:|---:|
| Development / primary | +0.69 | −0.06 | 25 / 15,360 | +7.01 | −6.06 |
| Held out / primary | +0.79 | −0.01 | 7 / 10,240 | +6.61 | −5.13 |
| Replication / primary | +0.96 | −0.13 | 25 / 15,360 | +7.62 | −5.85 |
| Development / enriched | +3.51 | −0.16 | 87 / 15,360 | +12.83 | −3.70 |
| Held out / enriched | +4.02 | −0.33 | 37 / 10,240 | +12.65 | −3.05 |
| Replication / enriched | +4.92 | −0.25 | 78 / 15,360 | +12.77 | −4.17 |

Sources: [primary.csv](primary.csv), [transitions.csv](transitions.csv). In **every cell**, Balanced + GPT's placement gain and coverage decline relative to Anchor + GPT exclude zero under the simultaneous intervals described below. For held-out enriched markets, these differences are **+12.65 pp [11.45, 13.84] placed** and **−3.05 pp [−4.24, −1.85] coverage**.

The anchor's smaller coverage changes need different wording. Its held-out enriched change is −0.33 pp, with pointwise interval [−0.53, −0.15] but simultaneous interval [−1.52, +0.86]. All six simultaneous intervals for Anchor − GPT coverage include zero. This establishes neither coverage equivalence nor a family-adjusted coverage loss.

The baseline matters. In the held-out enriched cell:

| Policy | Coverage (%) | Placed (%) |
|---|---:|---:|
| Random | 16.00 | 66.87 |
| GPT | 24.13 | 63.17 |
| Anchor + GPT | 23.80 | 67.20 |
| Balanced + GPT | 20.75 | 79.84 |

Both coordinated policies have higher **mean** coverage and placement than random in each enriched window, although the held-out anchor placement difference is uncertain: +0.33 pp, simultaneous interval [−0.86, +1.53]. In primary markets, Anchor + GPT remains below random in mean placement. Balanced + GPT exceeds random on both mean outcomes in all six cells. Thus the evidence does not imply that a placement–coverage tradeoff is unavoidable for every improvement or comparator.

## Individual access and spillovers

In the held-out enriched cell, Anchor + GPT produces **449 entrants and 37 exits**, for a net gain of 412 placements. Its gross-exit rate is 0.36% of all renters, pointwise interval [0.29%, 0.42%], or 0.57% of GPT recipients. Relative to random, the same anchored policy produces 862 entrants and 828 exits. Similar aggregate headcounts can conceal substantial changes in who receives a home.

Balanced + GPT adds 1,466 entrants and causes 171 exits relative to Anchor + GPT in that cell. Neither policy provides an individual retention guarantee. The tables also report switches, rank worsening among retained renters and normalized ordinal rank costs. These modeled rank measures are not measured household welfare.

Renters with one to five feasible listings see their entire feasible set under both GPT and the anchor: their own menus cannot change. Across the six cells, **13 gross exits occur in this unchanged-menu subgroup** (3 development enriched, 1 held-out enriched, 7 replication enriched, and 1 each in development/replication primary). These are at least 13 exits caused by changes elsewhere in the simulated market. They are not a field estimate of harm from deploying advice, or a count of every unchanged-menu exit outside this degree subgroup.

## Uncertainty and reproducibility

Each window/design has 20 original seed blocks. In general, counts are pooled within a seed and seed rates are averaged equally. Primary coverage first computes each market's conditional coverage, retaining the earlier rental estimand; pooled-recipient and unconditional coverage remain separate endpoints. Undefined denominators stay visible rather than becoming zero outcomes.

We use 10,000 shared, paired seed-block bootstrap draws. [primary.csv](primary.csv) gives pointwise intervals and approximate 95% simultaneous intervals based on the maximum absolute centered bootstrap deviation, for four comparisons × two endpoints **within each window/design**. The six families are separate; this is not a simultaneous guarantee over all windows or secondary subgroup analyses. The follow-up remains post hoc despite these adjustments.

Twenty index permutations are integrated within each market for anchor sensitivity, with per-person random deletion fixed in the random-menu arm. They are not 20 additional independent observations. Conditional transition rates for randomized policies divide expected counts rather than averaging conditional rates. Reported uncertainty describes the reconstructed model's seed variation, not population or field-effect uncertainty.

The release contains **316,680 seed-level aggregate rows**, all output tables, input/source hashes, and [rebuild_validation.json](rebuild_validation.json). Every released estimate and confidence interval was reproduced from these aggregates to absolute tolerance `1e-10`, without loading raw data, a replay packet or per-market logs:

```sh
python -m experiments_jobs.anchor_access_release rebuild --release findings/rental-anchor-access
```

See [PROTOCOL.md](PROTOCOL.md), [release_manifest.json](release_manifest.json), [levels.csv](levels.csv), [contrasts.csv](contrasts.csv), and [seed_counts.csv.gz](seed_counts.csv.gz). The social-impact question is now concrete: how much placement and recommendation quality can a policy achieve while protecting current recipients, including people whose own menus never change? These replays establish the accounting and comparison; behavioral response and field validity remain open.

## Compute record

The replay ran in a separate `cs2880-anchor-access-20261122` directory on kkl. Its single eight-CPU Slurm job completed in 5 minutes 18 seconds, using about 2.54 GB peak memory. No GPU allocation or new LLM call was required. A one-CPU validation job and a one-CPU aggregate-release job also completed. Other projects' files and jobs were not changed. See [Slurm accounting](slurm-accounting.txt) and the three logs in [compute/](compute/).

Raw replay outputs were copied from scratch into this project's home-directory results folder after completion. The repository release contains aggregate counts and provenance, not private input records or the replay packet. The [single-allocation launcher](../../scripts/slurm/rental_anchor_access.sbatch) and [batch runner](../../experiments_jobs/anchor_access_batch.py) reproduce the execution structure.
