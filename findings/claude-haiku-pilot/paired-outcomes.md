# Who kept a placement in the Haiku pilot?

We replayed the archived Haiku and uniform five-listing menus for the same renters, feasible options, preferences and listing priorities. This adds two questions to the pilot’s hidden-blocker analysis. Did a renter see their full-list deferred acceptance (DA) partner? Did they get any placement when everyone used the new menu?

Among higher-income renters, Haiku put the full-list partner on the menu more often in both designs. Fewer renters were placed. The same group also had fewer hidden blocking pairs, so treating a lower hidden-blocker rate as evidence that more people obtained housing would give the wrong conclusion.

## Paired comparisons

Each entry is the Haiku rate minus the uniform rate, in percentage points, followed by a 95% paired seed-block bootstrap interval. All three outcomes use **every renter in the indicated income group** as denominator. A full-list partner is an assignment in the simulator when every feasible pair is displayed. A renter unmatched there contributes zero to the partner-shown rate under both policies.

| Market design | Income group | Full-list partner shown | Receives a placement | Has a hidden blocking pair |
| --- | --- | ---: | ---: | ---: |
| Ordinary | Lower | +0.11 [0.00, 0.24] | +0.002 [−0.19, 0.21] | +0.10 [−0.02, 0.23] |
| Ordinary | Higher | +9.17 [7.99, 10.34] | **−4.58 [−5.27, −3.86]** | −5.26 [−6.39, −4.05] |
| Enriched | Lower | +0.78 [0.19, 1.43] | −0.92 [−2.23, 0.37] | **+4.28 [2.77, 5.79]** |
| Enriched | Higher | +10.92 [9.47, 12.29] | **−9.88 [−10.43, −9.36]** | −7.18 [−8.20, −6.14] |

A blocking pair means the renter and a hidden listing would each prefer the other to the partner obtained under the specified display. Someone can have such a pair even while placed. Renters with no feasible listings cannot have a blocking pair and still count in the denominator.

The pilot’s primary contrast is the change in the **signed** lower-minus-higher blocker gap. It went from −73.25 to −67.89 points in ordinary markets and from −48.16 to −36.70 in enriched markets. Both signed contrasts are positive and both absolute gaps **shrank**. In enriched markets the lower-income rate rose from 37.03% to 41.31% while the higher-income rate fell from 85.19% to 78.01%. The group whose menus produced most of the movement in the switch experiment and the people whose outcomes changed are distinct quantities.

## How many people changed placement status?

These are raw counts over repeated simulated renter instances. “Lost” means placed under uniform and unplaced under Haiku. “Gained” is the reverse. The raw net count and the weighted rate difference above use different weights.

| Design | Group | Renter instances | Placed under uniform | Placed under Haiku | Lost | Gained |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Ordinary | Lower | 6,617 | 297 | 244 | 127 | 74 |
| Ordinary | Higher | 8,743 | 7,561 | 7,144 | 818 | 401 |
| Enriched | Lower | 7,680 | 2,866 | 2,707 | 1,203 | 1,044 |
| Enriched | Higher | 7,680 | 7,402 | 6,663 | 907 | 168 |

In enriched markets 2,247 lower-income renter instances changed placement status across the two policies. The net loss was 159. This is substantial turnover even though the interval for their weighted net change includes zero.

## What was checked

- The replay used the 20,740 saved, valid Haiku replies from PR 4, checked each required (market, renter) exactly once and validated all five distinct indices against the regenerated feasible set. Renters with at most five options received every feasible listing. The legacy cache stores local indices and lacks candidate-identity hashes or verbatim API replies. It supports a deterministic results replay. Independent authentication of the original prompt payloads would require more provenance.
- The original experiment runner and analyzer reproduced all 10 blocker-contrast rows, six group-switch rows, 80 descriptive group rows and 16 audit-method rows to an absolute and relative tolerance of 1e−12, excluding runtime fields. The placement follow-up separately recomputes each DA allocation and its blocking pairs. It also checks its group-level blocker rates and primary contrast intervals against the published rows.
- Each design has 240 paired markets clustered into 20 seed blocks. A block contains four supported boroughs in each of three development months. Staten Island contributed 120 unestimable market draws across the two designs. We retain the original census recent-mover borough weights, equal month weights and normalization over the 12 supported borough-month cells (97.95% of planned weight). Each interval resamples entire seed blocks 10,000 times with the same draw for both arms and both groups. Renters within a block can be dependent.
- This is a secondary analysis of the June–August 2026 development pilot. Intervals are pointwise simulator intervals conditional on the observed catalogs, the model’s saved replies and the benchmark draw. They do not account for between-prompt variability. The market generator supplies renter preferences and listing priorities. A simulated placement is not an observed lease.

## Reproduce

With the stage-1 development catalogs and renter profiles prepared, run from the repository root

```sh
python -m experiments.pilot_outcomes --out results/haiku-followup
```

The command reads the tracked compressed reply cache. It writes a summary and a table of paired numerator and denominator counts by design, seed, borough, month, income group and endpoint. The checked-in copies are [paired-outcomes.csv](paired-outcomes.csv) and [paired-outcomes-counts.csv](paired-outcomes-counts.csv). For the original runner and group-switch analysis, use the commands in [the pilot README](README.md).

## How many blocks does the protocol call for?

The [sample-size rule in §6](../../course/rental-experiment-protocol.md) specifies **400 held-out blocks if a pilot group-cell is empty**. Staten Island has no estimable ordinary market in any development month, so the five-borough plan triggers that fallback. More blocks improve precision on supported cells while leaving an infeasible cell unrepresented.

For comparison, we repeated the calculation on the four supported boroughs. Dropping one *whole seed block* at a time gave single-month standard errors for the ordinary all-renter signed contrast of 0.8201 pp in June, 0.8682 pp in July and 1.0613 pp in August. Using proportions, the largest standard error times the square root of 20 is 0.04746. The one-point target gives 87 blocks before applying the protocol’s lower limit and 100 afterward. This four-borough calculation changes the target population. The pilot used Haiku, so its variance is also a planning proxy for a future model. Neither calculation certifies the achieved interval width. The updated stage-2 configuration applies 400 blocks to the primary comparison and retains the original 100 blocks for the separate 2025 replication.

```sh
python -m experiments.pilot_size
```

[Machine-readable jackknife and sample-size calculation](paired-outcomes-planning.json)
