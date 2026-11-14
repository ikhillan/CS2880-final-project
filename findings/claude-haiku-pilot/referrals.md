# Who could use a vacancy after the Haiku pilot?

Starting with the stable allocation produced by each actual five-listing display in the simulator, we gave initially unmatched renters access to coordinated, otherwise hidden listings. The market, asking rents, priorities, and existing recipients’ menus stay fixed. The first row for each policy restricts a referral to the renter’s five favorite feasible listings according to their simulated preferences. The second allows any mutually acceptable vacancy. These are scenarios for willingness to consider a listing. There are no observed acceptance data.

| Development design | Initial menus | Allowed rank | Initially placed | Additional placements | Lower-income additions among largest extensions |
| --- | --- | --- | ---: | ---: | ---: |
| Ordinary | Haiku | Top 5 | 7,388 | 204 | 6–8 |
| Ordinary | Haiku | Any | 7,388 | 683 | 22–25 |
| Ordinary | Uniform | Top 5 | 7,858 | 35 | 1 |
| Ordinary | Uniform | Any | 7,858 | 215 | 6 |
| Enriched | Haiku | Top 5 | 9,370 | 370 | 133–190 |
| Enriched | Haiku | Any | 9,370 | 1,926 | 1,002–1,085 |
| Enriched | Uniform | Top 5 | 10,268 | 49 | 33–35 |
| Enriched | Uniform | Any | 10,268 | 518 | 341–363 |

Each row pools 15,360 renter instances in 240 estimable markets. The ordinary sample includes 6,617 lower-income instances and the enriched sample has 7,680. Counts are raw so a reader can add the stages. Across ordinary markets, unrestricted referrals take Haiku from 7,388 to 8,071 placements and uniform from 7,858 to 8,073. Under the top-five restriction, the final counts are 7,592 and 7,893. In enriched markets the initial 898-placement advantage for uniform becomes a 510-placement advantage for Haiku under unrestricted referrals, while uniform stays ahead by 577 under the top-five restriction. Comparisons are conditional on these fixed policy menus and full referral take-up within each specified rank limit. A smaller first-round allocation also leaves more people and vacancies available to the referral stage.

**Why lower-income access differs.** The ordinary Haiku markets have 7,972 vacant listing instances after the first round. Just 37 initially unmatched lower-income renter instances can reach even one of those vacancies under the feasibility rules. In the enriched design that number is 1,601, with 5,990 vacancies. The range in the table is the exact minimum and maximum lower-income count among assignments achieving the largest overall extension in every market. It quantifies the room available to prioritize income while keeping the same placement count.

**Calculation.** On each residual graph we first maximize cardinality. Within that cardinality we solve twice, favoring each income group in turn. A row can use one listing and each listing can house one renter. The cardinality bonus exceeds the largest possible group count. For both endpoints of every range we rerun applicant-proposing deferred acceptance after replacing the selected renter’s least preferred displayed option with the assigned vacancy. Each accepted referral reached its named listing, all incumbents received a weakly preferred placement, and every row kept its quota. A separate bipartite matching algorithm verified the cardinalities. The rank and income information used for this retrospective calculation comes from the simulator. Implementation by a service would require current tentative offers and a way for renters to report which options they would accept.

To reproduce from the repository root, with the June–August 2026 development data installed, run

```sh
python -m experiments.pilot_referrals
```

The command uses the committed Haiku selections and writes [the aggregate counts](referrals.csv). The [paired pilot report](paired-outcomes.md) gives the primary contrast, the supported geography and the model’s provenance. This follow-up was chosen after seeing the pilot outcomes and was not part of its prespecified hypothesis.
