# Who keeps an offer?

[Print poster (PDF)](who-keeps-an-offer.pdf) · [Editable LaTeX source](who-keeps-an-offer.tex) · [Research paper](../publication/ACM_EC_RECOVERY_AND_PLACEMENT.pdf)

This one-page landscape poster is 48 × 36 inches. Print at actual size. To rebuild, run the following from this directory with a full TeX Live installation. The vector plot reads the 20 measured rank thresholds from the paper's replication data.

~~~sh
latexmk -pdf -halt-on-error who-keeps-an-offer.tex
~~~

## Figure data

- The 2022 and 2023 school cohorts and historical responses are from Chile's Ministry of Education. The upper table compares the two frozen 2019 predictors, with and without submitted rank. Its offer column measures coverage of recorded initial offers. Its placement column measures the result of a fixed-report deferred acceptance replay. [Directional protocol](../publication/research/policy/equity/HOLDOUT.md) · [Paired gross transitions](../publication/research/policy/equity/transitions.csv.gz) · [Explicit acceptance results](../publication/research/policy/equity/response-coverage.csv) · [Primary data](https://datosabiertos.mineduc.cl/sistema-de-admision-escolar-sae/)
- The university vacancy example is the 2026 rank-informed network, five displayed programs, using the published regular-seat and overbooking counts. The bound fixes the original contracts in a feasible stable assignment. In the new DA outcome incumbents can keep or improve their original ranks. The rank limit represents which referrals the platform may make. Individual take-up under changed menus was not observed. [Reachable seats](../publication/research/policy/design/reachable.csv) · [Rank limits used in the graph](../publication/research/policy/design/referral-ranks.csv) · [Proof](../publication/research/policy/design/theorem.tex) · [Primary data](https://datosabiertos.mineduc.cl/pruebas-de-admision-a-la-educacion-superior/)

This poster presents the team's Harvard CS 2880 project. OpenAI Codex assisted with the design, text and data graphic. The manuscript and replication notes document the assistance with the research itself.
