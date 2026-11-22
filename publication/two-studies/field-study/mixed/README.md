# Who loses a seat when other pupils' menus change?

This fixed-report simulation applies the frozen rank-blind (A) and rank-aware (B) five-course policies to Chile's 2021 national application market. Each clear uses the reconstructed institutional priorities. April enrollment defines linkage and historical coverage elsewhere in the paper. The outcome here is any simulated placement. A menu can only retain courses already present on the final report. The field experiment randomized what users saw inside the School Explorer. The 64 replicates below simulate individual assignment to the two five-course policies.

Of 200,591 register-linked entry pupils, 6,133 have different menus. Complete adoption reduces the placement count from 183,634 to 183,325. Among 194,458 pupils with identical menus it falls from 177,799 to 177,562. Thus this group accounts for 237 of the 309 **net** lost placements (76.7%). This is a displacement calculation conditional on fixed reports and fixed institutional inputs. Both endpoints are deterministic.

## Simulated individual randomization

At adoption probability `p`, every applicant, in every grade, independently receives B with probability `p`. Applicants with identical rows have own-treatment effect exactly zero at every profile. We skip their redundant coin flips, keep them in every clear, and retain them in subgroup denominators. The primary population is the 200,591 register-linked pre-K, kindergarten and grade 1 applicants. For a reporting population `G`, switchers `C`, own treatment `Z_i` and indicator of placement `Y_i`, the released own-effect sum is the average across paired draws of

```
D_hat = sum over i in G ∩ C of [(Z_i-p)/(p(1-p))] [Y_i(Z) - Y_i(all A)].
```

The centering term is fixed, so its expected weighted contribution is zero. Conditioning on other applicants' draws shows that each remaining term estimates the change in pupil i's own placement when just their menu changes. This gives the own-effect sum over all of G, including the zero effects. Percentage points equal 100 times the sum divided by the full subgroup size. The proportions at 45%, 50% and 55% have the **adoption rate** in the whole national market as their argument. They do not describe a change in total treated headcount restricted to the evaluation group.

Each replicate generates one independent uniform integer from 0 to 999 per switcher, with rejection sampling to avoid modulo bias. A second draw uses 999 minus the key. At each grid probability both assignments have the specified Bernoulli law. At one half they are complements. The same keys are reused across all saturations. Monte Carlo standard errors use the 64 independent pair means (sample SD divided by eight) and preserve the pairing in any difference. They measure simulation precision in this one market. The slope of total placements between 45% and 55% is a finite difference. It does not identify the exact spillover derivative at the midpoint.

| Entry-grade linked outcome | Estimate, placement count | Monte Carlo SE |
|---|---:|---:|
| Own-treatment effect summed over eligible pupils, 50% adoption | −64.42 | 0.46 |
| Total change from all A, 50% adoption | −158.70 | 0.45 |
| Change among identical-menu pupils, 50% adoption | −122.97 | 0.44 |
| Slope of total placements over 45–55%, per unit adoption | −306.09 | 8.45 |

## Replication

Python 3 and a C++17 compiler suffice for the mixed simulations. Recalculate estimates from the released aggregate draws, then check the estimator against an exactly enumerated three-pupil DA market at all five saturations.

```sh
python3 check_estimator.py
python3 summarize_mixed.py
```

To reproduce all 640 national mixed-policy clears, set `PUB` to the repository's `publication` directory, `BUNDLE` to this directory, `PREFIX` to the 2021 market basename (the file is `$PREFIX.bin`), `SIDE` to its institutional sidecar, `REGISTER` to the binary 2021–April 2022 linkage sidecar, and `OUT` to a new output directory. The wrapper checks five private inputs by SHA-256, builds the same institutional engine and checks all three output CSVs byte for byte against the release. Change the number of workers without changing results.

```sh
python3 "$BUNDLE/reproduce_mixed.py" --publication "$PUB" --prefix "$PREFIX" --side "$SIDE" --register "$REGISTER" --results "$OUT" --workers 8
```

If starting with the ministry archives, set `COHORT` to the directory containing the accompanying `frozen_2021.py`, `ARCHIVES` to the folder holding SAE_2019.zip and SAE_2021.rar, `APRIL` to the 2022 enrollment register archive, and `STAGE` to an empty private staging directory. Install the same Python and archive dependencies as the national cohort replication. These commands construct the five inputs locally.

```sh
python3 "$COHORT/frozen_2021.py" --school "$PUB/research/observed/school" --archives "$ARCHIVES" --output "$STAGE/2021.bin"
c++ -O3 -std=c++17 "$PUB/research/observed/school/check_partners.cpp" -o "$STAGE/menus"
"$STAGE/menus" "$STAGE/2021.bin" 2021 "$STAGE"
python3 "$PUB/two-studies/institutional/prepare_institution.py" --archive "$ARCHIVES/SAE_2021.rar" --numeric "$STAGE/2021.bin" --out "$STAGE/2021.side"
python3 "$PUB/two-studies/institutional/prepare_register.py" --publication "$PUB" --archive "$ARCHIVES/SAE_2021.rar" --year 2021 --register "$APRIL" --out "$STAGE/2021-register"
PREFIX="$STAGE/2021"
SIDE="$STAGE/2021.side"
REGISTER="$STAGE/2021-register"
```

The optional `--cohort "$COHORT/entry-2021.csv"` checks the four totals at full adoption against the separate historical-cohort replay. The bundled CSVs contain only group counts, randomization diagnostics and estimates. Repeated seed 47319 is combined with the year and replicate number via C++ seed_seq and mt19937_64.
