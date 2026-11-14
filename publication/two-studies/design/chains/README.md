# Vacancy chains after a fixed-quota shortlist

The chain calculation starts from a stable assignment with strict, responsive program priorities. The paper-ready statement, proof, counts and interpretations are in `chains.tex`. Every file in this directory is aggregate data or source code. Inputs containing school applicants and their preferences are held separately.

## Reproduce

Use Python with NumPy, SciPy and pandas plus the project's rental-market dependencies. `REPO` is the main research repository containing `publication/research/policy/equity/transitions.py`, the NYC stage-1 catalogs, ACS household inputs and frozen pilot menus. `EARLY` contains the isolated data files `2022.bin`, `2023.bin` and the corresponding `blind` and `rank` masks and partner vectors. `LATE` contains the same files for 2024 and 2025. From this directory run

```sh
python chain_flow.py chile --repo "$REPO" --early "$EARLY" --late "$LATE" --out chile.csv
python chain_flow.py chile --repo "$REPO" --early "$EARLY" --late "$LATE" --top 5 --out chile-top5.csv
python chain_flow.py housing --repo "$REPO" --out housing.csv
python chain_flow.py housing --repo "$REPO" --top 5 --out housing-top5.csv
python verify_small.py
python summarize.py
```

The small check uses only NumPy and SciPy. The summary uses only the Python standard library. `summarize.py` reads all four archived aggregates and checks the table, denominators, comparison with unchanged uniform/unchanged rank-blind, and the market counts at which certificates meet ceilings. To write its output use `python summarize.py > summary.json`.

## What is optimized

The direct column is the maximum count when every old contract stays in one witness. Chains optimize the count subject to the baseline priority threshold at each program, applicant welfare and baseline program occupancy. The last column is the maximum on the applicant-welfare graph after relaxing program priorities. This maximum bounds any DA mechanism with those preferences, outside option, capacities and acceptable candidate pool. A maximum ordinary flow in that graph is attained from the original matching by simple augmentations, so requiring incumbent retention and nondecreasing occupancy in the construction does not lower the ceiling. For the top-five ceiling, existing displayed options remain usable at all ranks. Every hidden option added must be in that applicant's top five. A school continuity option, if present, occupies its place in the reported-order ranking used by this check and stays on the menu.

`placed` gives original recipients. `direct`, `chain` and `upper` give additional people, not the final placed count. `edited` counts menu substitutions. `paths_with_incumbents` counts selected paths of length at least two people. `longest_path` counts people including the entrant. `movers_witness` and `movers_da` count incumbents strictly improving in the constructed witness and in the DA rerun. `entrants_at_witness_option` counts new recipients who obtain the exact witness contract in DA. For other entrants, DA assigns a more preferred option. Path lengths and witness assignments need not be unique. `tight=1` certifies global optimality in recipient count for that market and baseline under the specified option set.

For each flow we compare its value with an explicit residual min cut, check every applicant and program, remove pure cycles, and test stability of the witness even when every old menu edge remains visible. The final menu keeps each protected course, each row keeps the same length, the DA assignment is stable and every applicant weakly prefers it to the witness. Each national baseline is independently rerun and compared against its archived partner vector. Housing counts repeat 64 simulated renters per market over four boroughs, three months, and 20 market draws per cell and design. Counts pool those 240 markets per design. Comparing headcounts does not identify the same beneficiaries across policies. The four separate Chilean cohorts use pooled priorities with a fixed realized tie order. Neither setting records takes of these hypothetical offers.

## Why the bounds differ

1. **Direct 0, chain 1.** Ana ranks y ahead of x and initially sees x. Ben accepts only x. Both units have capacity one, and x prefers Ana to Ben. A single menu substitution for Ana creates a placement for Ben.
2. **Chain 0, one-swap optimum 1.** Three unit programs x, y, z and three applicants A, C, B. A ranks z,x,y. C ranks x,y,z. B accepts only x. A and C initially see x,y and B sees x. Program x ranks A,C,B. The initial outcome gives A x and C y. Give A and C the menu y,z. DA now assigns A z, C y and B x. The original claim of C on x rules out B in the threshold construction.
3. **One-swap optimum 0, ceiling 1.** Keep the preceding market but let C accept only x,y, so both must appear on a two-slot display. The priority reversal at x makes any three-person DA placement weakly improving every applicant impossible. The priority-free allocation A z, C y, B x remains feasible.

These examples are independently enumerated in `verify_small.py`. It also enumerates all 1,296 two-applicant/two-program cases with strict complete rankings, each program capacity in {0,1,2}, and each initial row a nonempty subset of the programs. It enumerates all capacity-feasible matchings and all menus of the original row cardinalities with at most one swap. A separate locked-row test gives an otherwise eligible entrant no discretionary slot. The implementable bound excludes its hidden edge while the relaxed ceiling retains it. The archived result is `small-check.json`.

Related literature includes [Trabelsi et al. on allocation facilitation](https://arxiv.org/abs/2508.18325), [Gajulapalli et al. on stability-preserving second rounds](https://arxiv.org/abs/1904.04431), [Feigenbaum et al. on school seat reassignment after cancellations](https://pubsonline.informs.org/doi/10.1287/mnsc.2019.3469), and [Lepe on guaranteed unlisted school alternatives](https://consiliumbots.github.io/working-papers-cb/L_2025/working_paper/L_2025.pdf). The new calculation is a sufficient fixed-priority certificate on an edited display, accompanied by a numerical upper bound. Augmenting paths, DA optimality and the matched-set invariance theorem are classical.
