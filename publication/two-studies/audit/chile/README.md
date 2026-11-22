# Rank outcomes and capacity in Chilean school admissions

These are exploratory counterfactual comparisons of the frozen 2019 rank-informed and rank-blind five-choice policies on the 2022–25 application years. Every cohort member enters the denominator. An individual can appear in several application years.

**Rank outcomes.** `school_ranks.py` joins the archived assignments to A1 course IDs, B1 applicant IDs, and original C1 application ranks. It checks the full public application rows against the numerical DA input. Entries below the guaranteed program are removed. Every retained raw C1 rank equals its position in this order. Every administratively appended option lies last. The file `school-rank-thresholds.csv` reports the number receiving an option at rank at most *k*. The `submitted_` columns exclude automatically appended courses using C1's `AGREGADA_POR_CONTINUIDAD` flag. A current-school course voluntarily ranked by a family counts as submitted. No automatically appended course is a first choice received. The last threshold of the all-course series equals total placements. Everyone unplaced has a rank below all acceptable courses in the modeled ordering.

`school-rank-changes.csv` compares each pupil with themselves, counting ordinal improvements and deteriorations among people placed both ways, with gains and losses of placement reported separately. Counting a move to or from an appended course as an ordinal comparison assumes that the family prefers any higher entered option to the protected fallback, which it prefers to being unplaced. The columns ending in `both_reported` require that both assignments came from family-entered choices. The groups that compare full-list DA with the historical offer partition a market after replay. They leave competitors’ menus and assignments in place.

Across four cohorts 2,218 more pupils received a family-entered first choice and 1,787 fewer received any offer. The counts of families receiving a top-three and a top-five submitted choice changed by +162 and −922. Among pupils placed under both policies, 9,590 moved higher and 6,888 lower in the replay order. Including moves into and out of placement gives 10,330 improvements and 9,415 deteriorations. Among the 1,977,930 records with unchanged lists, 2,254 improved and 7,215 worsened. Among the 66,719 records with edited lists, 8,076 improved and 2,200 worsened. Preference intensities and interpersonal welfare weights are unobserved.

**Capacity specification.** `school_capacity_audit.py` and its C++ solver compare the baseline maximum of the A1 `VACANTES` field and the number of current-school claims at the course with the sum of those two quantities. Both formulas use pre-offer inputs. Summing them makes an exploratory larger-capacity market and does not automatically reproduce official SAE. The `sum_frozen_menus` rows hold all archived displays fixed and recompute full-list DA and both allocations. Historical coverage stays fixed by construction. The `sum_rescored_menus` rows change every capacity-dependent scoring feature (log capacity, squared congestion, rank-by-congestion), holding the old 2019 coefficients, scaling factors, historical program priors, demands, priorities and tie orders fixed. The script regenerates both five-slot displays.

In the larger-capacity scenarios full-list exact agreement with D1 fell to 82.01–85.68%, compared with 95.38–96.59% at baseline. Each of the eight year-by-menu comparisons under expanded capacities retained higher full-list coverage and lower placement. Rescoring also retained higher historical coverage under every year. These results describe the two specified capacity formulas. They do not certify the sign for intermediate or other capacities.

**Reproduction.** Requires the original 2022 and 2023 RAR releases and numeric files prepared using the earlier pipeline, the 2024 and 2025 RAR releases and numeric files prepared using the later pipeline, NumPy, pandas, a C++17 compiler, and a RAR-enabled bsdtar. The environment variables below point to those inputs, the parent publication directory, and a writable output directory. Only aggregated files are shared.

```sh
python school_ranks.py --early-archives "$EARLY_ARCHIVES" --late-archives "$LATE_ARCHIVES" \
  --early-numeric "$EARLY_NUMERIC" --late-numeric "$LATE_NUMERIC" --out "$OUTPUT"
python verify_school_ranks.py --publication "$PUBLICATION" \
  --early-numeric "$EARLY_NUMERIC" --late-numeric "$LATE_NUMERIC" --out "$OUTPUT"
python school_capacity_audit.py --publication "$PUBLICATION" \
  --early-archives "$EARLY_ARCHIVES" --late-archives "$LATE_ARCHIVES" \
  --early-numeric "$EARLY_NUMERIC" --late-numeric "$LATE_NUMERIC" --out "$OUTPUT/school-capacity-bracket.csv"
```

The rank verifier reruns batched DA, compares each of 6,133,947 person–policy outcomes to the archived serial solver, and reconstructs all threshold counts from the retained numeric row order. The capacity driver checks original policies against the archived DA outputs and verifies feasibility, entitlements and stability for every new outcome. It independently recomputes full-reference agreement, coverage changes and placement levels for the expanded-capacity fixed-menu comparison with batched DA.

**Pre-lottery simulation benchmark.** `run_simulated_menu.py` builds a comparator with access to the entire cohort’s submitted lists, capacities, and coarse priorities. For each of 64 fresh lotteries, it draws an independent uniform permutation within each course and priority category and runs full-list DA. It keeps up to five discretionary courses per pupil, ordered by frequency of receiving them in simulated full-list assignments. Equal frequencies are resolved by the pupil's original reported order. The current-school entitlement stays available outside those five slots. The actual within-course order and D1 offers are held aside during selection. The frozen menus are then evaluated on that actual order. This is a prospective rule for a clearinghouse that already has every family’s application. It does not directly serve a family building its initial application. Program-specific independent lottery draws are a model of the priority uncertainty. They condition on the published priority flags as fixed and omit family dependence. The target is the pooled full-list offer under that model.

For reproducibility, the first RNG batch and the 16-, 32- and 64-draw checkpoints appear in `school-simulated-menu.csv`. `school-simulated-menu-batches.csv` reports all eight independent batches under the identical realized priority order and checks simulation variability, not external sampling error. Only the 64-draw policy is the specified main comparison. Counts of ties at the discretionary five/six boundary and selected zero-frequency choices describe sampling sparsity. Diagnosis fields for the fitted policies are inapplicable (stored as zero). To rerun:

```sh
python run_simulated_menu.py --publication "$PUBLICATION" --early-archives "$EARLY_ARCHIVES" \
  --early-numeric "$EARLY_NUMERIC" --out "$OUTPUT/school-simulated-menu.csv"
```

Relative to the fitted rank-informed policy, the 64-draw simulated menus cover 494–541 additional recorded offers in 2022 and 494–535 in 2023, and place 25–53 and 31–56 more pupils, across eight random seeds. First-choice placements decline by 106–140 and 124–163 in those same comparisons. All simulated menus still place fewer pupils than the fitted rank-blind menus. Joint improvements in coverage and placement are therefore feasible under the richer information set. They also show how comparisons change when ranking outcomes enter the policy objective.

`python verify_aggregates.py "$OUTPUT"` checks the published tabulations without individual-level data.

A direct check rebuilds the 2019 program priors and every feature from raw A1–D1 tables and applies the original matrix multiplication to each policy, rather than updating the archived logits by their algebraic differences. The original logits must reproduce byte-for-byte. Expanded-capacity logits agreed within 8 × 10⁻¹⁵, and all coverage, placement, joint gain/loss, first-choice and replay-accuracy counts agreed exactly. Floating-point ties can affect a few *counts of changed menus*. The direct raw-feature table `school-capacity-direct.csv` supplies the comparison for those menu totals.

```sh
python check_score_reconstruction.py --publication "$PUBLICATION" --training-archives "$TRAINING_ARCHIVES" \
  --early-archives "$EARLY_ARCHIVES" --late-archives "$LATE_ARCHIVES" \
  --early-numeric "$EARLY_NUMERIC" --late-numeric "$LATE_NUMERIC" \
  --published "$OUTPUT/school-capacity-bracket.csv" --direct "$OUTPUT/school-capacity-direct.csv" \
  --out "$OUTPUT/school-capacity-score-check.json"
```

**Later-cohort comparator check.** The dated local plan in `2024-COMPARATOR-PLAN.md` records the unchanged rule before the 2024 comparator was run. The 2024 archive and original fitted-policy outcomes were already known. Run the additional year separately from the original two. The driver authenticates the public archive and constructs coarse priorities directly from C1 flags without loading the lottery field. The C++ policy and seed generation remain identical to the original two years. The 2025 application cohort used a different tie procedure.

```sh
python run_simulated_menu.py --publication "$PUBLICATION" --years 2024 \
  --late-archives "$LATE_ARCHIVES" --late-numeric "$LATE_NUMERIC" \
  --out "$OUTPUT/school-simulated-menu-2024.csv"
python check_2024_simulated_menu.py --publication "$PUBLICATION" --directory "$OUTPUT"
```

For the second command, keep the policy source and dated protocol alongside the generated aggregate CSVs in `$OUTPUT`. The check verifies both file digests, the frozen comparison arms, and the full grid of 24 contrasts at the realized 2024 order.
