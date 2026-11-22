# Compare the two prospective rules on the same lotteries

This follow-up applies the original, frozen structural prospective menus to the learned rule's 128 matched-course and 32 school-family test draws per cohort. The seed batches and priority generation are exactly the same as in the learned-rule holdout. The three structural menus were already frozen before the learned-rule test. We use the original compiled audit function to re-clear both whole markets. The structural mask hashes and every year, lottery architecture, batch, draw index and baseline number of seats are checked against the existing records before aggregation. No new menu tuning takes place. The structural rule's protection across lottery orders is a theorem conditional on the specified mechanism and inputs. These simulations check the promised property and measure actual simulated school changes.

Use the generic C++ executable built from the learned rule's `retention.cpp` and the same paths JSON as `run_policy.py`. The three structural masks are the files `2022-prospective.mask`, `2023-prospective.mask`, and `2024-prospective.mask`. Supply the public CSV receipt containing year, mode and SHA-256 in the structural rule release. Direct the output somewhere separate from the learned rule's holdout tables; both use `holdout_course.csv` and `holdout_family.csv`.

~~~sh
python3 run_comparison.py --binary "$SCRATCH/retention" --config /data/paths.json \
  --masks "$STRUCTURAL_PRIVATE" --freeze "$STRUCTURAL/frozen-masks.csv" \
  --learned "$LEARNED_RESULTS" --private "$COMPARISON_PRIVATE" --results "$COMPARISON_RESULTS" --workers 4
~~~

An exit counts a pupil who held a rank-blind seat and is unplaced under the structural policy. An entrant had no blind seat and receives one. A school switch counts a pupil who was placed under both rules and changed school identifier. The same pupil can contribute outcomes in multiple draws.

On the 384 matched-course lotteries the structural rule changed 9,643 retained pupil–school assignments (2,176, 2,250, and 5,217 by cohort), with zero entries and zero exits. On the 96 family-ticket lotteries it changed 2,417 (544, 563, and 1,310), again with zero entries and exits. The learned rule's changes on these identical panels were respectively 3,445 school switches, 104 entries, one exit; and 879 school switches, 28 entries, one exit. Totals sum repeated pupil–draw observations.
