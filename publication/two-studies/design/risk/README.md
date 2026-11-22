# Designing one menu against several priority draws

The C++ programs import the published applicant-proposing DA engine and within-course, within-tier shuffle. Inputs are the frozen private numeric market, its two menu masks, public-priority tier vector and optional linked register used in the [portfolio reconstruction](../portfolio/README.md). Only aggregate results are distributed. A school and an applicant never appear by identifier in these files.

Read the [initial protocol](PLAN.md), [matched-budget follow-up](FOLLOWUP.md), and [held-out validation protocol](LABELS-AND-VALIDATION.md) in that order. Experiments reuse three cohorts already studied in the paper. The 256 new evaluation draws are independent from both the original 64 test draws and all design draws within the specified simulator. A new priority draw does not provide a new school district or population.

From this directory, given the paths under the portfolio reconstruction, run

```sh
c++ -O3 -std=c++17 -I"$PUBLICATION/research/observed/school" -I../../audit/chile lottery_design.cpp -o "$PRIVATE/lottery"
c++ -O3 -std=c++17 -I"$PUBLICATION/research/observed/school" -I../../audit/chile draw_sensitivity.cpp -o "$PRIVATE/sensitivity"
c++ -O3 -std=c++17 -I"$PUBLICATION/research/observed/school" -I../../audit/chile label_validation.cpp -o "$PRIVATE/validation"
c++ -O3 -std=c++17 -I"$PUBLICATION/research/observed/school" -I../../audit/chile label_ceiling.cpp -o "$PRIVATE/ceiling"
for year in 2022 2023 2024; do
    numeric="$EARLY_NUMERIC"
    if [ "$year" = 2024 ]; then numeric="$LATE_NUMERIC"; fi
    "$PRIVATE/lottery" "$numeric" "$year" "$PRIVATE/$year-tiers" "$PRIVATE/$year.csv"
    "$PRIVATE/sensitivity" "$numeric" "$year" "$PRIVATE/$year-tiers" "$PRIVATE/nested-$year.csv"
    "$PRIVATE/validation" "$numeric" "$year" "$PRIVATE/$year-tiers" "$PRIVATE/$year-register" "$PRIVATE/labels-$year.csv" "$PRIVATE/validation-$year.csv"
    "$PRIVATE/ceiling" "$numeric" "$year" "$PRIVATE/$year-tiers" "$PRIVATE/$year-register" "$PRIVATE/ceiling-$year.csv"
done
python3 verify_results.py
python3 check_menus.py --markets 24000 --seed 801601
python3 verify_lottery_boundary.py
python3 verify_uniform_lottery.py
```

The reported comparison protects common choices as well as continuity entitlements. The exact per-row overlap equals the original number of differing slots minus the larger of the number of mandatory old-only and forbidden new-only choices. Old baseline allocations may change with the lottery. Each output lists the total slots adopted, departures and entrants, worse assignments in the reported order, and separate counts of missing old assignments and pupils with at least one blocking pair. A single pupil may contribute to both latter counts. The excluded mask for the unrestricted rank switch is all ones as a sentinel. Every policy and baseline uses the same priority draw in each comparison. Seed batch 513 trains, batch 514 supplies the first holdout, and batch 515 the independent follow-up. The menu construction never reads historical offers or enrollments. See [scope and prior work](PRIOR-WORK.md) for the complexity statement.

Slurm CPU job 60423893 under the project's kkl account independently enumerated 24,000 synthetic markets, 72,000 optima for the three risk budgets, and 560,514 menu/scenario combinations. Its program compares exact overlap values against enumeration of every eligible menu and checks DA welfare implications. A second verifier exhaustively replays the graph gadget. The former cases use seeded market generation and exhaustive within-market evaluation.

For each known historical label, the oracle starts from the old menu. A previously absent target can be shown precisely if an eligible proposed-only course matches it and an old-only choice can be dropped. Contributions separate by applicant. The graph is restricted to the two policies’ union with all common choices fixed. Different targets have different hindsight maximizers. The three cohorts gain at most 278 April-school hits or 48 initial-offer hits after protecting 1,024 draws.
