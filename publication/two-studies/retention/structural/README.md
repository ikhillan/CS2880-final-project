# A rule that retains seats under every lottery order

This rule only changes what can be offered before clearing. It uses ordinary and reserved capacity, common courses, possible competitors, family links, submitted ranks, and features of statutory priorities that are known before the lottery. The rank-based construction never opens an April register. The generic file reader also loads historical offer and priority fields. None enters a comparison used to select this menu. A retrospective variant uses historical April enrollment only to select among the same admissible menus. The proof is in the seat-retention appendix.

For each year the all-order rule maximizes the sum of decreasing submitted-rank scores within its restricted set of permitted substitutions. A protected course that appears on both starting menus always remains. Among linked applicants the new menu produces 28 additional historical enrollment hits across the three years (−2, +5, +25). The hindsight variant gains 118 (29, 32, 57), which is the exact optimum for the restricted set of substitutions. It gives a lower bound for protection under every lottery order. These are two different decision rules.

## Reproduce

The historical portion of the institutional driver and the input documentation in the parent retention directory describe how to prepare the frozen markets, sidecars, tiers and registers. Set the following arguments to those local directories. The two numeric directories can be the same. The scratch folder holds the compiled code and private individual-level masks.

~~~sh
python3 reproduce.py --publication "$PUB" --numeric-early "$EARLY" --numeric-late "$LATE" \
  --sides "$SIDES" --features "$FEATURES" --scratch "$WORK" --output "$OUT"
~~~

The executable builds all masks first. The wrapper compares all six against the frozen fingerprints before re-clearing any reserved lottery. Both national rules are evaluated under the released order plus 32 new matched-course draws per year (batch 90267319). Each result file records exits, entrants, changes of course, changes of school and retrospective coverage. Several seats at a school can be in different courses. A school switch counts a pupil with a seat under both policies at different schools. Summing switches across draws counts simulated pupil–draws, which can include repeated pupils. The universal guarantee follows from the proof for the specified institutional mechanism. The reserved simulations check the implementation.

The independent synchronous typed-seat solver from the institutional audit checked the compiled rule in 1,000 generated two-grade markets with six orders each. This check includes statutory priorities, family links, continuity fallback and competition across categories. It found 596 school switches and zero exits across 40,374 baseline-recipient draws. Reproduce with `python3 test_small.py --publication "$PUB"` and compare with `small-check.json`.
