# When families go beyond the shortlist

The same-grade complementary records identify who reapplied and which schools they named. A Chilean field experiment supplies evidence on how frequently a risk warning led families to add schools before the regular round closed. No applicant received the two hypothetical menus in this study. We use these sources to vary who searches, what they pursue, and how their choices affect other applicants. [Ministry records](https://datosabiertos.mineduc.cl/sistema-de-admision-escolar-sae/) · [Field experiment](https://doi.org/10.1093/qje/qjac013) · [Experiment replication](https://doi.org/10.7910/DVN/VCXL6A)

| Question | Evidence |
| --- | --- |
| What happened when omitted options were restored to the regular-round menus? | [Institutional re-clears, annual totals, means and Monte Carlo uncertainty](evidence/response-summary.csv) |
| How often did a later application identify an omitted course or a school absent from the entire original report? | [Annual linkage, 2022 example](evidence/relisted-2022.csv) · [Pupils who lost a simulated seat](evidence/loss-history.csv) |
| Could selection into search reverse the coverage result? | [Sharp responder-set bounds and allocations at their endpoints](evidence/selection-summary.csv) · [Unequal uptake across rules](evidence/differential-response.csv) · [Actual returners as responders](evidence/participation-summary.csv) |
| How far could newly named schools and published complementary vacancies close the gap? | [Two-stage capacity counts and bounds](evidence/second-chances.csv) |

The column called April hits counts linked pupils whose pursued set contains the school they historically attended. The original menu is included. School coverage can change when someone seeks an extra choice. Actual enrollment is held fixed across the policy comparisons. Linked placements count simulated seats among the same 1,969,120 records, after the entire regular cohort clears. The second-stage exercise has a different allocation objective. Each arm maximizes additional headcount over the recorded choices with earlier places protected. The linked variant gives linked candidates first claim on new capacity.

## Reproduction

From a checkout containing this paper, prepare the numeric market files, masks, institutional sidecars, April registers and course-level family lottery inputs following the [institutional instructions](../../institutional/REPRODUCTION.md). The regular and complementary files belong to the same annual SAE archive. Install the project's Python packages, including pandas, NumPy and SciPy; the programs use a C++17 compiler and bsdtar with RAR support. Run the following from this folder, setting paths for your checkout. The last two directories keep individual records and aggregates separate.

```sh
python3 reproduce.py --publication "$PUB" --archives "$SAE" \
  --early-numeric "$EARLY" --late-numeric "$LATE" \
  --institutional-private "$INSTITUTIONAL" --private "$WORK" --results "$RESULTS"
```

By default the driver prepares the complementary links, clears all response scenarios, then computes the coverage and second-stage bounds. Individual stages can be repeated with `--stage prepare`, `--stage replay` or `--stage bounds`. The bounds stage uses the prior stages' inputs and aggregate files. `--family` and `--family-template` accept previously prepared family files with different directory or naming conventions. The institutional solver recomputes the seat category for each allocation. These categories are unobserved in the ministry’s offer file. The public evidence contains aggregates only.

The [reapplication guide](reapplication/REAPPLICATION.md) defines the historical cohorts. The [simulation guide](engine/RESPONSE_README.md) documents the original and restored lists, priority pre-estimates and independent synchronous re-clears. The [field evidence](field/FIELD_EVIDENCE.md) records the source estimates and identities for the responder-set bounds. The [second-stage analysis plan](recourse/PLAN.md) gives the supply sensitivities. Run `python3 verify.py` from this folder to check the released aggregates, including the selection, field-rate and capacity comparisons, without raw records.
