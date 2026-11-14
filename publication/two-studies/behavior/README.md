# What changes when applicants search beyond a shortlist?

We asked two questions. Do the frozen menus predict where families actually enrolled? How sensitive are placements to the order in which families use their recommendations? Only the first outcome can be scored directly against administrative records. All alternative allocations come from clearing the submitted or modeled lists under specified rules. These are follow-up analyses. The original offer and placement results were already known.

## Results

- **Where pupils enrolled.** In the 1,969,120 Chilean applicant-years linked to the next April register, the rank-informed menu contained the eventual school for 4,638 more pupils than the rank-blind one. On the same linked population, the informed rule placed 1,609 fewer pupils in the binding-menu replay. Both signs held in each cohort. Accounting for every unlinked pupil allows a net historical coverage gain between 878 and 8,395 across all 2,044,649 records. Only 20.12% of registered pupils who finished the regular wait list without an offer had their April school anywhere on the eligible original report. Of those who went without a regular offer, 30,699 later applied to a school absent from their entire original report, received a complementary offer at that school and were enrolled there in April. [Enrollment and search linkage](field/README.md)
- **How they might use advice.** In Chile, adding the highest omitted course at its reported rank produced 846 fewer first choices under informed menus. Appending the same course last produced 2,163 more. Appending every omitted course equated the available sets and left 1,331 fewer pupils placed. In NYC the enriched placement difference moved from −5.40 percentage points with binding menus to +0.28 points when all feasible apartments were available after the five suggestions. The latter paired interval was [−0.07, +0.65]. [School scenarios](chile/README.md) · [Rental scenarios](nyc/README.md)
- **Why equal sets can still displace someone.** On the same graph, trace each lost assignment through the new holders of the vacated seats. With fixed responsive priorities, separate losses must lead to separate applicants who reversed their order over the two programs they received. The gain paths have the same property in the other direction. We traced all 1,871 loss paths and 540 gain paths in the four school cohorts and all 1,231 gross changes in the 480 housing markets. [School certificates](chile/school-order-witnesses.csv) · [Rental certificates](nyc/nyc-full-tail-witnesses.csv) · [Finite verification of the bound](theory/verify_order.cpp)

## Verification

From this directory, run the following. The field checker uses the Python standard library. The simulation checks require NumPy and pandas. Both shortlists were frozen before scoring the enrollment outcome. Entire application cohorts clear together, including pupils absent from the later enrollment register. Individual records and assignment vectors are excluded from the release.

```sh
python field/verify_counts.py
python verify_behavior.py
python nyc/verify_advice.py nyc
c++ -O2 -std=c++17 theory/verify_order.cpp -o /tmp/verify-shortlist-orders
/tmp/verify-shortlist-orders
```

The C++ checker enumerates acceptable graphs, applicant orders consistent with each graph, and program priorities up to applicant relabeling, over three specified capacity vectors. It compares every ordered pair of stable assignments, including alternatives to the applicant-optimal assignment. The archived [result](theory/order-check.json) gives the number of comparisons and tight cases. The analytical result and the example attaining both bounds appear in the paper. The site-specific guides above give the raw-input and full-replay commands.
