# Baseline agreement within entire competition components

Individual agreement between the full-report DA replay and an official initial offer leaves open whether the model assigns an applicant's competitors correctly. In each application year we formed an undirected bipartite graph connecting every applicant to every course on their eligible original report and to the official initial offer if one exists. An outside outcome contributes no edge. Including all eligible edges keeps a whole component closed under either model shortlist and keeps the applicant's recorded offer within that component. Every applicant with a mismatched official and full-report outcome marks their entire component as having some disagreement. A course with no connected applicant contributes no component to the reported component count.

Across 2022–25, entire components with exact full-report agreement contained 13,363 out of 2,044,649 applicant-years, 11 of the 66,719 changed menus, three gained historical course hits and one lost simulated placement. Outcomes for changed menus differed only in 2022. All other gross placement losses and all placement gains occurred in components with a discrepancy somewhere. A discrepancy could be distant from a particular applicant. A large connected component gives many chances for at least one discrepancy. The exact components contain too few changes to provide an informative independent validation of the tradeoff. Neither full graph agreement nor individual agreement proves that a hypothetical policy would reproduce the ministry's allocation rules. The reported components are closed on modeled lists and recorded offers. Cross-grade family priorities or unreported alternatives could link them in the actual mechanism.

From this directory run the following, using the frozen numeric inputs for both pairs of cohorts. Only aggregates are written.

```sh
python check_components.py --early-numeric "$EARLY" --late-numeric "$LATE" --out school-component-agreement.csv
c++ -std=c++17 -O2 check_component_partition.cpp -o /tmp/school-component-check
for year in 2022 2023; do /tmp/school-component-check "$EARLY" "$year"; done
for year in 2024 2025; do /tmp/school-component-check "$LATE" "$year"; done
```

The Python script uses SciPy's undirected graph solver. The independent C++ checker uses union–find, compares complete applicant outcomes and reads both policy masks independently. Each C++ output line gives application year, applicants in wholly agreeing components, component count, number of edited menus, historical-offer hits gained and lost, and replay placement gains and losses. The two computations agree on every field in all years.
