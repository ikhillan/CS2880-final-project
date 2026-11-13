# Administrative confirmation and distributional spillovers in Chilean school admissions

The primary predictor and the comparison are those of the first observed-preferences school study. Fitted coefficients and prior program frequencies use only 2019 applications. Application-year 2020 and 2021 results were known. HOLDOUT.md froze the direction of both primary contrasts before acquiring the following two cohorts. These are local protocol records, not a registry submission. PROTOCOL.md records the downstream distributional and priority-seat analyses. RESPONSE-PROTOCOL.md covers the post-hoc observed acceptance check.

## What is observed?

The publisher is Chile's Ministerio de Educación. A1 reports capacity, B1 reports the applicant-level priority designation, C1 supplies the preference order/continuity/lottery, and D1 gives the official initial school-course and the family's reply. The paper's primary target is an initial administrative offer rather than an enrollment. Both full and pruned DA use reported preferences with priority rules/entitlements, including protected continuity; counterfactual reassignments are modeled. The designation is neither a direct income measure nor a randomized exposure. A nonprofit recommender designed for fair housing cannot infer its impact on income or race from these school cohorts. [Official release index](https://datosabiertos.mineduc.cl/sistema-de-admision-escolar-sae/) · [Ministry description of vulnerable-priority rule](https://www.ayudamineduc.cl/ficha/criterios-de-prioridad)

For a five-choice list plus entitlement, the primary 2022 and 2023 criterion is met. Compared with rank-blind, rank-informed covers 1,143 and 1,350 more actual official initial offers, but places 468 and 517 fewer applicants in the two replays. The national denominators are 570,891 and 536,353, with some people potentially present in both years. Within the 2022/23 cohorts 227 fewer full-list DA recipients retain placement and 758 fewer full-list DA nonrecipients enter under rank-informed vs blind. The preserved historical pair and aggregate retained-placement outcomes measure different things. During single-group adoption, designated pupils lose placements even when their personal menus do not change. The four-year gross transition, shift, entitlement and common-composition estimates are fully reported.

The reserve sensitivity chooses returning pupils first. It ranks all other applicants by the ordinary strict priority/lottery, fills up to the published count of earmarked priority places using designated applicants, and then considers every remaining applicant for the open pool. Unused priority seats release; total seats do not change. This is a substitutable, quota-filling choice rule at fixed capacity. Its choice from a tentative set is implemented directly. Other reserve types, dynamic sibling priorities and family block admissions are outside the replay. Full-list agreement with observed outcomes is reported for each year and group. The official codebooks for 2022/23 retain the same inputs for the frozen model. The 2022 D1 codebook additionally allows response code 7 (one such record). Neither training nor menu selection uses the response.

## Reproduce

Requires Python with NumPy and pandas, SciPy for the earlier preparation, bsdtar with RAR support, curl, and a C++17 compiler. Use the earlier school study's README to prepare oldraw (official original archives 2019/20/21) and oldnumeric (2020/21 market binaries). Set SCHOOL to its directory containing prepare.py, replay.cpp and fit-and-inputs.json. Keep all downloaded records, applicant vectors, and resulting binary graphs private. The D1 outcomes and lotteries are used for evaluation and priority ordering only.

~~~sh
mkdir -p newraw newnumeric aggregates private-checks
curl --fail --location --output newraw/SAE_2022.rar https://datosabiertos.mineduc.cl/wp-content/uploads/2023/03/SAE_2022.rar
curl --fail --location --output newraw/SAE-2023.rar https://datosabiertos.mineduc.cl/wp-content/uploads/2024/04/SAE-2023.rar
OPENBLAS_NUM_THREADS=1 python prepare_later.py "$SCHOOL" oldraw newraw newnumeric aggregates/cohorts.json
c++ -std=c++17 -O3 -I"$SCHOOL" export_menus.cpp -o private-checks/export_menus
c++ -std=c++17 -O3 -I"$SCHOOL" check_reserve.cpp -o private-checks/check_reserve
for y in 2020 2021 2022 2023; do
  case "$y" in 2020|2021) numeric=oldnumeric;; *) numeric=newnumeric;; esac
  private-checks/export_menus "$numeric/$y.bin" "$y" "$numeric"
  private-checks/check_reserve "$numeric/$y.bin" "$y" "newnumeric/$y-reserves.bin" private-checks
done
python transitions.py oldnumeric newnumeric aggregates --serial-checks private-checks
python summarize.py aggregates
python responses.py "$SCHOOL" newraw newnumeric aggregates/response-coverage.csv
~~~

cohorts.json records the source URL, input SHA-256, denominators, edge counts, continuity adjustments and the archived 2019 model fit hash. Each YEAR-cells.csv.gz is a disjoint grade × designation × list-length band × entitlement × policy/seat-rule table; program/course-level or per-student transitions are never shared. Codes for the list-length band are 0 (0–1), 1 (2–5), 2 (6–10) and 3 (>10). transitions.csv.gz aggregates these strata in explicitly named partitions. A pupil can appear in both 'all' and 'short'; sum disjoint rows only. 'gain'/'loss' refer to a move from the blind menu policy; 'new_nonrecipient' and 'kept_full' always use the full-list DA under the same allocation rule as their reference. 'exact_official' includes jointly unassigned applicants; 'exact_placed_official' counts only assigned official courses retained. Response code 1 records explicit acceptance; 2 records explicit acceptance while awaiting an upgrade; 3/4 reject; 5 has no reply; 6 indicates no initial assignment; 7 is exit.

standardized.csv uses within-cell group rates, weighted by full-cohort frequencies renormalized on grade/list-length/entitlement strata containing both groups. edit-decomposition.csv names the recipient and editor group. Order averages split the net joint change exactly into two editor-group contributions. They are descriptive attribution under this fixed model. Fine-stratum counts represent no actual interventions on people, and no individual IDs or coordinates are redistributed. verification.json records individual-vector and blocking checks; each batch and sequential allocation agrees exactly.
