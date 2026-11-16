# Chilean school-admission cohorts

We trained two transparent ridge logistic models to predict official initial school/course offers on every family submitting a regular-round list in 2019 and evaluated the policies without refitting on the complete 2020 and 2021 cohorts. Rank-blind prediction conditions on the choice set and program demand. The rank-informed comparison adds the declared preference position. The strict unit of comparison is one applicant and one program in one nationwide application cycle.

The quota counts discretionary reported courses. A returning applicant's enrolled course is additionally protected and always shown. The effective order truncates at this entitlement. Many applicants have five or fewer discretionary options, including applicants who have none. These applicants remain in the denominator. Making a short list requires knowing the reported choice set. Chile observed submitted preferences, official lotteries and the initial historical offers, but did not deploy these menus. The pooled-capacity, fixed-priority DA assignment is computed. It approximates the administrative mechanism, which also implements reserve-specific routing, family applications and dynamic sibling priorities.

The 2019 fit used 483,070 applicants and 1,685,900 application records. The 2020 and 2021 tests contain 915,638 applicant-cycle records altogether. Relative to uniform at five, both official-offer recovery and pooled DA placement increased under the rank-blind score in each year. The explicit rank features further improved coverage while reducing the number placed, by 525 and 519. Across all applicants whose five-slot menus always stayed fixed, moving others to rank-blind scores reduced these applicants' placement rate. Program reserves and secured seats matter for reconstruction. The raw- and regular-capacity sensitivity arms produce the same directions; neither establishes a causal effect on actual assignments or enrollments.

## Inputs and re-execution

Use the three public regular-round application-year archives linked from the [Chilean Ministry of Education data catalog](https://datosabiertos.mineduc.cl/sistema-de-admision-escolar-sae/). The A1, B1, C1 and D1 charts, regular-round files and the Spanish manual are in the downloads; the data catalog calls these application years 2019 to 2021, for admission in 2020 to 2022 respectively. Source hashes and direct official addresses are in fit-and-inputs.json. The output bundle contains coefficients, aggregate grade/group/policy counts and source code only. No administrative source file, applicant ID, participant-specific preferences, physical coordinates or lottery order is distributed. The experimental derivative is descriptive academic research. Public accessibility of the source archives is not a claim about unrestricted redistribution or a particular database license.

Run from this directory on a CPU, allowing 1 GB working storage and several GB of RAM. Python needs pandas, NumPy, and SciPy. A C++17 compiler and bsdtar (for the ministry RAR archive) are also required.

~~~sh
mkdir -p raw private results
curl -fsSLo raw/SAE_2019.zip https://datosabiertos.mineduc.cl/wp-content/uploads/2021/12/SAE_2019.zip
curl -fsSLo raw/SAE-2020.zip https://datosabiertos.mineduc.cl/wp-content/uploads/2021/12/SAE-2020.zip
curl -fsSLo raw/SAE_2021.rar https://datosabiertos.mineduc.cl/wp-content/uploads/2022/02/SAE_2021.rar
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python -B prepare.py --data raw --scratch private --out results
c++ -O3 -std=c++17 replay.cpp -o private/replay
for year in 2020 2021; do
  private/replay private/$year.bin $year private/$year-counts.csv
done
python -B summarize.py --out results private/2020-counts.csv private/2021-counts.csv
c++ -O3 -std=c++17 check_partners.cpp -o private/check_partners
for year in 2020 2021; do
  private/check_partners private/$year.bin $year private
done
python -B verify.py --scratch private --audit-dir private --counts results/counts.csv.gz --out results/verification.json
python -B capacity.py --scratch private --data raw
for year in 2020 2021; do
  private/replay private/$year.bin $year private/$year-regular.csv private/$year-regular.bin protected
  private/replay private/$year.bin $year private/$year-raw.csv private/$year-raw.bin unprotected
done
python -B capacity_summary.py --scratch private --out results
~~~

The source binary format is little-endian, with 32-bit integers and 64-bit floating-point scores, tested on macOS arm64 using Clang. All seeds are fixed. Comparisons of rank ties are stable within the declared pseudo-random generator. Training program offer frequencies use held-out-year independent counts; the 2019 edge is omitted from its own 2019 program frequency estimate. Test IDs and lottery outcomes serve allocation or linkage only. Scores, program capacity and list length are evaluated after reported choice sets are available. No statements about unseen schools or application behavior before submission follow from the design.

At quota five we show the mean across 48 independent uniform menu profiles, with a Student Monte Carlo half-width conditional on these fixed applicants and the published lotteries. All deterministic policy counts are exact conditional on the benchmark, including tie policy; these are administrative cohorts, not 915,638 independent replications of policy. Raw comparisons are stratified by published grade and disadvantage designation. A positive rank score does not establish that preserving cover of official offers improves allocation. See school-main.tex, school-detail.tex, and PROTOCOL.md for model distinctions and timing of secondary analyses.

Detailed outputs include fit-and-inputs.json (fitted coefficients and sample construction), counts.csv.gz (all policies), summary.csv (mean rates and estimates), capacity-counts.csv.gz and capacity-summary.csv (capacity sensitivities), verification.json (independent batch-proposer cross-check and exact uniform edge-inclusion calculation). Group 'affected' counts applicants with more than the quota discretionary choices. The complementary 'unedited' group can change assignment only through others' menus. Group selections overlap: do not sum subgroup statistics without verifying a partition. The vulnerable classification is an administration-provided designation, not a measurement of equivalence to the NYC income group.
