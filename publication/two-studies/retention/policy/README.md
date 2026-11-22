# A prospective menu with seat retention

The rule retains seats for all of a training panel's rank-blind baseline recipients while allowing changes of course, category and school. The market clears after a family submits its complete preferences. Recommendations use a frozen April-school calibration trained only on applicants from the 2021 application round, whose April 2022 registers predate all three evaluation rounds. At application time they use the submitted rankings, institutional capacities and rules, the two historical 2019 offer scores, and simulated priority draws. A fresh course-level test uses 128 draws per year; a second test uses 32 fresh school-level family-ticket draws per year. The plan and model freeze precede scoring these policies on the later enrollment registers. Public products contain aggregate results and code.

## Build from the ministry files

Use the [institutional reproduction driver](../../institutional/REPRODUCTION.md) to generate the 2022–24 typed market, frozen market/masks, tier, family, and private register files. The [2021 reproduction](../../field-study/cohort/README.md) and its [2021 staging instructions](../../field-study/mixed/README.md) generate the earlier training market and April 2022 register. Preserve the private input files on your machine. The worked example below assumes this directory is located two levels below the paper, as in the repository. Use a complete repository checkout to supply the shared C++ engine.

Make a JSON configuration containing an object per application year. Every path must refer to its matching cohort. Here is its schema, using a separate private data root:

~~~json
{
  "2021": {"prefix": "/data/2021", "side": "/data/2021.side", "register": "/data/2021-register"},
  "2022": {"prefix": "/data/2022", "side": "/data/2022.side", "register": "/data/2022-register", "tiers": "/data/2022-tiers", "families": "/data/2022-families.bin"},
  "2023": {"prefix": "/data/2023", "side": "/data/2023.side", "register": "/data/2023-register", "tiers": "/data/2023-tiers", "families": "/data/2023-families.bin"},
  "2024": {"prefix": "/data/2024", "side": "/data/2024.side", "register": "/data/2024-register", "tiers": "/data/2024-tiers", "families": "/data/2024-families.bin"}
}
~~~

For a prefix such as `/data/2022` the main binary is `/data/2022.bin`, with `2022-blind.mask` and `2022-rank.mask` in that same directory. The register is sorted in B1 pupil order. The year-specific families file is the school-wide ticket file emitted by `institutional/prepare_family.py` without the course-specific flag. The tiers refer to the archived five-tier grouping that enters `institutional/institution_ceiling.cpp`. Its Interleave lottery matches the priorities used to compute the 314-hit ceiling.

~~~sh
python3 run_policy.py --publication "$PUBLICATION" --config /data/paths.json \
  --private /data/retention-private --results /data/retention-results --workers 5
python3 summarize_policy.py /data/retention-results
~~~

Run with `--phase design` to obtain and freeze each policy before accessing the holdout lotteries or later labels, then `--phase test` to audit those fixed menus. The flag shards use one bit represented as a byte per eligible course. The driver ORs their per-draw-group exclusions. Course recommendations are deterministic given the fitted 175-cell probability table and full design support. Shard workers may be varied without affecting any result. Inputs require Python 3, NumPy, SciPy and a C++17 compiler.

## Inputs seen by each step

The 2021 fit consumes only the earlier observed April enrollment, school ID dictionary and the A and B features. The course scores are stored as logits, so the bin cutoffs use log-odds. At allocation time the C++ input class overwrites the legacy raw binary's historical offer vector and all recorded lottery positions before running any policy or simulated allocation. Current-cohort April labels are opened only by the separate final coverage scorer. Families are grouped using the historical family file for the school-family lottery sensitivity. The secondary outcome of a higher or lower reported rank compares original course positions, including pupils affected by sibling priority.

Within a school, the feature bins use the best rank and the two maximum offer scores among the pupil's union choices. Given a required base and a fixed number of remaining slots, maximizing the sum over distinct schools means choosing the available schools with the largest scores not already counted. If there are more slots than distinct schools, any additional eligible course is equivalent for this objective. Menu rows whose objective is already maximal keep all old choices. The rule never drops any mandatory overlap.

## Verification and hindsight diagnostic

The proof in `prospective-proof.tex` truncates the new expanded list at the original contract for an ordinary course and at the *end of the course* for an always-available course. Retaining each old winner follows even if the within-course seat type changes. The final grade has no younger grade to which changed assignments could propagate. For all higher grades the exact old prefix keeps family states unchanged. `python3 check_truncation.py` independently tests the argument by generating 2,000 two-type markets and clearing all 5,201 permitted menu profiles. It uses synchronous proposals and explicit rankings of school-type contracts.

The register-dependent `hindsight.cpp` can be run after every prospective menu is frozen. Among the rows missing their April school under A, an extra hit is attainable exactly when an admissible proposed-only course has the target school and a removable old-only course is available. These sufficient constraints separate by row, so all additional hits for this class are simultaneously attainable while leaving each existing hit intact. This gives a lower bound on the general seat-retention optimum. It does not make April enrollment a design input to the prospective policy. With shell variables pointing to this directory and the original private files, use

~~~sh
c++ -O3 -std=c++17 -I"$PUBLICATION/research/observed/school" \
  -I"$PUBLICATION/two-studies/institutional" -I"$PUBLICATION/two-studies/audit/chile" \
  hindsight.cpp -o "$SCRATCH/hindsight"
"$SCRATCH/hindsight" "$PREFIX" "$SIDE" "$REGISTER" "$SCRATCH/$YEAR.flags" "$YEAR" "$SCRATCH/hindsight-$YEAR.csv"
~~~

Concatenate the three per-year outputs, retaining the header once, to obtain `class-oracle.csv`. These are separate results from the prospective scores and placement audits.

## Out-of-time prediction

The 2021 training sample comprised 82,467 pupil–candidate-school pairs, including 9,996 enrollments at a candidate school. Across the three evaluation cohorts, the learned rule changes 7,721 pupils' menus (8,828 course substitutions) and raises historical coverage among 1,524,149 register-linked pupils from 1,239,953 to 1,240,003 (+50; 134 gains, 84 losses). Net changes were +9, +22 and +19 in the three application years. The unrestricted rank-informed menus gain 3,590. Even a designer given the April labels could gain only 47, 52 and 64 hits above baseline within this deliberately conservative, row-separable subclass, or 163 in total. The learned rule retains 30.7% of the net gain possible in its own subclass; this fraction is descriptive and uses an oracle denominator. Protection under the broader seat-only optimization is examined separately.

## Generalization over priority draws

On 128 independently drawn national matched-course lotteries per cohort, one old recipient lost a seat in 2023 and none did in the other two years. The corresponding counts of retained recipients moving to another school were 1,390, 901, and 1,154. Over all 384 draws the rule preserved 103 more placements on net relative to A (104 newly placed, one gross exit) while the unrestricted rank-informed switch had 277,663 gross exits on the same draws. These totals count recurring pupil–lottery observations. They are not unique children. The yearwise one-sided exact 95% upper bounds for the chance of any loss nationally are 2.31%, 3.65%, and 2.31%.

Under fresh school-wide family lotteries, one gross exit occurred in 2022 and none in 2023/24 (32 draws per year). The yearwise bounds are respectively 13.98%, 8.94% and 8.94%. These calculations condition on fixed reports and the specified lottery laws. Program attendance contributed historical labels. Family response to deploying the rule was not observed. The new menus were not selected or adjusted after either holdout. To verify the aggregate release, run `python3 verify_results.py`.

The 2022 school-family failure (draw 12) had a positive net headcount: two pupils gained seats and one baseline winner lost theirs. Total placement rose from 520,849 to 520,850. This illustrates why the diagnostic is the number of prior winners who become unassigned rather than the aggregate placement difference. In the matched-course failure (2023 draw 42), the single loss was accompanied by zero entrants.
