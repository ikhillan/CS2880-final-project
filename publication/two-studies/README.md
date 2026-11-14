# Who Gets a Place When Applicants Follow a Shortlist?

This paper compares five-choice recommendations in Chilean school admissions and NYC rental simulations. It links four Chilean application cohorts to enrollment the following April, tests what happens when applicants go beyond the five recommendations, and gives a displacement bound when two policies list the same acceptable options in different orders. The rental pilot uses archived model outputs. The Chilean offer predictors were frozen after training on 2019 data.

[Read the current paper](who-gets-a-place.pdf) · [Final referee assessment](FINAL_REVIEW.md) · [Anonymous review build](REVIEW.md) · [Search, enrollment and replication](behavior/README.md) · [Earlier referee assessment](REFEREE_AUDIT.md) · [Previous robustness evidence](audit/README.md)

## Read the results

| Question | Aggregate evidence |
| --- | --- |
| Does full-list partner visibility increase while placement falls? | [Rental policy effects and block intervals](evidence/policy-rates.csv) · [School counts and denominators](evidence/school-levels.csv) · [Alternative school benchmarks](evidence/school-benchmarks.csv) |
| Who gained or lost a placement? | [Rental joint transitions](evidence/joint.csv) · [School joint transitions](evidence/school-joint.csv) · [Rental named assignments](evidence/named-partners.csv) · [School named assignments](evidence/school-named.csv) |
| Where can the model trace displacement? | [Number of paths by length](evidence/displacement-paths.csv) · [Exhaustive three-by-three market verifier](analysis/verify_paths.cpp) |
| Could a referral recover a place? | [NYC](evidence/rental-referrals.csv) · [Chile](evidence/school-referrals.csv) · [Chilean display capacity](evidence/school-capacity.csv) |

The rental effect estimates average borough-month rates with frozen ACS weights and equal month weights. Their intervals resample entire simulation blocks. Gross placement and referral counts pool repeated renter instances without geographic reweighting. The school cohorts report exact counts for each deterministic allocation replay. A pupil can appear in successive application years. The two pairs of Chilean cohorts have separate pre-data sign records. The [later-cohort validation plan](robustness/VALIDATION_PLAN.md) is timestamped at Git commit 261e1e7. The [shared benchmark, additional tests and reproduction instructions](robustness/README.md) distinguish the directional validation from exploratory analyses.

## Build the paper

The standalone source archive includes the class, bibliography style, figures, aggregate evidence and a copy of the working paper. With TeX Live installed, run `latexmk -pdf -halt-on-error main.tex` from its top-level directory. In a repository checkout, run the following from this paper's directory to use the archived class and bibliography style in its parent.

```sh
TEXINPUTS=.:..: BSTINPUTS=.:..: latexmk -pdf -halt-on-error main.tex
```

The figures and the aggregate consistency checks require Python, NumPy, pandas and matplotlib. To rebuild the figures from the distributed data, run `python analysis/loss_figure.py` and `python robustness/rollout_figure.py`. To reconcile the independently computed estimates, run `python analysis/verify_tables.py` and `python robustness/verify_robustness.py`. The [referee audit](audit/README.md) has additional verification commands.

## Reproduce the analyses

Start from the [upstream research checkout](https://github.com/ikhillan/CS2880-final-project/tree/f0c4f653159062df31992e96c4f967ed02f9a33d). Install its Python requirements. Set `REPO` to that checkout's absolute path. The rental code uses its three June–August 2026 cleaned listing releases, calibrated Census renter profiles and committed Haiku selections. Data rebuilding instructions are in the [rental pilot](https://github.com/ikhillan/CS2880-final-project/tree/f0c4f653159062df31992e96c4f967ed02f9a33d/findings/claude-haiku-pilot). From this paper's directory run

```sh
mkdir -p replay
python analysis/rental_replay.py --repo "$REPO" --out replay
(cd "$REPO" && python -m experiments.pilot_referrals --output "$REPO/results/rental-referrals-check.csv")
```

Prepare the private school market binaries, masks and partner vectors using the [original 2019 fits and school preparation](https://github.com/ikhillan/CS2880-final-project/tree/f0c4f653159062df31992e96c4f967ed02f9a33d/publication/research/observed/school) and the [2022/23 preparation and export instructions](https://github.com/ikhillan/CS2880-final-project/tree/f0c4f653159062df31992e96c4f967ed02f9a33d/publication/research/policy/equity). Set `NUMERIC` to the resulting private directory. The [original ministry releases](https://datosabiertos.mineduc.cl/sistema-de-admision-escolar-sae/) provide the inputs. The paper distributes aggregate results. The public protocols in the research checkout distinguish the original sign test and the later accepted-offer analysis.

```sh
mkdir -p replay
python analysis/paired_school.py --publication "$REPO/publication" --numeric "$NUMERIC" --out replay
python analysis/school_capacity.py --published "$REPO/publication" --numeric "$NUMERIC" --out replay/school-capacity.csv
python analysis/school_referrals.py --published "$REPO/publication" --numeric "$NUMERIC" --out replay/school-referrals.csv
python analysis/displacement_paths.py --repo "$REPO" --school-reader analysis/paired_school.py --school-numeric "$NUMERIC" --out replay/displacement-paths.csv
c++ -O3 -std=c++17 -I"$REPO/publication/research/observed/school" analysis/school_controls.cpp -o replay/school_controls
c++ -O3 -std=c++17 -I"$REPO/publication/research/observed/school" analysis/school_flow_check.cpp -o replay/school_flow_check
for year in 2022 2023; do
  replay/school_controls "$NUMERIC/$year.bin" "$year" "replay/school-controls-$year.csv"
  replay/school_flow_check "$NUMERIC/$year.bin" "$year" "$NUMERIC" > "replay/school-independent-$year.csv"
done
c++ -O3 -std=c++17 analysis/verify_paths.cpp -o replay/verify_paths
replay/verify_paths > replay/path-exhaustive.json
```

The independent school flow verifier uses a single national graph per cohort. The primary visible-capacity calculation solves disjoint grade graphs and the referral routine uses the programs' residual capacity. The verified maxima agree under both implementations. The enumerator checks stability by direct blocking-pair tests and includes all three-by-three unit markets up to relabeling. These finite checks supplement the analytical proofs in the manuscript.
