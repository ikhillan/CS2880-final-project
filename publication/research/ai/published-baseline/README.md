# Reproducing two-sided recommendation under congestion

Su, Bayoumi and Joachims (WWW 2022) count successful applications in a two-stage attention model. A person can receive several positive replies; one apartment can be leased once. We first reproduce their reply-count experiment, then test a top-five adaptation inside the apartment simulator. [Published paper, Eq. 10, Fig. 2 and Table 1](https://www.cs.cornell.edu/people/tj/publications/su_etal_22a) · [Authors' code and numerical reference tables, pinned release](https://github.com/bayoumi17m/SW-matching-markets/tree/1d4a34d81b7f4affebfe3efc2a5724b181adbc2f)

## Direct rerun with the authors' solver

We ran the release's PyTorch/CVXPY/SCS Frank–Wolfe optimizer, followed by the unmodified Monte Carlo program. We used random relevance, 20 employers, 30 applicants, inverse-rank examination, crowding 0.5, 50 maximum epochs, ten seeds 621 through 630 and 10,000 simulated markets per policy and seed. Seed 621 is the release default. The authors report a ten-run average without publishing the ten individual market seeds. The final line that serializes the optimized candidate relevances references an undefined name, `cand_exprank_rel`. The wrapper fixes that one reference to `cand_rel` after optimization. It verifies both upstream file hashes before running. The rerun uses newer numerical library versions, listed in `release-requirements.txt`. The archived policy comparison comes from the release README; the conference figure includes an additional LP policy.

| Expected positive replies | Naive | Reciprocal | Welfare |
|---|---:|---:|---:|
| Authors' release reference | 18.7 | 21.2 | 23.2 |
| Our ten seeds, authors' numerical code | 18.411 | 20.864 | 22.772 |
| Our ten seeds, analytical integration and exposure-space solver | 18.398 | 20.859 | 22.752 |

In the direct rerun, the welfare method improved the mean by 1.91 replies over reciprocal ranking. The original numerical values and our ten-run estimates differ by about half a reply or less. `release-mc.csv` contains the 30 per-market estimates and their conditional Monte Carlo errors. The ten market draws, rather than the 300,000 Monte Carlo samples across their policies, are the independent units for generalizing to a new simulated market. Neither a field effect nor the separate dating benchmark was estimated here.

## Crowding and the priority order

We reproduced the 100-employer, 150-applicant crowding sweep on the same ten seeds, using the release's independent uniforms and linear interpolation toward a common ordering. We ranked all jobs for each applicant. The table reports the independent implementation's mean expected replies. The last column applies the priority convention from Eq. 10. The other welfare column follows the released optimizer.

| Crowding | Naive | Reciprocal | Welfare, release order | Welfare, equation order |
|---:|---:|---:|---:|---:|
| 0 | 157.06 | 201.05 | 199.12 | 203.52 |
| 0.25 | 127.70 | 166.64 | 173.67 | 176.86 |
| 0.50 | 106.38 | 129.97 | 152.30 | 154.09 |
| 0.75 | 94.34 | 104.11 | 133.31 | 133.69 |
| 1 | 91.33 | 91.33 | 117.37 | 117.37 |

At crowding 0.5, the release's reference is 106.9, 131.0, 152.7. At crowding 0 the released order loses slightly to reciprocal ranking in both the archived reference and our rerun. For each listing the released optimizer constructs the descending vector of candidate IDs and takes `argsort` of that permutation again. Iterating this inverse permutation can give a different set of higher-priority applicants than the paper defines. The authors' simulation sorts actual applicants by decreasing relevance. We retain the two optimizer interpretations and evaluate every policy with that same simulated interview process. [Released optimizer](https://github.com/bayoumi17m/SW-matching-markets/blob/1d4a34d81b7f4affebfe3efc2a5724b181adbc2f/market_match_sw_optim.py) · [Released Monte Carlo](https://github.com/bayoumi17m/SW-matching-markets/blob/1d4a34d81b7f4affebfe3efc2a5724b181adbc2f/market_match_mc.py)

For inverse-rank examination let $x_{ij}$ denote applicant-side exposure and $p_{ij}=f_{ij}x_{ij}$ the application probability. Employer $j$'s expected interview contribution is
$$
\sum_i g_{ij}p_{ij}\int_0^1\prod_{h\succ_j i}(1-p_{hj}+p_{hj}t)\,dt .
$$
It follows by writing $1/(1+N)=\int_0^1t^N\,dt$ for the number $N$ of higher-priority applicants, with independent applications by those applicants. The integrand has degree at most the number of applicants minus one. Gauss–Legendre quadrature with half the number of applicants rounded up therefore integrates it exactly in real arithmetic. For optimizing the Jensen lower bound, the matrix gradient for a renter is the outer product of their exposure gradient and the position weights. The rearrangement inequality gives the linear maximizer by sorting. This avoids CVXPY and large three-dimensional matrices without changing the linear oracle. Ties, solver tolerance and rounding can still change finite iterations. `checks.json` records gradient, exhaustive integration and permutation-oracle checks.

## Conference networking benchmark

We also used the authors' 925-by-925 matrix of imputed asymmetric relevance scores from a computer science conference. The paper describes a model fitted to participants' interactions. The released artifact contains scores for self-contact; our main comparison removes these pairs by setting the diagonal to zero. We assign each participant to both sides and optimize the order of only the top 100 reciprocal suggestions per sender. The suffix remains in reciprocal order, as specified in the paper. Within the top 100 we initialize uniform exposure over those positions and use the same fixed Frank–Wolfe schedule. We integrated expected positive replies for every receiver with 32 Gauss–Legendre nodes; the largest absolute difference from 16 nodes across all policies and self-edge conventions was 0.00050. [Original Table 1 and network construction](https://www.cs.cornell.edu/people/tj/publications/su_etal_22a) · [Released relevance matrix](https://github.com/bayoumi17m/SW-matching-markets/tree/1d4a34d81b7f4affebfe3efc2a5724b181adbc2f/data/networking)

| Expected positive replies | Naive | Reciprocal | Welfare |
|---|---:|---:|---:|
| Published Table 1 | 604.0 | 763.9 | 824.1 |
| Reimplementation, self-contact excluded | 603.0 | 764.1 | 862.8 |

The inverse-priority implementation yields 850.7 welfare replies. Thus the two controls agree closely and the treatment ranking reproduces, but the optimized magnitude differs by 38.7 replies under the equation convention. The release includes no dedicated end-to-end networking runner. Its tie handling, solver path or additional preprocessing could differ; this comparison cannot identify the source of the discrepancy. Retaining self-pairs gives 825.5, 996.9 and 1102.6, respectively, illustrating why that convention matters. None of these estimates measures realized conference conversations. The input is a model-fitted score matrix and the outcome is model-based expected replies.

## Transfer to apartment menus

The inputs to the housing policy are a feasible-edge matrix and the same two observable scores provided to other policies. We convert renters' observable scores to mid-distribution ranks over feasible listings in each row, and landlords' scores to mid-distribution ranks over feasible applicants in each column. Under a tie all equal values receive their average rank. The formula is (rank minus 0.5) divided by feasible degree; outside edges have zero probability. These are ordinal proxies for bilateral relevance. Neither score was calibrated against applications or landlord decisions.

Let $A_j(i)$ be the applicants a listing ranks above $i$ by its observable score. With equal probability of examining any displayed item and inverse expected rank on the landlord side, the adapted objective is
$$
L(x)=\sum_{ij}\frac{g_{ij} f_{ij}x_{ij}}{1+\sum_{h\in A_j(i)} f_{hj}x_{hj}},
\qquad 0\leq x_{ij}\leq 1,\quad \sum_j x_{ij}=\min(5,\deg(i)).
$$
Infeasible edges have zero exposure. With equal attention inside the displayed set, the row polytope is the convex hull of feasible top-five indicators. We initialize each feasible row uniformly, use 50 Frank–Wolfe steps of size 0.2, then deterministically keep its largest fractional exposures up to the quota. Both axes are independently shuffled using a market-keyed generator before ranking to prevent data order from assigning ties to a particular group. Every resulting row was checked. All 768 calibration, 1,152 summer and 1,152 winter masks were computed before comparing their DA outcomes. The transfer uses an equal-attention shortlist, transformed relevance, fixed rounding and subsequent unit-capacity renter-proposing DA. The Jensen objective remains an interview surrogate. It supplies no guarantee about the DA outcome or subgroup match rates.

## Running the experiments

Run these commands from the project root with the two prepared test catalogs described in the learned policy protocol. The benchmark itself requires no housing data.

```bash
git clone https://github.com/bayoumi17m/SW-matching-markets.git work/su-replication
git -C work/su-replication checkout 1d4a34d81b7f4affebfe3efc2a5724b181adbc2f
python3.11 -m venv work/su-reproduction-env
work/su-reproduction-env/bin/python -m pip install -r outputs/ec-ai-revision/published-baseline/release-requirements.txt
src=outputs/ec-ai-revision/published-baseline
work/su-reproduction-env/bin/python "$src/run_release.py" work/su-replication work/su-release-ten "$src/release-mc.csv"
work/todd-project/.venv/bin/python "$src/figure2.py" "$src"
work/todd-project/.venv/bin/python "$src/networking.py" work/su-replication "$src/networking.json"
work/todd-project/.venv/bin/python "$src/make_masks.py" work/ec-ai-learn cal summer winter
work/todd-project/.venv/bin/python "$src/verify.py" --data work/ec-ai-learn
work/todd-project/.venv/bin/python "$src/screening.py" work/ec-ai-learn "$src"
```

The last command requires scores from the six frozen learned models as well as the prepared panels. The outcome-blind design and scope for its coefficient comparison appear in `SCREENING-PROTOCOL.md`. No address, household record or private interaction log is included in the outputs.

## Screening sensitivity results

Among lower-income renters whose menus stayed uniform, giving the ordinary contextual top-five menus to higher-income renters increased lower-income placement in the enriched markets by 7.72 percentage points in summer and 7.47 in winter under the original screening coefficient. With landlords' income coefficient set to zero at allocation time, the gains were 4.04 and 4.16 points. Average lower-income placement itself rose under both menus when income stopped determining listing priorities. For example, in enriched summer markets the uniform level rose from 36.96% to 51.20%, and the treatment level rose from 44.69% to 55.23%. Under ordinary supply the corresponding estimated spillovers were small and negative. The zero-coefficient intervals span zero in both periods.

The fixed-menu design, primary denominators, eight-block uncertainty and absolute rates appear in `screen-results.tex`. All total, lower- and higher-income estimates under both deployments appear in `screen-summary.csv`; `delta` means difference from uniform in percentage points, and `change` subtracts the corresponding difference at coefficient 2.5. `screen-counts.csv.gz` contains 12,288 market-level counts. The main subgroup estimator is `ratio_of_weighted_counts`; a companion equal-geography standardization described in `SCREENING-ESTIMAND.md` preserves the signs of the four focal spillover point estimates at both coefficients. Use `verify_screen.py .` from the workspace root to audit the first block per season against the team's serial DA implementation and independently constructed listing ranks.
