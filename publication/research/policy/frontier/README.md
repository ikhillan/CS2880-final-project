# The price of assignment capacity in Chilean university shortlists

Each applicant gets up to five of their own previously submitted programs. The target is the number of simultaneous feasible placements the display can accommodate. The objective is total predicted probability mass summed over the visible edges of the frozen 2026 PAES offer model. DA operates on those displays only after the design step. The graph, score and tie conventions follow the previous university experiment. See PROTOCOL.md. Different optimal displays need not induce identical DA outcomes.

The scripts require Python 3.11+, NumPy, SciPy, pandas and the OR-Tools graph package (9.15.6755 tested); the DA replay also uses a C++17 compiler. From the extracted archive, rebuild the university panels/scores following research/observed/university/README.md. Use the frozen GPU ensemble if available, or run ensemble.py on all six predictions made with the archived weights and the previous CUDA settings. The inputs below are the private administrative directory, test matrix/slot mapping directory, and offline ensemble directory respectively. Fit files are never needed for the theory proofs.

~~~sh
python research/policy/frontier/solve.py research/observed/university INPUTS FEATURES FIT OUT
python research/policy/frontier/verify_national.py research/observed/university INPUTS FEATURES FIT OUT
python research/policy/frontier/plot.py research/policy/design/referral-ranks.csv referral-rank.pdf
~~~

The integer cost of an anchored edge is its true row-replacement cost rounded to billionths of one applicant's probability. The chosen flow comes with a capacity witness. A separate max-flow/min-cut implementation checks the realized display. verify_national.py reconstructs all eight nonzero target witnesses on the fixed edge ordering and compares each primal objective to a feasible integer dual, derived without reading the min-cost solver's prices. In the source→applicant→program→sink graph, each arc has reduced cost equal to cost plus tail potential minus head potential. The resulting flow-value-times-potential-difference plus the sum of arc capacities times their negative reduced costs is a lower bound on cost. Equality with the integer primal proves optimality. The verification internally compresses two-step residual paths through applicants into program-to-program arcs to find the potentials. The conditional estimation error of the learned probabilities is separate from numerical optimization error.
