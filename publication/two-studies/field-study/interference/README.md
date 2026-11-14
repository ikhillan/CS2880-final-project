# Exact enumeration of mixed assignments

Three labeled pupils each choose one of three unit-capacity schools under policy A and one under policy B. Each school has a strict priority order over the three pupils. A school fills its seat with its highest-priority applicant, the outcome of applicant-proposing deferred acceptance with one-item lists. A and B can coincide for a pupil. Every ordered pair of menu profiles and every priority profile appears once, giving 27² × 6³ = 157,464 markets. Each has eight policy-assignment vectors. These are exhaustive combinatorial counts under equal weight on labeled configurations. The calculations contain no Chilean records.

The scalar code checks equality of all three coefficients of the degree-two polynomial for the derivative of expected placements and the sum of direct and spillover effects. It also integrates that polynomial and checks it against the all-B versus all-A placement difference. At assignment probability 1/4, it reports average direct and spillover effects using the common denominator 48 (three pupils and sixteen equally weighted peer draws after expanding the Bernoulli probabilities). In this design, 8,424 markets have a positive direct effect and fewer placements under full adoption. Among the 38,880 markets with fewer placements, the fraction is 13/60. Requiring the two menus to differ for every pupil leaves 5,184 qualifying markets out of 46,656.

## Reproduce

Run the scalar implementation from this folder with Python 3.

```sh
python3 verify_scalar.py scalar-repeat.json
```

Run the separate tensor implementation with PyTorch, NumPy, and a CUDA-capable GPU. The same command accepts `cpu` as the device.

```sh
python3 verify_tensor.py cuda > tensor-repeat.json
python3 -c "import json; a,b=(json.load(open(x)) for x in ['scalar-repeat.json','tensor-repeat.json']); assert all(a[k]==b[k] for k in ('markets','assignments','positive_direct_and_rollout_loss','market_record_sha256'))"
```

The scalar computation awards each seat by scanning its priority order. The tensor computation compares every pair of applicants listing the same school. Both emit a SHA-256 digest of the canonical signed 8-bit records. In each record the first eight bytes are three-bit masks encoding who won for the eight assignment vectors, and the last three bytes encode the full rollout change in seats, 48 times the average direct effect and 48 times the average spillover effect at p=1/4. The outer order is lexicographic on the A profile, B profile and priority profile. Within a market, the treatment bits run from 0 to 7, with pupil zero mapped to the least significant bit.

`scalar.json` records the local CPU calculation. `h200.json` is standard output from Harvard FASRC Slurm job 60435228, which ran on one NVIDIA H200 with PyTorch 2.9.0+cu128 and exited 0 after 71 seconds. Both hashes equal `1e2cbcef47aec046d3bbc79bd72d0601ea83020d0d5c09e8e728817c9d5577f0`.

The paper’s three-pupil, three-school example is reproduced by `python3 interference_example.py --out /tmp/interference-market.csv`. Compare it with the distributed `interference-market.csv`.
