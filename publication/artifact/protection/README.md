# End-to-end protected repair on the synthetic core

`protected_results.json` contains the run design, aggregate results and all 960 planning logs. Each log records the returned audit bit mask, verified upper bound, certified lower bound, elapsed wall time and each master call. The two algorithms optimize the same all-proposer protection objective. One enforces the protection thresholds in the stable-support master. The other adds cuts when a DA candidate breaches a threshold. Both use prefix cuts for full-market blocking pairs and a protection-verified reverse-delete warm start. Initialization and final verification are included in the measured planning time.

With Python 3.13.6, NumPy 2.2.6 and SciPy 1.17.1, run from this directory with the archived synthetic `artifact` directory available:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B run_protected.py --archive /path/to/artifact --out replay.json
```

The archive must contain `methodology/audit_solver.py` and `experiments/core/instances/*.json`. The run pins HiGHS to one thread. This experiment used macOS ARM64 and SciPy's HiGHS 1.12.0. The script checks the returned plans with an independent round-based DA implementation. It replays every recorded ordinary minimum-cost mask and, where necessary, enumerates more costly masks in ascending cost order until it finds a protected optimum. It also checks 40 small random markets with incomplete lists, nonuniform costs and potentially unequal sides against a separate exhaustive search. A per-market method-order alternation and three timing repeats reduce order effects. These three repeats are timing replicates; the uncertainty calculation resamples the 20 seed blocks, retaining all eight scenario cells in each block. Timing is specific to this implementation and hardware.
