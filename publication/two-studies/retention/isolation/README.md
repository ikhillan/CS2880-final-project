# Isolation of assignment outcomes and ordinary tie-break realizations

The legacy Chilean input format places an official initial course outcome and an ordinary recorded priority vector alongside pre-assignment reports. The institutional sidecar also stores the original ordinary lottery ordering. This audit keeps every other byte unchanged and changes all three fields in fresh private copies. It then regenerates each prospective structural menu and the learned policy's final selection and compares their complete serialized masks to the published SHA-256 receipts. For the learned classifier the original calibration and all 1,024 previously computed design-draw constraints stay fixed for the final selection. To test the construction of those constraints as well, the audit recomputes them from both versions on draws 1–4, 511–514 and 1021–1024 of the pre-specified training panel. The three per-window guard vectors must be byte-identical within each application year.

The official outcome is cyclically relabeled including the outside option. Both ordinary priority vectors are numerically reversed. The school-supplied special PIE and transitional academic ranks, fallback and statutory priorities, family links, capacities, submitted reports, historical prediction scores, and school identities remain fixed. All contemporary April register files are left unopened. The test detects reliance on the perturbed evaluation fields in these input/output comparisons. It does not independently validate when every special-priority field becomes available or whether families' reports would change in response to a different policy.

Use the identical paths file and compiled policy executables as the national replays, and pass the original learned calibration/receipts folder and private 1,024-draw guard files from the design phase. Python 3 and NumPy suffice for the driver.

~~~sh
python3 check_isolation.py --config /data/paths.json --private "$CHECK_SCRATCH" \
  --learned "$LEARNED_RELEASE" --flags "$LEARNED_SCRATCH" \
  --structural-binary "$STRUCTURAL_SCRATCH/structural" --learned-binary "$LEARNED_SCRATCH/retention" \
  --structural-receipt "$STRUCTURAL_RELEASE/frozen-masks.csv" --out "$CHECK_RESULTS/isolation.csv"
~~~

Only the aggregate receipt and code are distributed. Original inputs remain unchanged and the perturbed private copies are deleted on completion.
