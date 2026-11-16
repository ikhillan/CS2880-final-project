# Replaying the learned exposure policies

The question is whether a five-listing menu can anticipate competition. A renter who can afford several apartments may prefer an apartment that dozens of other renters also want. The target for learning is the partner the renter would receive in the simulator if every affordable, bedroom-compatible apartment were shown. The market network sees the observable feasible graph. It learns from the simulated assignments, then recommends a shortlist before any evaluation ranking is revealed.

`PROTOCOL.md` records the design and partition/seed map. `partner-certificate.tex` explains why predicting the full-market partner has a placement interpretation and why that interpretation is aggregate. Every learner uses exactly the same simulated training, checkpoint-selection, calibration and test partitions. The supervised fits use only train/validation. The two encoders are trained independently and their three restarts are averaged within architecture; there is no held-test choice of ensemble member. The checkpoint files and full validation traces are under `models/`. `evaluate.py` compares all-renter and higher-income-only rollouts, with and without a maximum-cardinality assignment anchor. H-only rollout computes the candidate policy for the full market and replaces lower-income rows by their original uniform menus. `*-counts.csv.gz` records the two margins, gross losses/gains, all placements, and retained full-market partners. A zero-option renter contributes to the group denominator. `contrasts.csv.gz` reports estimates in percentage points, except for natural-log loss in nats. Each cell rate is averaged across the three months with equal month weight and across four boroughs using the calibration survey shares renormalized to these boroughs. Intervals resample entire simulation seed blocks 5,000 times; they are pointwise conditional simulation intervals. Per-market incidence, the conditional displacement rate among initially matched renters, and the all-lower-income incidence have distinct denominators. Rounding them prematurely can obscure substantive differences in the scarce sample.

## Reproduction from the workspace

Run commands from the project root. The team’s existing generator and six source catalog files must be prepared as described in the paper’s base-data replay. `prepare.py` verifies the known winter manifests and filters identifiable summer/winter listing overlap. Choose an external work directory for the generated arrays. The 64×64 tensors contain synthetic coordinates and no names, addresses or other IDs. Use a recent Python environment with NumPy, SciPy and pandas and a PyTorch environment for training and prediction. Our fits ran on NVIDIA H200s with CUDA 12.8 and PyTorch 2.9.0. One GPU per fit is enough; the six jobs are independent. To reproduce their deterministic CUDA setting, use `CUBLAS_WORKSPACE_CONFIG=:4096:8`. A different GPU type, precision mode or PyTorch build can change floating-point scores slightly.

```bash
src=outputs/ec-ai-revision/learned
data=work/ec-ai-learn
OPENBLAS_NUM_THREADS=1 python -B "$src/prepare.py" --root . --out "$data" --workers 6
python -B outputs/ec-ai-revision/published-baseline/make_masks.py "$data" cal summer winter
# Use the archived models directly for inference, or refit each independently first.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
for model in context pointwise; do
  for seed in 8101 8102 8103; do
    python -B "$src/train.py" train --model "$model" --seed "$seed" --data "$data" --out "$src/models"
  done
done
python -B "$src/train.py" infer --data "$data" --out "$src/models"
OPENBLAS_NUM_THREADS=1 python -B "$src/evaluate.py" --data "$data" --out "$src"
OPENBLAS_NUM_THREADS=1 python -B "$src/summarize.py" --data "$data" --counts "$src"
python -B "$src/audit.py" --root .
python -B "$src/check_certificate.py"
```

Each model receives rent relative to the renter's budget, spare bedrooms, geographic distance, listing quality and the capped income-to-rent screening ratio. Budgets alone already encode income. Training data, network inputs and scores are simulated using one priority process. A model can learn that process and displace renters with lower incomes. Its accuracy or placement rate does not validate the process, establish legal eligibility or measure a real housing outcome. Compare the companion frozen-mask screening sensitivity and individual displacement accounting when interpreting these results.

The extracted paper archive can use different directory names. Pass `--repo research/implementation --winter-script research/ec/winter/run.py --summer-raw nyc-inputs/data/firstmover --winter-raw replay/winter-inputs` to `prepare.py`, in addition to `--root . --out replay/learned`; pass `--repo research/implementation --data replay/learned` to `audit.py`. If the source package places `learned/` elsewhere, set `src` to that directory. To reproduce the published counts exactly using archived predictions, copy `predictions/*-scores.npz` and `receipts/*-scores.json` to the generated tensor directory. For an independent CPU run use `python -B "$src/train.py" infer --device cpu --data "$data" --out "$src/models"`. Put regenerated predictions in a separate tensor directory to preserve the archived GPU results. `cpu_portability.py` compares a summer CPU replay and the archived H200 scores on every feasible log probability, top-five menu and assigned partner.

`ablation.py`, `ablation.sbatch` and `evaluate_ablation.py` implement the subsequent equal-parameter comparison requested during review. Both the contextual and own-row models have 11,906 trainable parameters. Each uses two residual updates; the own-row control supplies the focal edge embedding and the renter's degree where the market model receives the listing's other-renter aggregate and degree. See `MATCHED-ABLATION.md` for the frozen rule. Run `ablation.py train --seed 8101 --data ... --models ...` for each of the three seeds, then `ablation.py infer` with the same data/model options. Run `evaluate_ablation.py --data ... --out ...` in the evaluation environment to rebuild the paired test table. Averages in `ablation-contrasts.csv` use the same month and borough weighting as the main estimates.
