#!/bin/bash
# One task of a CPU job array: runs shard $SLURM_ARRAY_TASK_ID of NUM_SHARDS for one config, on one core.
# Submit it with slurm/submit_array.sh (which sets the array range, CONFIG and NUM_SHARDS), or by hand:
#     sbatch --account=YOUR_LAB --array=0-299 --export=ALL,CONFIG=configs/stage3_protection.json,NUM_SHARDS=300 slurm/run_array.sbatch
#
# Defaults, from the measured timings in WIP.md (exact solves about 1 s median at 64 per side, rare 900 s capped
# timeouts; stage 3 about 100-250 core-hours over 1,200 markets, i.e. about 5-13 core-minutes per market on average):
#   -c 1        HiGHS runs one thread (hiddenoptions/audit.py), and the runner pins BLAS/OpenMP to one thread.
#   --mem=2G    one shard peaked at about 220 MB RSS locally with the three development months loaded; held-out
#               configs load a few more months. 2 GB leaves room.
#   -t 0-08:00  at about 4 markets per task (300 shards for stage 3) the typical task takes well under an hour. A single
#               64-per-side market that hits every cap costs at most 6 exposures x 3 exact objectives x 900 s = 4.5 h.
#               A task that runs out of time loses only its unfinished market: rerunning the same array resumes.
#   -p serial_requeue  FASRC's partition for single-core array tasks under a day. Tasks there may be killed and
#               requeued; that is safe because the runner is resumable (finished rows are skipped), and
#               --open-mode=append keeps the requeue messages in the log. Use `shared` or `sapphire` for 3-day limits
#               without preemption.
#SBATCH -J hiddenoptions
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 0-08:00
#SBATCH -p serial_requeue
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH --mail-type=FAIL
## FILL IN (or pass on the command line, as submit_array.sh does): your lab's fairshare account.
##SBATCH --account=YOUR_LAB
set -euo pipefail

: "${CONFIG:?set CONFIG, e.g. --export=ALL,CONFIG=configs/stage3_protection.json,NUM_SHARDS=300}"
: "${NUM_SHARDS:?set NUM_SHARDS to the array size}"
ENV_ROOT="${ENV_ROOT:-/n/holylabs/YOUR_LAB/Lab/$USER/conda}"   # FILL IN: same value as in setup_env.sh
ENV_NAME="${ENV_NAME:-hiddenoptions}"
PYTHON_MODULE="${PYTHON_MODULE:-python}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"   # submit from the repo root
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONUNBUFFERED=1

module load "$PYTHON_MODULE"
export CONDA_ENVS_PATH="$ENV_ROOT/envs" CONDA_PKGS_DIRS="$ENV_ROOT/pkgs"
eval "$(mamba shell hook --shell bash 2>/dev/null || conda shell.bash hook)"
mamba activate "$ENV_NAME"

echo "$(date -Is) host=$(hostname) job=${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-} restart=${SLURM_RESTART_COUNT:-0} config=$CONFIG shard=$SLURM_ARRAY_TASK_ID/$NUM_SHARDS"
# Writes results/<name>/shards/{runs,markets}-<shard>.jsonl; experiments/analyze.py reads those shard files directly.
srun -c 1 python -m experiments.run "$CONFIG" --workers 1 --shard "$SLURM_ARRAY_TASK_ID" --num-shards "$NUM_SHARDS"
echo "$(date -Is) done"
