#!/bin/bash
#SBATCH --job-name=cs2880-learn
#SBATCH --account=kempner_gil_lab
#SBATCH --partition=kempner_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:25:00
#SBATCH --output=slurm-%A-%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
runtime="$HOME/.conda/envs/boltzgen/bin/python"
if [[ "${INFER:-0}" == 1 ]]; then exec "$runtime" train.py infer; fi
specs=(pointwise context)
exec "$runtime" train.py train --model "${specs[$((SLURM_ARRAY_TASK_ID/3))]}" --seed "$((8101+SLURM_ARRAY_TASK_ID%3))"
