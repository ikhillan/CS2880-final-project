#!/bin/bash
#SBATCH --job-name=observed-offer
#SBATCH --account=kempner_gil_lab
#SBATCH --partition=kempner_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --array=0-5%6
#SBATCH --output=slurm-%A-%a.log
set -euo pipefail
export OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "$HOME/.conda/envs/boltzgen/bin/python" -B train_offer.py "$SLURM_ARRAY_TASK_ID"
